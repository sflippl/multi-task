import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
import pathlib
import os
import time
import copy
from tqdm import tqdm

## Define the transformer
class ViT(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )


    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

def build_dataloaders(train_set, test_set, train_indices, test_indices, batch_size=128):
    """ Returns dataloaders and dataset_sizes """
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, 
            sampler=SubsetRandomSampler(train_indices)
        ),
        'valid': torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, 
            sampler=SubsetRandomSampler(test_indices)
        )
    }
    dataset_sizes = {
        'train': len(train_indices),
        'valid': len(test_indices)
    }
    return dataloaders, dataset_sizes

def data_to_numpy(train_set, test_set, train_indices, test_indices):
    batch_size=max(len(train_indices), len(test_indices))
    dataloaders, dataset_sizes = build_dataloaders(train_set, test_set, train_indices, test_indices, batch_size)

    for inputs, labels in dataloaders['train']:
        X_train = inputs.data.numpy()
        y_train = labels.data.numpy()
        
    for inputs, labels in dataloaders['valid']:
        X_valid = inputs.data.numpy()
        y_valid = labels.data.numpy()
    return X_train, y_train, X_valid, y_valid

def apply_relative_scaling(model, alpha: float, model_name: str, policy: str = "first_vs_rest", 
                         finetune_scaling_last: float = 1,layer:int=0 ):
    """
    Adjust the relative scale of model layers based on the specified policy.
    Args:
        model: The neural network model
        alpha: Scaling factor
        model_name: Type of model ('resnet' or 'vit')
        policy: Scaling policy ('first_vs_rest', 'constant', 'linear')
        k: Number of layers to apply scaling to (for constant/linear policies)
        prevent_weight_increase: If True, skip scaling when it would increase weights
    """
    print('alpha:', alpha)
    print('policy:', policy)
    if alpha != 1.0:
        print('got in')
        if policy == "first_vs_rest":
            _apply_first_vs_rest_scaling(model, alpha, model_name)
        elif policy == "fr_gamma_alpha_fc":
             _apply_first_vs_rest_gamma_alpha_fc(model, alpha, model_name,finetune_scaling_last)
        elif  policy == "fr_fc":
            _apply_fc(model, alpha, model_name,finetune_scaling_last)
        elif policy == "fr_gamma_alpha":
            _apply_first_vs_rest_gamma_alpha(model, alpha, model_name)
        elif  policy == "fr_gamma":
            _apply_first_vs_rest_gamma(model, alpha, model_name)
        elif  policy == "fr_block_1":
            _apply_first_vs_rest_block_1(model, alpha, model_name)
        elif  policy == "fr_block_12":
            _apply_first_vs_rest_block_12(model, alpha, model_name)
        elif  policy == "fr_block_123":
            _apply_first_vs_rest_block_123(model, alpha, model_name)
        elif  policy == "fr_block_1234":
            _apply_first_vs_rest_block_1234(model, alpha, model_name)
        elif policy == "fr_block_12alpha":
             _apply_first_vs_rest_block_12alpha(model, alpha, model_name)
        elif policy == "fr_block_vit":
            _scale_layer_weights_vit(model, layer, alpha, model_name)
        else:
            raise ValueError(f"Unknown policy: {policy}")


def _apply_first_vs_rest_scaling(model, alpha: float, model_name: str):
    """Original scaling behavior: scale first layer only."""
    with torch.no_grad():
        if model_name == "resnet":
            # First layer = conv1
            model.conv1.weight.div_(alpha)
            if model.conv1.bias is not None:
                model.conv1.bias.div_(alpha)
            # ResNet has BatchNorm -> no readout scaling     param.data = param.data * args.finetune_scaling
        elif model_name == "vit":
            # First layer = patch embedding
            model.emb.weight.data.div_(alpha)
            if model.emb.bias is not None:
                model.emb.bias.data.div_(alpha)
            # Also scale learned tokens
            if getattr(model, "cls_token", None) is not None:
                model.cls_token.data.div_(alpha)
            model.pos_emb.data.div_(alpha)
            # ViT has LayerNorm -> no readout scaling
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
    
def _apply_first_vs_rest_gamma_alpha_fc(model, alpha: float, model_name: str, finetune_scaling_last: float = 1.0):
    """
    Scales the first layer (weights/bias), the first BatchNorm (gamma),
    and applies optional finetune scaling to the readout.
    """
    with torch.no_grad():
        if model_name == "resnet":
            # 1. Scale First Conv Layer
            model.conv1.weight.div_(alpha)
            if model.conv1.bias is not None:
                model.conv1.bias.div_(alpha)
            # 2. Scale First BatchNorm Gamma (Weight)
            # This ensures the scaling isn't "reset" by normalization
            if hasattr(model, 'bn1'):
                model.bn1.weight.div_(alpha)
            # 3. Scale Readout Layer (fc)
            # Applying the finetune_scaling to the final classifier
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.mul_(finetune_scaling_last)

def _apply_first_vs_rest_gamma_alpha(model, alpha: float, model_name: str):
    """
    Scales the first layer (weights/bias), the first BatchNorm (gamma),
    and applies optional finetune scaling to the readout.
    """
    with torch.no_grad():
        if model_name == "resnet":
            # 1. Scale First Conv Layer
            model.conv1.weight.div_(alpha)
            if model.conv1.bias is not None:
                model.conv1.bias.div_(alpha)
            # 2. Scale First BatchNorm Gamma (Weight)
            # This ensures the scaling isn't "reset" by normalization
            if hasattr(model, 'bn1'):
                model.bn1.weight.div_(alpha)
                # Note: Scaling bn1.bias (beta) is optional depending on


def _apply_first_vs_rest_gamma(model, alpha: float, model_name: str):
    with torch.no_grad():
         if model_name == "resnet":
            if hasattr(model, 'bn1'):
                model.bn1.weight.div_(alpha)
                # Note: Scaling bn1.bias (beta) is optional depending on
                # if you want to shift the threshold or just scale the signal.
                # model.bn1.bias.div_(alpha)

def _apply_first_vs_rest_block_1(model, alpha: float, model_name: str): 
  # Calculate the scale factor
    scale = 1.0 / alpha
    with torch.no_grad():
         if model_name == "resnet":
            # Option A: BLOCK (The first block)
            for name, param in model.layer1.named_parameters():
                param.copy_(param * scale)
            # Option A: BLOCK (The first block) and second 
           
def _apply_first_vs_rest_block_12alpha(model, alpha: float, model_name: str): 
    scale = 1.0 / alpha
    with torch.no_grad():
            if model_name == "resnet":
                #Scale First Conv Layer
                model.conv1.weight.div_(alpha)
                if model.conv1.bias is not None:
                    model.conv1.bias.div_(alpha)
                if hasattr(model, 'bn1'):
                    model.bn1.weight.div_(alpha)
                for name, param in model.layer1.named_parameters():
                    param.copy_(param * scale)        
                for name, param in model.layer2.named_parameters():
                    param.copy_(param * scale)       
      
def _apply_first_vs_rest_block_12(model, alpha: float, model_name: str): 
  # Calculate the scale factor
    scale = 1.0 / alpha
    with torch.no_grad():
         if model_name == "resnet":
            # Option A: BLOCK (The first block)
            for name, param in model.layer1.named_parameters():
                param.copy_(param * scale)
            # Option A: BLOCK (The first block) and second 
            for name, param in model.layer2.named_parameters():
                param.copy_(param * scale)
            
def _apply_first_vs_rest_block_123(model, alpha: float, model_name: str): 
  # Calculate the scale factor
    scale = 1.0 / alpha
    with torch.no_grad():
        if model_name == "resnet":
            # Option A: BLOCK (The first block)
            for name, param in model.layer1.named_parameters():
                param.copy_(param * scale)
            # Option A: BLOCK (The first block) and second 
            for name, param in model.layer2.named_parameters():
                param.copy_(param * scale)
            for name, param in model.layer3.named_parameters():
                param.copy_(param * scale)

def _apply_first_vs_rest_block_1234(model, alpha: float, model_name: str): 
  # Calculate the scale factor
    scale = 1.0 / alpha
    with torch.no_grad():
        if model_name == "resnet":
            # Option A: BLOCK (The first block)
            for name, param in model.layer1.named_parameters():
                param.copy_(param * scale)
            # Option A: BLOCK (The first block) and second 
            for name, param in model.layer2.named_parameters():
                param.copy_(param * scale)
            for name, param in model.layer3.named_parameters():
                param.copy_(param * scale)
            for name, param in model.layer4.named_parameters():
                param.copy_(param * scale)
                
def _apply_fc(model, alpha: float, model_name: str, finetune_scaling_last: float = 1.0):
    """
    Scales the first layer (weights/bias), the first BatchNorm (gamma),
    and applies optional finetune scaling to the readout.
    """
    with torch.no_grad():
        if model_name == "resnet":
            # Applying the finetune_scaling to the final classifier
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.mul_(finetune_scaling_last)

def scale_layer_weights_vit(model, layer_idx:int ,alpha:float ,model_name: str ):
    """
    Directly scales the parameter values of a specific encoder layer.
    
    Args:
        model: Your ViT instance.
        layer_idx: The index of the encoder layer (0 for the 1st layer).
        scale: The multiplier (e.g., 2.0 to upscale values, 0.5 to downscale).
    """
    # Access the specific TransformerEncoder block
    scale = 1.0 / alpha
    target_layer = model.enc[layer_idx]
    
    print(f"Scaling parameters in encoder layer {layer_idx} by factor: {scale}")
    
    with torch.no_grad():
        if model_name == "vit":
            for name, param in target_layer.named_parameters():
                # In-place multiplication
                param.mul_(scale)
            
    return model
 
def train_model(model, X_train, y_train, X_train_aux, y_train_aux, criterion, optimizer, device,
                batch_size=128, batch_size_aux = 128, X_valid=None, y_valid=None, X_valid_aux=None, y_valid_aux=None,
                n_epochs=25, loss_thresh=0.0, aux_scale=1.0, model2=None):
    since = time.time()
    losses = {'train': [], 'train_aux': [], 'valid': [], 'valid_aux': []}
    accs = {'train': [], 'train_aux': [], 'valid': [], 'valid_aux': []}
    
    # building data loaders
    dataloaders = {}
    dataset_sizes = {}
    phases = []
    if X_train is not None and y_train is not None:
        dataset_sizes['train'] = X_train.shape[0]
        X_train = torch.Tensor(X_train)
        y_train = torch.LongTensor(y_train)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), 
            batch_size=batch_size, shuffle=True,
        )
        dataloaders['train'] = train_loader
        phases.append('train')
    
    if X_train_aux is not None and y_train_aux is not None:
        dataset_sizes['train_aux'] = X_train_aux.shape[0]
        X_train_aux = torch.Tensor(X_train_aux)
        y_train_aux = torch.LongTensor(y_train_aux)
        train_loader_aux = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_aux, y_train_aux), 
            batch_size=batch_size_aux, shuffle=True,
        )
        dataloaders['train_aux'] = train_loader_aux
        phases.append('train_aux')

    if X_valid is not None and y_valid is not None:
        dataset_sizes['valid'] = X_valid.shape[0]
        X_valid = torch.Tensor(X_valid)
        y_valid = torch.LongTensor(y_valid)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_valid, y_valid), 
            batch_size=batch_size, shuffle=True,
        )
       

        dataloaders['valid'] = val_loader
        phases.append('valid')

    if X_valid_aux is not None and y_valid_aux is not None:
        dataset_sizes['valid_aux'] = X_valid_aux.shape[0]
        X_valid_aux = torch.Tensor(X_valid_aux)
        y_valid_aux = torch.LongTensor(y_valid_aux)
        val_loader_aux = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_valid_aux, y_valid_aux), 
            batch_size=batch_size, shuffle=True,
        )
       

        dataloaders['valid_aux'] = val_loader_aux
        phases.append('valid_aux')


    stop_next = False
    for epoch in (range(1, n_epochs+1)):
        
        if stop_next:
            break
        print('Epoch {}/{}'.format(epoch, n_epochs))

        # each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train' or phase == 'train_aux':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0
            # iterate over data
            for inputs, labels in (dataloaders[phase]):
                # inputs = inputs.view(-1, 28*28)
              
                inputs = inputs.to(device)
                int_labels = labels.to(device)
                #labels = int_labels#.float()
                labels = int_labels#.float()

                # zero the parameter gradients
                

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train' or phase == 'train_aux'):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    if model2 is not None:
                        outputs = outputs + model2(inputs)
                    if phase in ['train_aux', 'valid_aux']:
                        outputs = outputs[:, 2:]
                    else:
                        outputs = outputs[:, :2]
                    _, preds = torch.max(outputs, 1)
                    #preds = torch.relu(torch.sign(2*outputs-1))#
                    
                    loss = criterion(outputs, labels)
                    if phase == 'train_aux':
                        loss = loss * aux_scale

                    # backward + optimize only if in training phase
                    if phase == 'train' or phase == 'train_aux':
                        loss.backward()
                        optimizer.step()
                        

                # statistics
                running_loss = running_loss + loss.item() * inputs.size(0)
                running_corrects = running_corrects + torch.sum(preds == int_labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            #if phase == phases[len(phases)-2]:
            #    optimizer.step()
            if phase == "train":
                if epoch_loss < loss_thresh:
                    stop_next = True
            if phase == "train_aux":
                if X_train is None and epoch_loss < loss_thresh:
                    stop_next = True
            print(' - {:5s} loss: {:.4f} acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


    return model, losses, accs

def main(args):
    print("Creating save directory...")
    print("Save path:", args.save_path)
    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform)

    test_set = datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    randperm = np.random.permutation(100)

    for t in range(len(train_set.targets)):
        train_set.targets[t] = randperm[train_set.targets[t]]
    for t in range(len(test_set.targets)):
        test_set.targets[t] = randperm[test_set.targets[t]]

    train_digits_04 = np.where(np.array(train_set.targets) < 2)[0]
    train_digits_59 = np.where(np.logical_and(np.array(train_set.targets) > 1, np.array(train_set.targets) > 1))[0]

    test_digits_04 = np.where(np.array(test_set.targets) < 2)[0]
    test_digits_59 = np.where(np.logical_and(np.array(test_set.targets) > 1,  np.array(test_set.targets) > 1))[0]

    (len(train_digits_04), len(test_digits_04)), (len(train_digits_59), len(test_digits_59))

    X_train_04, y_train_04, X_valid_04, y_valid_04 = data_to_numpy(train_set, test_set, train_digits_04, test_digits_04)


    X_train_59, y_train_59, X_valid_59, y_valid_59 = data_to_numpy(train_set, test_set, train_digits_59, test_digits_59)

    # fixing the issues with labels
    y_train_59 = y_train_59 - 2
    y_valid_59 = y_valid_59 - 2
    subsample_aux = np.random.choice(len(X_train_59), size=(args.n_aux_samples,), replace=False)
    subsample = np.random.choice(len(X_train_04), size=(args.n_samples,), replace=False)

    if args.model == 'resnet':
        model = torchvision.models.resnet18(pretrained=False, num_classes=100)
    if args.model == 'vit':
        model = ViT(num_classes=100)
   
    if args.mode == 'finetuning':
        model.load_state_dict(torch.load(args.load_path))
        for param in model.parameters():
            param.data = param.data * args.finetune_scaling
            
    apply_relative_scaling(model, args.alpha, args.model, args.alpha_policy, args.finetune_scaling_last
                         )
    
    if args.alpha != 1.0:
        if args.model == "resnet":
            print("Conv1 norm after scaling:", model.conv1.weight.norm().item())
        elif args.model == "vit":
            print("Embedding norm after scaling:", model.emb.weight.norm().item())

    print("Moving model to device and setting up training...")
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    if args.model == 'resnet':
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    if args.model == 'vit':
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        
    X_train_aux = X_train_59[subsample_aux]
    y_train_aux = y_train_59[subsample_aux]
    if args.mode == 'pretrain':
        model, losses, accs = train_model(
            model, 
            None, None,#X_train_04[subsample], y_train_04[subsample].reshape(-1, 1), 
            X_train_59[subsample_aux], y_train_59[subsample_aux],#.reshape(-1, 1), 
            criterion, optimizer, 
            device=args.device,
            X_valid_aux=X_valid_59, y_valid_aux=y_valid_59,#.reshape(-1, 1), 
            batch_size=min(args.n_samples, 128), batch_size_aux = 128,
            n_epochs=args.epochs, loss_thresh = args.loss_threshold
        )
    if args.mode == 'multitask':
        model, losses, accs = train_model(
            model, 
            X_train_04[subsample], y_train_04[subsample],#.reshape(-1, 1), 
            X_train_aux, y_train_aux, 
            criterion, optimizer, 
            device=args.device,
            X_valid=X_valid_04, y_valid=y_valid_04,#.reshape(-1, 1), 
            batch_size=min(args.n_samples, 128), batch_size_aux = 128,
            n_epochs=args.epochs, loss_thresh = args.loss_threshold,
            aux_scale=(1.0 if args.mode != 'multitask' else float(args.n_samples/min(args.n_samples, 128))/float(args.n_aux_samples/128))
        )
    if args.mode in ['singletask', 'finetuning']:
        model, losses, accs = train_model(
            model, 
            X_train_04[subsample], y_train_04[subsample],#.reshape(-1, 1), 
            None, None, 
            criterion, optimizer, 
            device=args.device,
            X_valid=X_valid_04, y_valid=y_valid_04,#.reshape(-1, 1), 
            batch_size=min(args.n_samples, 128), batch_size_aux = 128,
            n_epochs=args.epochs, loss_thresh = args.loss_threshold,
            aux_scale=(1.0 if args.mode != 'multitask' else float(args.n_samples/min(args.n_samples, 128))/float(args.n_aux_samples/128))
        )
    losses = {key: np.array(value) for key, value in losses.items()}
    accs = {key: np.array(value) for key, value in accs.items()}
    print("Saving results...")
    print("Results will be saved to:", args.save_path)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pt'))
    np.save(os.path.join(args.save_path, 'losses.npy'), losses)
    np.save(os.path.join(args.save_path, 'accs.npy'), accs)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_aux_samples', type=int, default=49000)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--mode', choices=['pretrain', 'multitask', 'singletask', 'finetuning'], required=True)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--finetune_scaling', type=float, default=1.0)
    parser.add_argument('--finetune_scaling_last', type=float, default=1.0)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--loss_threshold', type=float, default=1e-2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', choices=['vit', 'resnet'], default='resnet')
    parser.add_argument('--layers', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Relative scaling factor (first vs rest).')
    parser.add_argument('--alpha_load', type=float, default=1.0,
                        help='Relative scaling factor (first vs rest).')
    parser.add_argument('--alpha_policy', type=str, default='first_vs_rest',
                        choices=['first_vs_rest', 'all_except_output','fr_gamma_alpha_fc', 'fr_gamma_alpha','fr_gamma','fr_block_1','fr_block_12','fr_block_123','fr_block_1234','fr_block_12alpha','fr_fc','fr_block_vit'],
                        help='Which parts of the model to scale at init.')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
