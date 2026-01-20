import argparse
import sys
sys.path.append('')
from functions.array_training import ArgparseArray, name_instance
argparse_array = ArgparseArray(
random_seed=list(range(50)),
mode=['finetuning'],
alpha_load = [2],
alpha = [2],
n_samples = [10, 20, 50, 100, 200, 500, 1000],
finetune_scaling_last = [0.01,0.1,0.5,1.0,2,5,10],
save_path=name_instance('random_seed', 'mode', 'n_samples', 'finetune_scaling_last', base_folder='/nfs/gatsbystor/cdomine/cifar_testing/cifar/extentions_all_pretrain_fc'),
load_path=(lambda array_id,random_seed, **kwargs: f'/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_fc/random_seed={random_seed}/model.pt'), 
#load_path=(lambda array_id, random_seed, **kwargs: f'/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_fc/random_seed={random_seed}/model.pt'), 
model='resnet',
loss_threshold=0.0001,
alpha_policy=['fr_fc'],
)

def main(args):
    argparse_array.call_script('experiments/cifar/train_cifar100.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)



