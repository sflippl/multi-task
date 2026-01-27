
import argparse
import sys
sys.path.append('')
from functions.array_training import ArgparseArray, name_instance
argparse_array = ArgparseArray(
random_seed=list(range(30)),
mode=['finetuning'],
alpha = [1],
alpha_load = [0.1,0.5,1.0,2,5,10],
n_samples = [10, 20, 50, 100, 200, 500, 1000],
finetune_scaling = [1.0],
save_path=name_instance('random_seed', 'model', 'n_samples', 'alpha_load' ,base_folder='/nfs/scistore19/locatgrp/cdomine/multi-task/data/cifar/finetune_123alpha/'),
load_path=(lambda random_seed, alpha_load, **kwargs: f'/nfs/scistore19/locatgrp/cdomine/multi-task/data/cifar/pretrain_base/random_seed={random_seed}--model=resnet--alpha={alpha_load}/model.pt'), 
#load_path=(lambda array_id, random_seed, **kwargs: f'/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_fc/random_seed={random_seed}/model.pt'), 
model= ['resnet'],
loss_threshold=0.0001,
alpha_policy=['scaling_pre_train'],
)

def main(args):
    argparse_array.call_script('experiments/cifar/train_cifar100.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)




