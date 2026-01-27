import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    random_seed=list(range(30)),
    aux_n_samples=[10, 20, 50, 100, 200, 500, 1000],
    alpha_load = [0.1, 0.5, 1.0, 2, 5, 10], 
    scaling= [1.0],
    save_path=name_instance('random_seed', 'n_samples', 'alpha_load', base_folder='data/cifar/ensd_123alpha/'),
    load_path_pre=(lambda random_seed, alpha_load, **kwargs: f'/nfs/scistore19/locatgrp/cdomine/multi-task/data/cifar/pretrain_base/random_seed={random_seed}--model=resnet--alpha={alpha_load}/model.pt'), 
    load_path_post=(lambda random_seed, n_samples, alpha_load, **kwargs: f'/nfs/scistore19/locatgrp/cdomine/multi-task/data/cifar/finetune_123alpha/random_seed={random_seed}--alpha_load={alpha_load}--n_samples={n_samples}--model=resnet/model.pt'),
    model='resnet',
    device='cuda',
)
    # load_path_pre=(lambda random_seed, alpha_load, **kwargs: f'/nfs/scistore19/locatgrp/cdomine/multi-task/data/cifar/pretrain_base/random_seed={random_seed}--alpha_policy=scaling_pre_train--alpha={alpha_load}/model.pt'), 
    #load_path_pre=(lambda alpha_load,random_seed, **kwargs: f'/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_block_123/random_seed={random_seed}--alpha_policy=fr_block_123--alpha={alpha_load}/model.pt'), 
    #load_path_pre=(lambda array_id,random_seed, alpha_load, **kwargs: f'/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_block_12vit_saxe/random_seed={random_seed}--alpha={alpha_load}/model.pt'), 
    #load_path_post=(lambda random_seed, n_samples,alpha_load, **kwargs: f'/nfs/gatsbystor/cdomine/cifar_testing/cifar/extentions_all_pretrain_block_123/random_seed={random_seed}--mode=finetuning--n_samples={n_samples}--alpha_load={alpha_load}/model.pt'),
    #load_path_post=(lambda random_seed, n_samples, alpha_load, **kwargs: f'/nfs/gatsbystor/cdomine/cifar_testing/cifar/extentions_fr_block_12vit_saxe/random_seed={random_seed}--mode=finetuning--alpha_load={alpha_load}--n_samples={n_samples}/model.pt'),
    # f"random_seed={seed}--mode=finetuning--alpha_load={alpha}--n_samples={n_sample}"
    #load_path_pre=(lambda array_id, random_seed, **kwargs: f'data/cifar/pretrain/random_seed={random_seed}/model.pt'),
    #load_path_post=(lambda array_id, random_seed, n_samples, scaling, **kwargs: f'data/cifar/main/random_seed={random_seed}--mode=finetuning--n_samples={n_samples}--finetune_scaling={scaling}/model.pt'),
    
def main(args):
    argparse_array.call_script('experiments/cifar/ensd_sam.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)

