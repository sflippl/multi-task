import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    random_seed=list(range(30)),
    mode='pretrain',
    alpha_policy=['fr_block_vit'], #fr_gamma_alpha_fc, fr_gamma_alpha, fr_gamma, first_vs_rest, fr_block
    model='vit',
    #save_path=name_instance('random_seed','model', base_folder='/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_fc/')
    save_path=name_instance('random_seed','model','alpha', base_folder='/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_block_12vit_saxe/'),
    alpha=[0.3,0.5,1.0,2,5,10],
    layers=[[0, 1]]
)

def main(args):
    argparse_array.call_script('experiments/cifar/train_cifar100.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
