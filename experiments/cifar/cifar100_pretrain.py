import argparse
import sys
sys.path.append('')
from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
random_seed=list(range(50)),
mode='pretrain',
alpha_policy=['fr_block_123_alpha'], #fr_gamma_alpha_fc, fr_gamma_alpha, fr_gamma, first_vs_rest, fr_block
model=['resnet'],
save_path=name_instance('random_seed','model','alpha', base_folder='/nfs/scistore19/locatgrp/cdomine/multi-task/data/cifar/pretrain_base/'),
alpha=[0.1,0.5,1.0,2,5,10],
)

def main(args):
    argparse_array.call_script('experiments/cifar/train_cifar100.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
main(args)
