import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    random_seed=list(range(50)),
    mode='pretrain',
    save_path=name_instance('random_seed','alpha', base_folder='/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_all/'),
    model='resnet',
    alpha_policy=['first_vs_rest'],
    alpha=[0.01,0.1,0.5,1.0,2,5]
)

def main(args):
    argparse_array.call_script('experiments/cifar/train_cifar100.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
