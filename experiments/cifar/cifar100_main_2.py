import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
    random_seed=list(range(50)),
    mode=['finetuning'],
    alpha_load = [0.01,0.1,0.5,1.0,2,5],
    n_samples=[10, 20, 50, 100, 200, 500, 1000],
    #finetune_scaling=[0.125, 0.25, 0.5, 1., 2.],
    save_path=name_instance('random_seed', 'mode', 'n_samples', 'alpha_load', base_folder='/nfs/gatsbystor/cdomine/cifar_testing/cifar/extentions_all'),
    load_path=(lambda array_id,  alpha_load,random_seed, **kwargs: f'/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_all/random_seed={random_seed}--alpha={alpha_load}/model.pt'),
    model='resnet',
    alpha_policy=['first_vs_rest'],
)

def main(args):
    argparse_array.call_script('experiments/cifar/train_cifar100.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
