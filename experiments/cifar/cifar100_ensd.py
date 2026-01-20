import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

argparse_array = ArgparseArray(
random_seed=list(range(50)),
aux_n_samples=[10, 20, 50, 100, 200, 500, 1000],
alpha_load =  [1.0],
alpha = [1.0],
alpha_policy = ['fr_fc'],
scaling= [0.3,0.5,1.0,2,5,10],
save_path=name_instance('random_seed', 'n_samples', 'scaling' , base_folder='data/cifar/ensd_vit_base/'),
load_path_pre=(lambda array_id,  random_seed, **kwargs: f'/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_block_1vit/random_seed={random_seed}/model.pt'),
load_path_post=(lambda array_id, random_seed, n_samples, scaling, **kwargs: f'/nfs/gatsbystor/cdomine/cifar_testing/cifar/extentions_all_pretrain_fr_block_1vit_base/random_seed={random_seed}--mode=finetuning--n_samples={n_samples}--finetune_scaling={scaling}/model.pt'),
#load_path_pre=(lambda array_id, random_seed, **kwargs: f'/nfs/nhome/live/cdomine/old_sam/multi-task/data/cifar/pretrain_fr_fc/random_seed={random_seed}/model.pt'),
#load_path_post=(lambda array_id, random_seed, n_samples, finetune_scaling_last, **kwargs: f'/nfs/gatsbystor/cdomine/cifar_testing/cifar/extentions_all_pretrain_fc/random_seed={random_seed}--mode=finetuning--n_samples={n_samples}--finetune_scaling_last={finetune_scaling_last}/model.pt'),
model='vit',
device='cpu',
)

def main(args):
    argparse_array.call_script('experiments/cifar/ensd.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
