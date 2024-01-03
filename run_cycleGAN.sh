#!/bin/bash
#SBATCH  --output=./LOGS/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
#SBATCH --constraint='geforce_rtx_2080_ti'

# Source the environment and activate conda
source /itet-stor/bochen/net_scratch/conda/etc/profile.d/conda.sh
conda activate cyclegan

python train.py --dataroot /usr/bmicnas02/data-biwi-01/contrastive_dg/data/da_data/brain --srcroot /usr/bmicnas02/data-biwi-01/contrastive_dg/data/da_data/brain/hcp2/images --tgtroot /usr/bmicnas02/data-biwi-01/contrastive_dg/data/da_data/brain/hcp1/images --name hcp2_to_hcp1 --model cycle_gan --dataset_mode unaligned --input_nc 1 --output_nc 1 --batch_size 1 --save_epoch_freq 10 --load_size 256 --crop_size 256 --n_epochs 100 --n_epochs_decay 100 --use_wandb

# python test.py --dataroot /cluster/work/cvl/bochen/Project_DG/mri/abide_caltech/images/testA --name cal2hp --model test --no_dropout --input_nc 1 --output_nc 1 --load_size 256 --crop_size 256