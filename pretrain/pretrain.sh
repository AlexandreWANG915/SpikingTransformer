#!/bin/sh
#BSUB -J cifar1004reluscracth      
#BSUB -n 4                
#BSUB -q gpu              
#BSUB -gpgpu 1
#BSUB -o out1.txt
#BSUB -e err1.txt             

source /hpc/jhinno/unischeduler/exec/unisched

module load cuda-11.1
module load anaconda3
source activate pytorch 

/hpc/softwares/anaconda3/bin/python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 train_multi_gpu_using_launch.py --lr=0.0001