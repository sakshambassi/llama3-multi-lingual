#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=llama3

module purge

singularity exec --nv \
            --overlay overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python test-llama3-alternate-layers.py"
