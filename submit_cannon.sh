#!/bin/bash

#SBATCH --job-name=larmae
#SBATCH --output=gridlog_larmae.log
#SBATCH --mem-per-gpu=24g
#SBATCH --cpus-per-gpu=4
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=iaifi_gpu
#SBATCH --error=griderr_train_larmae.%j.%N.err

CONTAINER=/n/home01/twongjirad/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
WORKDIR=/n/home01/twongjirad/larmae

singularity exec --nv ${CONTAINER} bash -c "source ${WORKDIR}/run_larmae_training_cannon.sh"


