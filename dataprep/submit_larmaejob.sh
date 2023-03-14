#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=larmaedata
#SBATCH --output=larmaedata_test.log
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --exclude=c1cmp003,c1cmp004
#SBATCH --error=err_logs/griderr_make_larmaedata.%j.%a.%N.err
#SBATCH --array=1-856

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
DATA_PREP_DIR=/cluster/tufts/wongjiradlabnu/twongj01/larmae/dataprep/

module load singularity/3.5.3
cd /cluster/tufts/

# mcc9_v13_bnb_nu_corsika: 34254 files, strid 40, [0-856 jobs]
srun singularity exec ${container} bash -c "cd ${DATA_PREP_DIR} && source run_make_mlrecodata_mcc9_v13_bnb_nu_corsika.sh"


