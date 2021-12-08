#!/bin/bash 
#SBATCH --time=02:59:00
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --tasks-per-node=90
#SBATCH --job-name=G-IV_toy

# first non-empty non-comment line ends SLURM options

#module load devel/python/3.7
#module load mpi/impi/2020
#module list

export SCRATCH=/scratch
cd ${SLURM_SUBMIT_DIR}

#mpiexec --version

#####################
# To Calculate IVsd #
#####################

#python 14_matrices_main.py -ivcb settings/CNTs/9_0_CNT-16periods-CB-SZV.yml > out-cb-iv-CNT-SZV.out
#python 14_matrices_main.py -ivc settings/2TbPc_CNT/16_periods/2-TbPc-Antipar-9_0_CNT-DZVP-SR.yml
#python 14_matrices_main.py -ivcb settings/CNTs/9_0_CNT-16periods-CB-SZV.yml > out-sp-iv_9_0_CNT_SZV.out

time srun python3 14_matrices_main.py -ivchi toy_settings_d65/FE-AF/\
Toy_FE-AF.yml > out-sp_Toy_d65_FE-AF_iv.out
time srun python3 14_matrices_main.py -ivchi toy_settings_d65/FE-FE/\
Toy_FE-FE.yml > out-sp_Toy_d65_FE-FE_iv.out

date
