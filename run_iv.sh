#!/bin/bash 
#SBATCH --time=70:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --partition=normal
#SBATCH --mail-user=jorge.pena@kit.edu
#SBATCH --job-name=G-IV-CNT_9_0_16-periods_FE-FE-SP

# first non-empty non-comment line ends SLURM options

#module load devel/python/3.7
#module load mpi/impi/2020
module list

#mpiexec --version

#####################
# To Calculate IVsd #
#####################

#python 14_matrices_main.py -ivcb settings/CNTs/9_0_CNT-16periods-CB-SZV.yml > out-cb-iv-CNT-SZV.out
#python 14_matrices_main.py -ivc settings/2TbPc_CNT/16_periods/2-TbPc-Antipar-9_0_CNT-DZVP-SR.yml
#python 14_matrices_main.py -ivcb settings/CNTs/9_0_CNT-16periods-CB-SZV.yml > out-sp-iv_9_0_CNT_SZV.out
python 14_matrices_main.py -ivc settings/2TbPc_CNT/16_periods/CDFT/Ohne_Metall/2-TbPc-FE-FE-9_0_CNT_Metal-Ohne_Metall-DZVP-SR-CDFT-UP-UP_iv.yml > out-sp-iv_9_0_CNT_FE-FE-0p1K.out

date
