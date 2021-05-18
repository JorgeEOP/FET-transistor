#!/bin/bash 
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --mail-user=jorge.pena@kit.edu
#SBATCH --job-name=CNT_SZV

# first non-empty non-comment line ends SLURM options

#module load devel/python/3.7
#module load mpi/impi/2018
module list

########################
# To generate matrices #
########################

#python 14_matrices_main.py -bCP2K settings/TB-chains/H-Doppel-Kette-cb.yml > out-s.out
#python 14_matrices_main.py -bCP2K settings/TB-chains/H-Kette-cb.yml > out-s.out
python 14_matrices_main.py -bCP2K settings/CNTs/9_0_CNT-16periods-CB-SZV.yml > out-s.out
#python 14_matrices_main.py -bCP2K settings/2TbPc_CNT/16_periods/Kontrol/Kontroll-CB-s.yml > out-s.out
#python 14_matrices_main.py -bCP2K settings/2TbPc_CNT/16_periods/CDFT/Ohne_Metall/2-TbPc-FE-FE-9_0_CNT_Metal-Ohne_Metall-DZVP-SR-CDFT-UP-UP.yml > out-generate-FE-FE-ohne_metall.out
#python 14_matrices_main.py -bCP2K settings/2TbPc_CNT/16_periods/CDFT/Ohne_Metall/2-TbPc-Antipar-9_0_CNT_Metal-Ohne_Metall-DZVP-SR-CDFT-up-down.yml > out-generate-Antipar-ohne_metall.out

date
