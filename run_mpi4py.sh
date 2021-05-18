#!/bin/bash 
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --partition=develop
#SBATCH --mail-user=jorge.pena@kit.edu
#SBATCH --job-name=Toy_modell_FE-FE

# first non-empty non-comment line ends SLURM options

#module load devel/python/3.7
module load mpi/impi/2020
#module load mpi/openmpi/4.0
module list

#mpiexec --version

##################################################
# To Calculate Transmission: Parallel simulation #
##################################################

##############################################################################
# "mpiexec" hat viele verschiedene Unbestaendigkeit mit intel MPI. Man kann  # 
# vielleicht mit OpenMPI sein/ihre Glueck probieren                          #
##############################################################################
#mpiexec -n 2 python 14_matrices_main.py -cbscp settings/2TbPc_CNT/16_periods/Kontrol/Kontroll-CB.yml > out-p.out

#####################################################################################
# Das hier wird nur einzige Prozess in -N Knoten herstellen. Das heist dass man wird #
# das gleichenes Code in 2 Knoten laufen.                                           #
#####################################################################################
#srun -n $SLURM_NTASKS python 14_matrices_main.py -cbscp settings/2TbPc_CNT/16_periods/Kontrol/Kontroll-CB.yml > out-p.out

############################################################################
# "Richtige" Form um eines parallel Code zu laufen.                        #
# Achtung auf die Moeglichkeiten von srun Kommando. Sie fuehren            #
# interessante Sachen durch.                                               #
# doc: https://hpc-uit.readthedocs.io/en/latest/jobs/running_mpi_jobs.html #
# -> Wichtig ist, dass das SchluesselWoert --ntask-per-node=1 dabei ist
#     anders mpi nimmt mehr ranks zu arbeiten
############################################################################
#srun -l --propagate=STACK --mpi=pmi2 python 14_matrices_main.py -cbscp \

srun -l --propagate=STACK,CORE --cpu_bind=cores --distribution=block:cyclic \
        --mpi=pmi2 python 14_matrices_main.py -scpf \
        settings/TB-chains/Toy_modell/FE-AF/Toy_FE-AF.yml > out-sp_Toy_modell_FE-AF.out
srun -l --propagate=STACK,CORE --cpu_bind=cores --distribution=block:cyclic \
        --mpi=pmi2 python 14_matrices_main.py -scpf \
        settings/TB-chains/Toy_modell/FE-FE/Toy_FE-FE.yml > out-sp_Toy_modell_FE-FE.out

#mpiexec -ppn 3 python -m mpi4py 14_matrices_main.py -scp settings/TbPc_CNT/9periods-TbPc_CNT12_0-001-FE-DZVP.yml 
#mpiexec -ppn 4 python -m mpi4py 14_matrices_main.py -scp settings/2TbPc_CNT/2-TbPc-AF-AF-12_0_CNT-DZVP-d1.yml

date
