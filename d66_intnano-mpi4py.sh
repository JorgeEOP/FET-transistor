#!/bin/bash 
#SBATCH --time=02:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=5
#SBATCH --partition=short
#SBATCH --job-name=Toy

# first non-empty non-comment line ends SLURM options

#module load devel/python/3.7
module load mpi/impi/2020
#module load mpi/openmpi/4.0

export SCRATCH=/scratch
cd ${SLURM_SUBMIT_DIR}

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

#srun -l --propagate=STACK,CORE --cpu_bind=cores --distribution=block:cyclic \
#        --mpi=pmi2 python 14_matrices_main.py -scpf \
#        settings/2TbPc_CNT/16_periods/CDFT/Ohne_Metall/2-TbPc-FE-FE-9_0_CNT_Metal-Ohne_Metall-DZVP-SR-CDFT-UP-UP_Vg0.yml > out-sp_FE-FE_gated-Vg0.out

time srun python 14_matrices_main.py -scpf \
        toy_settings_d66/FE-AF/Toy_FE-AF.yml > out-sp_Toy_d66_FE-AF.out
time srun python 14_matrices_main.py -scpf \
        toy_settings_d66/FE-FE/Toy_FE-FE.yml > out-sp_Toy_d66_FE-FE.out

#mpiexec -ppn 3 python -m mpi4py 14_matrices_main.py -scp settings/TbPc_CNT/9periods-TbPc_CNT12_0-001-FE-DZVP.yml 
#mpiexec -ppn 4 python -m mpi4py 14_matrices_main.py -scp settings/2TbPc_CNT/2-TbPc-AF-AF-12_0_CNT-DZVP-d1.yml
