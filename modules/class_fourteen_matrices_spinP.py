import generate_matrices.class_systemdef as sdef
import modules.mo_print as moprint
import matplotlib.pyplot as plt
import scipy.constants as phys
import modules.mo as mo
import dask.array as da
import pandas as pd
import scipy.linalg
import modules.GC as GC
import numpy as np
import h5py
import time 
import sys
import os
from tempfile import TemporaryDirectory
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from pandas import DataFrame as df
from tabulate import tabulate
from math import ceil,floor
from mpi4py import MPI
from modules.OTfor import otmod


class fourteen_matrices_spin:
    # Parallel version of the serial fourteen matrices
    def __init__(self,config):
        self.Sname    = config["System name"]
        self.NE       = config["Number of energy points"]
        self.Ea       = config["Lower energy border"]
        self.Eb       = config["Upper energy border"]
        self.path_in  = config["Path to the system 14-matrices"]
        self.eta      = config["Small imaginary part"]
        self.path_out = config["Path of output"]

        try:
            self.restart_calc = config["Restart calculation"]
        except KeyError:
            print ("No restart keyword found.")
            self.restart_calc  = "No"
        if self.restart_calc   in ["Yes","yes","Y","y"]:
            self.restart_file  = config["Restart file"]
        elif self.restart_calc in ["No","no","no","n"]:
            pass
        try:
            self.gate_correction = \
                               config["Gate correction to central Hamiltonian"]
        except KeyError:
            print ("Gating not applied to central Hamiltonian")
            self.gate_correction = "No"

        try:
            self.pDOS   = config["Projected DOS"]
        except KeyError:
            print ("No key value Projected DOS. Set to NO")
            self.pDOS   = "No"
        if self.pDOS in ["Yes","yes","Y","y"]:
            self.mol1    = config["Molecule 1"]
            self.mol2    = config["Molecule 2"]
            self.is1atom = config["System1 first atom"]
            self.fs1atom = config["System1 last atom"]
            self.is2atom = config["System2 first atom"]
            self.fs2atom = config["System2 last atom"]
        elif self.pDOS in ["No","no","N","n"]:
            pass

        try:
            self.CB_regime = config["Coulomb blockade"]
        except KeyError:
            print ("No key value Coulomb blockade. Set to NO")
            self.CB_regime = "No"
        if self.CB_regime in ["Yes","yes","Y","y"]:
            try:
                self.U_cb      = config["Coulomb energy in eVs"]
                self.scaling_a = config["Density matrix re-scaling (alpha)"]
                self.scaling_b = config["Density matrix re-scaling (beta)"] 
            except KeyError:
                print ("No Coulomb energy or re-scaling factor"
                       "for Coulomb blockade found.")
                self.U_cb = 0
                self.scaling_a = 1.
                self.scaling_b = 1.
        else:
            self.U_cb = 0
            self.scaling_a = 1.
            self.scaling_b = 1.

        self.NVgs = config["Number of gs-voltage points"]
        self.Vg_a = config["Lower gate-source voltage"]
        self.Vg_b = config["Upper gate-source voltage"]
        
        self.structure = sdef.SystemDef(config)

    def load_electrodes(self,spin):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each.
        HL  = np.loadtxt(self.path_in+"/HL-"+spin+".dat")
        SL  = np.loadtxt(self.path_in+"/SL.dat")
        
        HR  = np.loadtxt(self.path_in+"/HR-"+spin+".dat")
        SR  = np.loadtxt(self.path_in+"/SR.dat")
        
        VCL = np.loadtxt(self.path_in+"/VCL-"+spin+".dat")
        SCL = np.loadtxt(self.path_in+"/SCL.dat")
        
        VCR = np.loadtxt(self.path_in+"/VCR-"+spin+".dat")
        SCR = np.loadtxt(self.path_in+"/SCR.dat")
        
        TL  = np.loadtxt(self.path_in+"/TL-"+spin+".dat")
        STL = np.loadtxt(self.path_in+"/STL.dat")
        
        TR  = np.loadtxt(self.path_in+"/TR-"+spin+".dat")
        STR = np.loadtxt(self.path_in+"/STR.dat")

        return HL,SL,HR,SR,VCL,SCL,VCR,SCR,TL,STL,TR,STR
    
    def load_center(self,spin):
        # Load the matrices representing the quantum region
        HC = np.loadtxt(self.path_in+"/HC-"+spin+".dat")
        SC = np.loadtxt(self.path_in+"/SC.dat")
        return HC,SC

    def load_matrices_h5(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each.
        H_all = h5py.File(self.path_in+"/H_all.h5", "r")
        S_all = h5py.File(self.path_in+"/S_all.h5", "r")
        PC    = h5py.File(self.path_in+"/PC.h5", "r")
        return H_all, S_all, PC
   
    def load_FermiE(self):
        # Load the chemical potential
        Ef = np.loadtxt(self.path_in+"/Ef")
        return Ef

    def NEGF(self):
        # Parallel Version of NEGF method.

        startT = time.time()

        path = self.path_out

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        print (size)
        all_extra_ranks = [i+1 for i in range(size-1)]
        print (all_extra_ranks)

        Energies_p_rank = []

        if rank == 0:
            info00 = ("NEGF+DFT. Parallel calculation. UKS. \nSystem name: {}")
            print (moprint.ibordered_title(info00.format(self.Sname)),\
                   flush=True,end='\n')

            name = MPI.Get_processor_name()
            
            #info01 = ("Name of the Processor(Node): {} ; Process: {}")
            #print (moprint.ibordered(info01.format(name, rank,\
            #       Energies_p_rank)), flush=True, end='\n')

            if not os.path.exists(path):
                os.makedirs(path)

            # Init Energy range
            Energies = np.linspace(self.Ea, self.Eb, num=self.NE, dtype=complex)

            perrank, resperrank = divmod(Energies.size, size)
            counts = [perrank + 1 if p < resperrank else perrank for p in\
                      range(size)]

            starts = [sum(counts[:p])   for p in range(size)]
            ends   = [sum(counts[:p+1]) for p in range(size)]

            Energies_p_rank = [Energies[starts[p]:ends[p]] for p in range(size)]
            

        else:
            name     = MPI.Get_processor_name()
            Energies = None
            #info01   = ("Name of the Processor(Node): {} ; Process: {}")
            #print (moprint.ibordered(info01.format(name, rank,\
            #       Energies_p_rank)), flush=True, end='\n')

        Energies_p_rank = comm.scatter(Energies_p_rank, root=0)

        for spin in ["alpha","beta"]: 
            Vgates = np.linspace(self.Vg_a, self.Vg_b, num=self.NVgs)

            H_all, S_all, PC_all = self.load_matrices_h5()

            PC_all = None

            HL  = np.array(H_all.get("HL-"+spin))
            SL  = np.array(S_all.get("SL"))
            HR  = np.array(H_all.get("HR-"+spin))
            SR  = np.array(S_all.get("SR"))
            VCL = np.array(H_all.get("VCL-"+spin))
            SCL = np.array(S_all.get("SCL"))
            VCR = np.array(H_all.get("VCR-"+spin))
            SCR = np.array(S_all.get("SCR"))
            TL  = np.array(H_all.get("TL-"+spin))
            STL = np.array(S_all.get("STL"))
            TR  = np.array(H_all.get("TR-"+spin))
            STR = np.array(S_all.get("STR"))
            HC  = np.array(H_all.get("HC-"+spin))
            SC  = np.array(S_all.get("SC"))
            
            Ef       = self.load_FermiE()
            Ef_alpha = Ef[0]
            Ef_beta  = Ef[1]
            Ef       = (Ef_alpha + Ef_beta)/2

            dimC = HC.shape[0]

            if self.pDOS in ["No","no","N","n"]:
                if rank == 0:
                    if spin == "alpha":
                        info02 = ("Average Fermi E. (eVs): {: 4.5f} eV\n"\
                                  "Fermi E. alpha/beta (eVs): {: 4.5f} / {:4.5f}")
                        print (moprint.ibordered(info02.format(Ef, Ef_alpha,\
                                                 Ef_beta)), flush=True, \
                                                 end='\n')
                    else:
                        pass
                else:
                    pass

                if self.gate_correction in ["Yes", "yes", "Y", "y"]:
                    with open(path+"/out-"+spin+"-sp-g.out", "w+") as f1:
                        if rank == 0:
                            if spin == "alpha":
                                print ("Gatting applied to central Hamiltonian")
                            else:
                                pass
                        else:
                            pass
            
                        for ivg, Vg in enumerate(Vgates):
                            if rank == 0:
                                info03 = ("Gate Voltage index: {: >23d}\n" 
                                          "Gate Voltage (V): {: >33.8f}\n")
                                print (info03.format(ivg, Vg), flush=True, \
                                       end='\n')
                            else:
                                pass

                            #Vgm_HL  = SL  * Vg   
                            #Vgm_HR  = SR  * Vg 
                            Vgm_VCL = SCL * Vg 
                            Vgm_VCR = SCR * Vg 
                            #Vgm_TL  = STL * Vg 
                            #Vgm_TR  = STR * Vg 
                            Vgm_HC  = SC * Vg

                            #HL  = HL  - Vgm_HL
                            #HR  = HR  - Vgm_HR
                            VCLg = VCL - Vgm_VCL
                            VCRg = VCR - Vgm_VCR
                            #TL  = TL  - Vgm_TL
                            #TR  = TR  - Vgm_TR
                            HCg = HC - Vgm_HC
                        
                            dos   = np.zeros(self.NE)
                            trans = np.zeros(self.NE)
                            rank_string = str(rank)

                            dos, trans = \
                                    self.dos_t_calculation_sp(Energies_p_rank,\
                                                              Ef, dimC, HL, TL,\
                                                              SL, STL, HR, TR, \
                                                              SR, STR, SCL, \
                                                              VCLg, SCR, VCRg, \
                                                              SC, HCg)

                            if rank == 0:
                                dosT = {}
                                dosT[0] = np.array(list(zip(dos, trans)))

                                for idosT, irank in enumerate(all_extra_ranks):
                                    dosT[irank] = comm.recv(source=irank, tag=irank)

                                comm.Barrier()

                                for i in dosT.keys():
                                    if i == 0:
                                        dost2print = dosT[i]
                                    else:
                                        dost2print = np.concatenate((dost2print,\
                                                              np.array(dosT[i])))
                                
                                vg2print = Vg * np.ones((self.NE))
                                dost2print = df(data=dost2print, \
                                                columns=["Dos(E)", "T(E)"])
                                dost2print.insert(loc=0, column="Gate voltage", \
                                                  value=vg2print.T)
                                dost2print.insert(loc=1, column='Energy', \
                                                  value=Energies.real)

                                fmt_01 = "% -05.7f", "% -05.7f", "% -05.7f", \
                                         "% -05.7f"
                                np.savetxt(f1, dost2print, delimiter='  ', fmt=fmt_01)

                                f1.write("\n")
                                f1.flush()
                                print (moprint.iprint_line(), flush=True, end='\n')
                                print ("\n", flush=True, end='\n')

                                comm.Barrier()

                            else:
                                comm.send(np.array(list(zip(dos, trans))), \
                                          dest=0, tag=rank)
                                comm.Barrier()
                                comm.Barrier()

                        if spin == "alpha":
                            stopH    = time.time()
                            tempH    = stopH-startT
                            hoursH   = tempH//3600
                            minutesH = tempH//60 - hoursH*60
                            secondsH = tempH - 60*minutesH
                            info04 = ("Half time for NEGF+DFT method "
                                      "(up(alpha) spin): "
                                      "{:.0f}:{:.0f}:{:.0f} h/m/s")
                            print (moprint.ibordered(info04.\
                                   format(hoursH, minutesH, secondsH)),\
                                   flush=True, end='\n')
                            print (moprint.iprint_line(), flush=True, end='\n')

                        elif spin == "beta":
                            # Check memory stats
                            memoryHC = HC.size*HC.itemsize
                            shapeHC  = HC.shape
                            sizeofHC = sys.getsizeof(HC)
                            print ("Size / itemsize / shape / "
                                   "sys.getsizeof(Kb) / "
                                   "Memory(Kb) of matrix to invert: "
                                   "{:} / {:} / {:} / {:} / {:} \n".\
                                   format(HC.size, HC.itemsize, shapeHC,\
                                   sizeofHC/1000, memoryHC/1000))               

                            stop    = time.time()
                            temp    = stop-startT
                            hours   = temp//3600
                            minutes = temp//60 - hours*60
                            seconds = temp - 60*minutes

                            info05 = ("Entire time for NEGF+DFT method "
                                      "(up(alpha) and down(beta) spin): "
                                      "{:.0f}:{:.0f}:{:.0f} h/m/s")
                            print (moprint.ibordered(info05.\
                                   format(hours, minutes, seconds)),\
                                   flush=True, end='\n')

                elif self.gate_correction in ["No", "no", "N", "n"]:
                    if rank == 0:
                        f1 = open(path+"/out-"+spin+"-sp-ng.out", "w+")
                        if spin == "alpha":
                            print ("Gatting not applied to central Hamiltonian")
                        else:
                            pass
                    else:
                        pass

                    #Vgm_HL  = SL  * Vg   
                    #Vgm_HR  = SR  * Vg 
                    #Vgm_VCL = SCL * Vg 
                    #Vgm_VCR = SCR * Vg 
                    #Vgm_TL  = STL * Vg 
                    #Vgm_TR  = STR * Vg 
                    #Vgm_HC  = SC  * Vg

                    #HL  = HL  - Vgm_HL
                    #HR  = HR  - Vgm_HR
                    #VCL = VCL - Vgm_VCL
                    #VCR = VCR - Vgm_VCR
                    #TL  = TL  - Vgm_TL
                    #TR  = TR  - Vgm_TR
                    #HC  = HC  - Vgm_HC
                    
                    dos   = np.zeros(self.NE)
                    trans = np.zeros(self.NE)
                    rank_string = str(rank)

                    dos, trans = self.dos_t_calculation_sp(Energies_p_rank,\
                                                           Ef, dimC, HL, TL,\
                                                           SL, STL, HR, TR, \
                                                           SR, STR, SCL, \
                                                           VCL, SCR, VCR, \
                                                           SC, HC)
                    if rank == 0:
                        dosT = {}
                        dosT[0] = np.array(list(zip(dos, trans)))
                        print (dosT)

                        for idosT, irank in enumerate(all_extra_ranks):
                            dosT[irank] = comm.recv(source=irank, tag=irank)
                        print(dosT)

                        comm.Barrier()

                        for i in dosT.keys():
                            if i == 0:
                                dost2print = dosT[i]
                            else:
                                dost2print = np.concatenate((dost2print,\
                                                            np.array(dosT[i])))
                        
                        dost2print = df(data=dost2print, \
                                        columns=["Dos(E)", "T(E)"])
                        dost2print.insert(loc=0, column="Energy", \
                                          value=Energies.real)
                        fmt_01 = "% -05.7f", "% -05.7f", "% -05.7f"
                        np.savetxt(f1, dost2print, delimiter="  ", fmt=fmt_01)

                        f1.write("\n")
                        f1.flush()
                        print (moprint.iprint_line(), flush=True, end='\n')
                        print ("\n", flush=True, end='\n')

                        comm.Barrier()

                    elif rank != 0:
                        comm.send(np.array(list(zip(dos, trans))), \
                                  dest=0, tag=rank)
                        comm.Barrier()
                        comm.Barrier()

                    if spin == "alpha":
                        stopH    = time.time()
                        tempH    = stopH-startT
                        hoursH   = tempH//3600
                        minutesH = tempH//60 - hoursH*60
                        secondsH = tempH - 60*minutesH
                        info04 = ("Half time for NEGF+DFT method "
                                  "(up(alpha) spin): "
                                  "{:.0f}:{:.0f}:{:.0f} h/m/s")
                        print (moprint.ibordered(info04.\
                               format(hoursH, minutesH, secondsH)),\
                               flush=True, end='\n')
                        print (moprint.iprint_line(), flush=True, end='\n')

                    elif spin == "beta":
                        # Check memory stats
                        memoryHC = HC.size*HC.itemsize
                        shapeHC  = HC.shape
                        sizeofHC = sys.getsizeof(HC)
                        print ("Size / itemsize / shape / "
                               "sys.getsizeof(Kb) / "
                               "Memory(Kb) of matrix to invert: "
                               "{:} / {:} / {:} / {:} / {:} \n".\
                               format(HC.size, HC.itemsize, shapeHC,\
                               sizeofHC/1000, memoryHC/1000))               

                        stop    = time.time()
                        temp    = stop-startT
                        hours   = temp//3600
                        minutes = temp//60 - hours*60
                        seconds = temp - 60*minutes

                        info05 = ("Entire time for NEGF+DFT method "
                                  "(up(alpha) and down(beta) spin): "
                                  "{:.0f}:{:.0f}:{:.0f} h/m/s")
                        print (moprint.ibordered(info05.\
                               format(hours, minutes, seconds)),\
                               flush=True, end='\n')

                else:
                    pass

            elif self.pDOS in ["Yes","yes","Y","y"]:
                print ("Parallel PDOS-simulation not implemented")
                sys.exit()

    def NEGF_CB(self):
        # Parallel Version of the serial NEGF_CB method.
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        all_extra_ranks = [i+1 for i in range(size-1)]
        Energies_p_rank = []

        if rank == 0:
            name = MPI.Get_processor_name()

            Energies = np.linspace(self.Ea, self.Eb, num=self.NE, dtype=complex)

            perrank, resperrank = divmod(Energies.size,size)
            counts = [perrank + 1 if p < resperrank else perrank for p in\
                      range(size)]

            starts = [sum(counts[:p])   for p in range(size)]
            ends   = [sum(counts[:p+1]) for p in range(size)]

            Energies_p_rank = [Energies[starts[p]:ends[p]] for p in range(size)]

        else:
            name     = MPI.Get_processor_name()
            Energies = None

        Energies_p_rank = comm.scatter(Energies_p_rank, root=0)
        
        if rank == 0:
            startT = time.time()
            info00 = ("Coulomb Blockade NEGF+DFT. Parallel calculation"
                      "\nSystem name: {}")
            print (moprint.ibordered_title(info00.format(self.Sname)),\
                   flush=True, end='\n')

            # Load the matrices with h5py.File()
            H_all, S_all, PC_all = self.load_matrices_h5()

            spin = 'alpha'
            comm.send('beta', dest=1, tag=111)
            
            #print ('0 barrier', flush=True, end='\n')
            comm.Barrier()

            #print ('... Fortfahren (0)!!! Der Allvater segne dich!',\
            #       flush=True)

            HC = np.array(H_all.get("HC-"+spin))
            SC = np.array(S_all.get("SC"))
            PC = np.array(PC_all.get("PC-"+spin))

            if size > 2:
                pass
            elif size <= 2:
                Xsmatrix = self.orthogonalization_basis(SC)
                print ('Done with X matrix', flush=True, end='\n')

            Ef       = self.load_FermiE()
            Ef_alpha = Ef[0]
            Ef_beta  = Ef[1]
            Ef       = (Ef_alpha + Ef_beta)/2
            SC = SC * self.scaling_a

            N_part_alpha_init, N_homos_alpha_init, occ_homlum_alpha_init,\
            HC_alpha_init, vHCg_alpha_init, HC_alpha_diag_init =\
            self.energy_levs_shifted(HC, SC, PC, 0, 0)

            info01 = ("Average Fermi energy: {: 4.5f} eV\nFermi energy "
                      "alpha/beta: {: 4.5f} eV / {:4.5f} eV")
            print (moprint.ibordered(info01.format(Ef, Ef_alpha, Ef_beta)),\
                   flush=True, end='\n')

            N_part_beta_init     = comm.recv(source=1, tag=11)
            N_homos_beta_init    = comm.recv(source=1, tag=12)
            occ_homlum_beta_init = comm.recv(source=1, tag=13)
            HC_beta_init         = comm.recv(source=1, tag=14)
            vHCg_beta_init       = comm.recv(source=1, tag=15)
            HC_beta_diag_init    = comm.recv(source=1, tag=16)

            info02 = ("Total number of {:4s} electrons (Tr(P * S)): "\
                      "{: 7.3f}\n"
                      "Scaling factor for alpha density matrix: {}")
            print (moprint.ibordered(info02.format(spin, N_part_alpha_init,\
                   self.scaling_a)), flush=True, end='\n')

            #print ('1 barrier', flush=True, end='\n')
            comm.Barrier()
            #print ('... Fortfahren (1)!', flush=True, end='\n')

            if size > 2:
                Xsmatrix = comm.recv(source=2, tag=222)
            elif size <= 2:
                pass

            occ_homlum_a_b_init =\
            pd.concat([occ_homlum_alpha_init, occ_homlum_beta_init],\
                      axis=1, ignore_index=True)
            occ_homlum_a_b_init.columns = ["occ_alpha",\
                                           "energy_levels_alpha",\
                                           "occ_beta",\
                                           "energy_levels_beta"]

            print ("\nInitial energy levels per spin and their occupation"\
                   " numbers", flush=True)
            print (tabulate(occ_homlum_a_b_init, headers="keys", \
                   tablefmt="fancy_grid"), flush=True)

            # Fuer Test: Fermi level ist gleich als 0 (Vsd=0)
            Ef_central = max(occ_homlum_alpha_init['energy_levels'].\
                             iloc[N_homos_alpha_init-1],\
                             occ_homlum_beta_init['energy_levels'].\
                             iloc[N_homos_beta_init-1])
            print ("Fermi energy Vsd in equilibrium set in:",\
                   Ef_central, "eVs")

            for spin in ["alpha","beta"]: 
                Vgates = np.linspace(self.Vg_a, self.Vg_b, num=self.NVgs)

                path = self.path_out
                if not os.path.exists(path):
                    os.makedirs(path)
            
                Vg_step = 0

                Ng_homos_alpha = {}
                Ng_lumos_alpha = {}
                Ng_homos_beta  = {}
                Ng_lumos_beta  = {}

                HCg_alpha = HC_alpha_init
                
                str_rank = str(rank)
                f1 = open(path+"/out-"+spin+"-cb.out", "w+")

                if spin == 'alpha':
                    debf = open(path+"/energy-levels.out", "w+")
                else:
                    pass

                N_homos_alpha_vg = N_homos_alpha_init
                N_homos_beta_vg  = N_homos_beta_init

                for ivg, Vg in enumerate(Vgates):
                    startVg = time.time()
        
                    extra_e_alpha = 0
                    extra_e_beta  = 0

                    if ivg == 1:
                        Vg_step = Vg
                    else:
                        pass

                    if ivg == 0:
                        #print ('2 barrier', flush=True, end='\n')
                        comm.Barrier()
                        #print ('... Fortfahren (2)!', flush=True, end='\n')
                    else:
                        pass

                    comm.send(ivg,      dest=1, tag=211)
                    comm.send(Vg_step,  dest=1, tag=212)
                    comm.send(Xsmatrix, dest=1, tag=221)

                    #print ('3 barrier', flush=True, end='\n')
                    comm.Barrier()
                    #print ('... Fortfahren (3)!', flush=True, end='\n')

                    N_part_alpha, N_homos_alpha, occ_homlum_alpha,\
                    HCg_alpha, vHCg_ortho_alpha, HCg_alpha_diag = \
                    self.energy_levs_shiftedXmat(HCg_alpha, SC, PC,\
                                                 Vg_step, 0, Xsmatrix)

                    N_part_beta     = comm.recv(source=1, tag=213)
                    N_homos_beta    = comm.recv(source=1, tag=214)
                    occ_homlum_beta = comm.recv(source=1, tag=215)
                    HCg_beta        = comm.recv(source=1, tag=216)
                    vHCg_ortho_beta = comm.recv(source=1, tag=217)
                    HCg_beta_diag   = comm.recv(source=1, tag=218)

                    #print ('4 barrier', flush=True, end='\n')
                    comm.Barrier()
                    #print ('... Fortfahren (4)!', flush=True, end='\n')

                    dimC  = HCg_alpha.shape[0]

                    occ_homlum_g_a_b =\
                    pd.concat([occ_homlum_alpha, occ_homlum_beta],\
                              axis=1, ignore_index=True)
                    occ_homlum_g_a_b.columns = ["occ_alpha",\
                                                "energy_levels_alpha",\
                                                "occ_beta",\
                                                "energy_levels_beta"]
                    homos_alpha = []
                    lumos_alpha = []
                    homos_beta  = []
                    lumos_beta  = []

                    for ihomlum in occ_homlum_g_a_b.itertuples():
                        if ihomlum.energy_levels_alpha <= Ef_central:
                            homos_alpha.append(ihomlum.energy_levels_alpha)
                        elif ihomlum.energy_levels_alpha > Ef_central:
                            lumos_alpha.append(ihomlum.energy_levels_alpha)
                        if ihomlum.energy_levels_beta <= Ef_central:
                            homos_beta.append(ihomlum.energy_levels_beta)
                        elif ihomlum.energy_levels_beta > Ef_central:
                            lumos_beta.append(ihomlum.energy_levels_beta)

                    Ng_homos_alpha[ivg] = len(homos_alpha)
                    # Was zu tun wenn homos alpha leer ist:
                    if len(homos_alpha) == 0:
                        homos_alpha = [-30]
                        Ng_homos_alpha[ivg] = len(homos_alpha)
                    Ng_lumos_alpha[ivg] = len(lumos_alpha)
                    # Was zu tun wenn lumos alpha leer ist:
                    if len(lumos_alpha) == 0:
                        lumos_alpha = [30]
                        Ng_lumos_alpha[ivg] = len(lumos_alpha)

                    Ng_homos_beta[ivg] = len(homos_beta)
                    # Was zu tun wenn homos beta leer ist:
                    if len(homos_beta) == 0:
                        homos_beta = [-30]
                        Ng_homos_beta[ivg] = len(homos_beta)
                    Ng_lumos_beta[ivg] = len(lumos_beta)
                    # Was zu tun wenn lumos beta leer ist:
                    if len(lumos_beta) == 0:
                        lumos_beta = [30]
                        Ng_lumos_beta[ivg] = len(lumos_beta)

                    occ_hom_alpha = pd.concat([occ_homlum_g_a_b['occ_alpha'], \
                                              df(homos_alpha)], axis=1, \
                                              ignore_index=True)
                    occ_hom_alpha.columns = (["occ", "energy"])
                    occ_hom_alpha = (occ_hom_alpha.dropna()).assign(occ=1)

                    occ_lum_alpha = pd.concat([occ_homlum_g_a_b['occ_alpha'], \
                                              df(lumos_alpha)], axis=1, \
                                              ignore_index=True)
                    occ_lum_alpha.columns = (["occ", "energy"])
                    occ_lum_alpha = (occ_lum_alpha.dropna()).assign(occ=0)

                    occ_hom_beta = pd.concat([occ_homlum_g_a_b['occ_beta'], \
                                             df(homos_beta)], axis=1, \
                                             ignore_index=True)
                    occ_hom_beta.columns = (["occ", "energy"])
                    occ_hom_beta = (occ_hom_beta.dropna()).assign(occ=1)

                    occ_lum_beta = pd.concat([occ_homlum_g_a_b['occ_beta'], \
                                             df(lumos_beta)], axis=1, \
                                             ignore_index=True)
                    occ_lum_beta.columns = (["occ", "energy"])
                    occ_lum_beta = (occ_lum_beta.dropna()).assign(occ=0)

                    if ivg == 0:
                        extra_e_alpha = 0
                        extra_e_beta  = 0
                    elif ivg != 0:
                        if abs(Ng_homos_alpha[ivg] - N_homos_alpha_vg) == 0:
                            extra_e_alpha += 0
                        elif abs(Ng_homos_alpha[ivg] - N_homos_alpha_vg) != 0:
                            extra_e_alpha += Ng_homos_alpha[ivg] - \
                                             N_homos_alpha_vg
                            N_homos_alpha_vg = occ_hom_alpha.energy.size
                        if abs(Ng_homos_beta[ivg] - N_homos_beta_vg) == 0:
                            extra_e_beta += 0
                        elif abs(Ng_homos_beta[ivg] - N_homos_beta_vg) != 0:
                            extra_e_beta += Ng_homos_beta[ivg] - \
                                            N_homos_beta_vg
                            N_homos_beta_vg  = occ_hom_beta.energy.size

                    if abs(N_homos_alpha_vg - N_homos_alpha_init) == 0 and \
                         abs(N_homos_beta_vg - N_homos_beta_init) == 0:
                        U_cb = 0
                    elif abs(N_homos_alpha_vg - N_homos_alpha_init) != 0 or \
                           abs(N_homos_beta_vg - N_homos_beta_init) != 0:
                        U_cb = self.U_cb

                    Umod_alpha = np.zeros((dimC,dimC))
                    Umod_beta  = np.zeros((dimC,dimC))
                    
                    if extra_e_alpha == 0 and extra_e_beta == 0:
                        pass
                    elif extra_e_alpha != 0 or extra_e_beta != 0:
                        if extra_e_alpha != 0 and extra_e_beta == 0:
                            for ienea in range(occ_hom_alpha.energy.size, dimC):
                                Umod_alpha[ienea][ienea] += U_cb 
                            for ieneb in range(0,dimC):
                                if HCg_beta_diag[ieneb][ieneb] >\
                                                occ_hom_alpha['energy'].iloc[-1]:
                                    Umod_beta[ieneb][ieneb] += U_cb
                                else:
                                    pass
                        if extra_e_alpha == 0 and extra_e_beta != 0:
                            for ieneb in range(occ_hom_beta.energy.size, dimC):
                                Umod_beta[ieneb][ieneb] += U_cb 
                            for ienea in range(0,dimC):
                                if HCg_alpha_diag[ienea][ienea] >\
                                                 occ_hom_beta['energy'].iloc[-1]:
                                    Umod_alpha[ienea][ienea] += U_cb
                                else:
                                    pass
                    if extra_e_alpha != 0 or extra_e_beta != 0:
                        HCg_U_alpha, HCg_U_alpha_diag, vHCg_U_ortho_alpha = \
                        self.energy_levs_HCg_pU(HCg_alpha, SC, Xsmatrix,\
                                                Umod_alpha)

                        HCg_U_beta, HCg_U_beta_diag, vHCg_U_ortho_beta = \
                        self.energy_levs_HCg_pU(HCg_beta, SC, Xsmatrix, Umod_beta)

                        HCg_alpha = HCg_U_alpha
                        HCg_beta  = HCg_U_beta
                        HCg_alpha_diag = HCg_U_alpha_diag
                        HCg_beta_diag  = HCg_U_beta_diag
                        vHCg_ortho_alpha = vHCg_U_ortho_alpha
                        vHCg_ortho_beta  = vHCg_U_ortho_beta
                    else:
                        pass

                    # U matrices
                    U_cb_mat_o = np.zeros((dimC,dimC))
                    U_cb_mat_u = np.zeros((dimC,dimC))
                    U_cb_mat_o_diag = np.zeros((dimC,dimC))
                    U_cb_mat_u_diag = np.zeros((dimC,dimC)) 

                    for i in range(dimC):
                        U_cb_mat_o_diag[i][i] += U_cb 
                        U_cb_mat_u_diag[i][i] += U_cb
                    for ja in range(occ_hom_alpha.energy.size):
                        U_cb_mat_o_diag[ja][ja] = 0
                    for jb in range(occ_hom_beta.energy.size):
                        U_cb_mat_u_diag[jb][jb] = 0

                    U_cb_mat_o =  scipy.linalg.inv(np.matrix.getH(Xsmatrix)) \
                                  @ scipy.linalg.inv(vHCg_ortho_alpha) \
                                  @ U_cb_mat_o_diag @ vHCg_ortho_alpha \
                                  @ scipy.linalg.inv(Xsmatrix)
                    U_cb_mat_u =  scipy.linalg.inv(np.matrix.getH(Xsmatrix)) \
                                  @ scipy.linalg.inv(vHCg_ortho_beta) \
                                  @ U_cb_mat_u_diag @ vHCg_ortho_beta \
                                  @ scipy.linalg.inv(Xsmatrix)

                    if spin == 'alpha':
                        all_gates = df(np.ones(dimC)*Vg) 
                        energy_a_p_gate_cb = []
                        energy_b_p_gate_cb = []
                          
                        [energy_a_p_gate_cb.append(HCg_alpha_diag[i][i]) \
                         for i in range(dimC)]
                        [energy_b_p_gate_cb.append(HCg_beta_diag[i][i]) \
                         for i in range(dimC)]
                        energy_p_gate = (pd.concat([all_gates,\
                                                    df(energy_a_p_gate_cb),\
                                                    df(energy_b_p_gate_cb)],\
                                         axis=1)).astype(float)
                        np.savetxt(debf, energy_p_gate,\
                                   fmt=['%-5.10f', '%-10.10f', '%-10.10f'])
                        debf.write("\n")
                        debf.flush()
                    else:
                        pass

                    n_occ_extra   = 0.5
                    n_occ_extra_1 = 0.5

                    print (moprint.iprint_line(), flush=True)

                    info03 = ("Gate Voltage index: {: >33d}\n" 
                              "Gate Voltage (V): {: >43.8f}\n"
                              "Coulomb energy (eVs): {: >34.3f}\n"
                              "Extra electrons(holes) alpha/beta : "
                              "{: >16d}/{: >2d}"
                              "\nTotal number of alpha/beta electrons (dim(HOMOS)): "
                              "{: >3d}/{: >3d}\n")
                    print (info03.format(ivg, Vg, U_cb, extra_e_alpha,\
                                         extra_e_beta, Ng_homos_alpha[ivg],\
                                         Ng_homos_beta[ivg]), flush=True,end='\n')

                    if spin == 'alpha':
                        HCg = HCg_alpha
                    elif spin == 'beta':
                        HCg = HCg_beta 

                    #print ('5 barrier', flush=True, end='\n')
                    comm.Barrier()
                    #print ('... Fortfahren (5)!', flush=True, end='\n')

                    for irank in all_extra_ranks:
                        comm.send(spin,          dest=irank, tag=irank)
                        comm.send(Ef,            dest=irank, tag=irank)
                        comm.send(n_occ_extra,   dest=irank, tag=irank)
                        comm.send(n_occ_extra_1, dest=irank, tag=irank)
                        comm.send(U_cb_mat_o,    dest=irank, tag=irank)
                        comm.send(U_cb_mat_u,    dest=irank, tag=irank)
                        comm.send(HCg,           dest=irank, tag=irank)
                        comm.send(Ef_central,    dest=irank, tag=irank)
                        comm.send(Vg,            dest=irank, tag=irank)

                    #print ('6 barrier', flush=True, end='\n')
                    comm.Barrier()
                    #print ('... Fortfahren (6)!', flush=True, end='\n')

                    HL  = np.array(H_all.get("HL-"+spin))
                    SL  = np.array(S_all.get("SL"))
                    HR  = np.array(H_all.get("HR-"+spin))
                    SR  = np.array(S_all.get("SR"))
                    VCL = np.array(H_all.get("VCL-"+spin))
                    SCL = np.array(S_all.get("SCL"))
                    VCR = np.array(H_all.get("VCR-"+spin))
                    SCR = np.array(S_all.get("SCR"))
                    TL  = np.array(H_all.get("TL-"+spin))
                    STL = np.array(S_all.get("STL"))
                    TR  = np.array(H_all.get("TR-"+spin))
                    STR = np.array(S_all.get("STR"))

                    #info05 = ("Name of the Processor(Node): {}"
                    #          "Process: {} \nEnergies per rank: {}")
                    #print (moprint.ibordered(info05.format(name, rank,\
                    #       Energies_p_rank)), flush=True, end='\n')

                    #Vgm_HL  = SL  * Vg   
                    #Vgm_HR  = SR  * Vg 
                    Vgm_VCL = SCL * Vg 
                    Vgm_VCR = SCR * Vg 
                    #Vgm_TL  = STL * Vg 
                    #Vgm_TR  = STR * Vg 

                    #HLg  = HL  - Vgm_HL
                    #HRg  = HR  - Vgm_HR
                    VCLg = VCL - Vgm_VCL
                    VCRg = VCR - Vgm_VCR
                    #TLg  = TL  - Vgm_TL
                    #TRg  = TR  - Vgm_TR

                    Energies_r = Energies_p_rank
                    dos_cb, trans_cb = self.dos_t_calculation(Energies_r,\
                                                              Ef, dimC, HL, TL,\
                                                              SL, STL, HR,\
                                                              TR, SR, STR,\
                                                              SCL, VCLg, SCR,\
                                                              VCRg,\
                                                              n_occ_extra,\
                                                              U_cb_mat_o,\
                                                              U_cb_mat_u, SC,\
                                                              HCg,\
                                                              n_occ_extra_1,\
                                                              Ef_central)

                    dosT    = {}
                    dosT[0] = np.array(list(zip(dos_cb, trans_cb)))
                    for idosT, irank in enumerate(all_extra_ranks):
                        dosT[irank] = comm.recv(source=irank, tag=irank)

                    #print ('7 barrier', flush=True, end='\n')
                    comm.Barrier()

                    endVg  = time.time()
                    tempVg = endVg - startVg
                    
                    vg2print = Vg * np.ones((self.NE))
                    for i in dosT.keys():
                        if i == 0:
                            dost2print = dosT[i]
                        else:
                            dost2print = np.concatenate((dost2print,\
                                                         np.array(dosT[i])))
                    dost2print = df(data=dost2print, columns=['Dos(E)', 'T(E)'])
                    dost2print.insert(loc=0, column='Gate voltage', value=vg2print.T)
                    dost2print.insert(loc=1, column='Energy', value=Energies.real)

                    fmt_01 = '% -05.7f', '% -05.7f', '% -05.7f', '% -05.7f'
                    np.savetxt(f1, dost2print, delimiter='  ', fmt=fmt_01)

                    f1.write("\n")
                    f1.flush()
                    print (moprint.iprint_line(), flush=True)
                    print (moprint.iprint_line(), flush=True, end='\n')
                    print ("\n", flush=True, end='\n')

                    #print ('8 barrier', flush=True, end='\n')
                    comm.Barrier()

                # Check memory stats
                memoryHCg = HCg.size*HCg.itemsize
                shapeHCg  = HCg.shape
                info05 = ("Size / itemsize / shape / Memory(GB) of largest "
                          "matrix to invert: {:} / {:} / {:} / {:}")
                print (moprint.ibordered(info05.format(HCg.size,\
                       HCg.itemsize, shapeHCg, memoryHCg*1e-9)),\
                       flush=True,end='\n')
                    
                if spin == "alpha":
                    stophalf    = time.time()
                    temphalf    = stophalf-startT
                    hourshalf   = temphalf//3600
                    minuteshalf = temphalf//60
                    secondshalf = temphalf - 60*minuteshalf

                    info06 = ("Time for half calculation (up(alpha) spin): "
                              "{:2.0f}:{:2.0f}:{:2.0f} h/m/s")
                    print (moprint.ibordered(info06.format(hourshalf,minuteshalf,\
                           secondshalf)),flush=True,end='\n')
                    print ("\n\n")
                else:
                    pass
            if spin == "beta":
                stop    = time.time()
                temp    = stop-startT
                hours   = temp//3600
                minutes = temp//60 - hours*60
                seconds = temp - 60*minutes

                info06_01 = ("Entire time for NEGF+DFT+Coulomb blockade method "
                             "(up(alpha) and down(beta) spin): "
                             "{:.0f}:{:.0f}:{:.0f} h/m/s")
                print (moprint.ibordered(info06_01.format(hours,minutes,seconds)),\
                       flush=True,end='\n')
            else:
                pass

        elif rank == 1:
            H_all, S_all, PC_all = self.load_matrices_h5()

            spin = comm.recv(source=0, tag=111)

            #print ('0 barrier', flush=True, end='\n')
            comm.Barrier()

            HC = np.array(H_all.get("HC-"+spin))
            SC = np.array(S_all.get("SC"))
            PC = np.array(PC_all.get("PC-"+spin))

            Ef       = self.load_FermiE()
            Ef_alpha = Ef[0]
            Ef_beta  = Ef[1]
            Ef       = (Ef_alpha + Ef_beta)/2
            SC       = SC * self.scaling_b 
            N_part_beta_init, N_homos_beta_init, occ_homlum_beta_init,\
            HC_beta_init, vHCg_beta_init, HC_beta_diag_init =\
            self.energy_levs_shifted(HC, SC, PC, 0, 0)

            comm.send(N_part_beta_init,     dest=0, tag=11)
            comm.send(N_homos_beta_init,    dest=0, tag=12)
            comm.send(occ_homlum_beta_init, dest=0, tag=13)
            comm.send(HC_beta_init,         dest=0, tag=14)
            comm.send(vHCg_beta_init,       dest=0, tag=15)
            comm.send(HC_beta_diag_init,    dest=0, tag=16)

            info02 = ("Total number of {:4s} electrons (Tr(P * S)): "\
                      "{: 7.3f}\n"
                      "Scaling factor for alpha density matrix: {}")
            print (moprint.ibordered(info02.format(spin, N_part_beta_init,\
                   self.scaling_b)), flush=True,end='\n')

            #print ('1 barrier', flush=True, end='\n')
            comm.Barrier()

            #print ('2 barrier', flush=True, end='\n')
            #comm.Barrier()

            for spin_all in ["alpha", "beta"]:
                #print ('2 barrier', flush=True, end='\n')
                comm.Barrier()

                HCg_beta = HC_beta_init

                for ivg_all in range(self.NVgs):
                    ivg      = comm.recv(source=0, tag=211)
                    Vg_step  = comm.recv(source=0, tag=212)
                    Xsmatrix = comm.recv(source=0, tag=221)

                    #print ('3 barrier', flush=True, end='\n')
                    comm.Barrier()

                    N_part_beta, N_homos_beta, occ_homlum_beta,\
                    HCg_beta, vHCg_ortho_beta, HCg_beta_diag = \
                    self.energy_levs_shiftedXmat(HCg_beta, SC, PC, Vg_step, 0, Xsmatrix)

                    comm.send(N_part_beta,     dest=0, tag=213)
                    comm.send(N_homos_beta,    dest=0, tag=214)
                    comm.send(occ_homlum_beta, dest=0, tag=215)
                    comm.send(HCg_beta,        dest=0, tag=216)
                    comm.send(vHCg_ortho_beta, dest=0, tag=217)
                    comm.send(HCg_beta_diag,   dest=0, tag=218)

                    #print ('4 barrier', flush=True, end='\n')
                    comm.Barrier()

                    #print ('5 barrier', flush=True, end='\n')
                    comm.Barrier()

                    spin          = comm.recv(source=0, tag=rank)
                    Ef            = comm.recv(source=0, tag=rank)
                    n_occ_extra   = comm.recv(source=0, tag=rank)
                    n_occ_extra_1 = comm.recv(source=0, tag=rank)
                    U_cb_mat_o    = comm.recv(source=0, tag=rank)
                    U_cb_mat_u    = comm.recv(source=0, tag=rank)
                    HCg           = comm.recv(source=0, tag=rank)
                    Ef_central    = comm.recv(source=0, tag=rank)
                    Vg            = comm.recv(source=0, tag=rank)

                    #print ('6 barrier', flush=True, end='\n')
                    comm.Barrier()

                    dimC = HCg.shape[0]

                    HL  = np.array(H_all.get("HL-"+spin))
                    SL  = np.array(S_all.get("SL"))
                    HR  = np.array(H_all.get("HR-"+spin))
                    SR  = np.array(S_all.get("SR"))
                    VCL = np.array(H_all.get("VCL-"+spin))
                    SCL = np.array(S_all.get("SCL"))
                    VCR = np.array(H_all.get("VCR-"+spin))
                    SCR = np.array(S_all.get("SCR"))
                    TL  = np.array(H_all.get("TL-"+spin))
                    STL = np.array(S_all.get("STL"))
                    TR  = np.array(H_all.get("TR-"+spin))
                    STR = np.array(S_all.get("STR"))
                    PC  = np.array(PC_all.get("PC-"+spin))

                    #Vgm_HL  = SL  * Vg   
                    #Vgm_HR  = SR  * Vg 
                    Vgm_VCL = SCL * Vg 
                    Vgm_VCR = SCR * Vg 
                    #Vgm_TL  = STL * Vg 
                    #Vgm_TR  = STR * Vg 

                    #HLg  = HL  - Vgm_HL
                    #HRg  = HR  - Vgm_HR
                    VCLg = VCL - Vgm_VCL
                    VCRg = VCR - Vgm_VCR
                    #TLg  = TL  - Vgm_TL
                    #TRg  = TR  - Vgm_TR

                    Energies_r = Energies_p_rank
                    dos_cb, trans_cb = self.dos_t_calculation(Energies_r,\
                                                              Ef, dimC, HL, TL,\
                                                              SL, STL, HR,\
                                                              TR, SR, STR,\
                                                              SCL, VCLg, SCR,\
                                                              VCRg,\
                                                              n_occ_extra,\
                                                              U_cb_mat_o,\
                                                              U_cb_mat_u, SC,\
                                                              HCg,\
                                                              n_occ_extra_1,\
                                                              Ef_central)

                    #Energies = Energies_p_rank
                    #dos_cb, trans_cb = self.dos_t_calculation(Energies, Ef,\
                    #                                          dimC, HL, TL, SL,\
                    #                                          STL, HR, TR, SR,\
                    #                                          STR, SCL, VCL, SCR,\
                    #                                          VCR, n_occ_extra,\
                    #                                          U_cb_mat_o,\
                    #                                          U_cb_mat_u, SC, HCg,\
                    #                                          n_occ_extra_1,\
                    #                                          Ef_central)

                    comm.send(np.array(list(zip(dos_cb, trans_cb))), dest=0, tag=rank)

                    #print ('7 barrier', flush=True, end='\n')
                    comm.Barrier()
                    #print ('8 barrier', flush=True, end='\n')
                    comm.Barrier()

        elif rank == 2:
            H_all, S_all, PC_all = self.load_matrices_h5()

            #print ('0 barrier', flush=True, end='\n')
            comm.Barrier()

            SC = np.array(S_all.get("SC"))

            Xsmatrix = self.orthogonalization_basis(SC)
            print ('Done with X matrix', flush=True, end='\n')

            #print ('1 barrier', flush=True, end='\n')
            comm.Barrier()

            comm.send(Xsmatrix, dest=0, tag=222)
            print ('Sent X matrix', flush=True, end='\n')

            #print ('2 barrier', flush=True, end='\n')
            #comm.Barrier()

            for spin_all in ["alpha", "beta"]:
                #print ('Spin calculation (from spin all):', spin_all, flush=True)
                #print ('2 barrier', flush=True, end='\n')
                comm.Barrier()

                for ivg_all in range(self.NVgs):
                    #print ('3 barrier', flush=True, end='\n')
                    comm.Barrier()

                    #print ('4 barrier', flush=True, end='\n')
                    comm.Barrier()

                    #print ('5 barrier', flush=True, end='\n')
                    comm.Barrier()

                    spin          = comm.recv(source=0, tag=rank)
                    Ef            = comm.recv(source=0, tag=rank)
                    n_occ_extra   = comm.recv(source=0, tag=rank)
                    n_occ_extra_1 = comm.recv(source=0, tag=rank)
                    U_cb_mat_o    = comm.recv(source=0, tag=rank)
                    U_cb_mat_u    = comm.recv(source=0, tag=rank)
                    HCg           = comm.recv(source=0, tag=rank)
                    Ef_central    = comm.recv(source=0, tag=rank)
                    Vg            = comm.recv(source=0, tag=rank)

                    #print ('6 barrier', flush=True, end='\n')
                    comm.Barrier()

                    dimC = HCg.shape[0]

                    HL   = np.array(H_all.get("HL-"+spin))
                    SL   = np.array(S_all.get("SL"))
                    HR   = np.array(H_all.get("HR-"+spin))
                    SR   = np.array(S_all.get("SR"))
                    VCL  = np.array(H_all.get("VCL-"+spin))
                    SCL  = np.array(S_all.get("SCL"))
                    VCR  = np.array(H_all.get("VCR-"+spin))
                    SCR  = np.array(S_all.get("SCR"))
                    TL   = np.array(H_all.get("TL-"+spin))
                    STL  = np.array(S_all.get("STL"))
                    TR   = np.array(H_all.get("TR-"+spin))
                    STR  = np.array(S_all.get("STR"))
                    PC   = np.array(PC_all.get("PC-"+spin))

                    #Vgm_HL  = SL  * Vg   
                    #Vgm_HR  = SR  * Vg 
                    Vgm_VCL = SCL * Vg 
                    Vgm_VCR = SCR * Vg 
                    #Vgm_TL  = STL * Vg 
                    #Vgm_TR  = STR * Vg 

                    #HLg  = HL  - Vgm_HL
                    #HRg  = HR  - Vgm_HR
                    VCLg = VCL - Vgm_VCL
                    VCRg = VCR - Vgm_VCR
                    #TLg  = TL  - Vgm_TL
                    #TRg  = TR  - Vgm_TR

                    Energies_r = Energies_p_rank
                    dos_cb, trans_cb = self.dos_t_calculation(Energies_r,\
                                                              Ef, dimC, HL, TL,\
                                                              SL, STL, HR,\
                                                              TR, SR, STR,\
                                                              SCL, VCLg, SCR,\
                                                              VCRg,\
                                                              n_occ_extra,\
                                                              U_cb_mat_o,\
                                                              U_cb_mat_u, SC,\
                                                              HCg,\
                                                              n_occ_extra_1,\
                                                              Ef_central)

                    #Energies = Energies_p_rank
                    #dos_cb, trans_cb = self.dos_t_calculation(Energies, Ef,\
                    #                                          dimC, HL, TL, SL,\
                    #                                          STL, HR, TR, SR,\
                    #                                          STR, SCL, VCL, SCR,\
                    #                                          VCR, n_occ_extra,\
                    #                                          U_cb_mat_o,\
                    #                                          U_cb_mat_u, SC, HCg,\
                    #                                          n_occ_extra_1,\
                    #                                          Ef_central)

                    comm.send(np.array(list(zip(dos_cb, trans_cb))), dest=0, tag=rank)

                    #print ('7 barrier', flush=True, end='\n')
                    comm.Barrier()
                    #print ('8 barrier', flush=True, end='\n')
                    comm.Barrier()

        else:
            H_all, S_all, PC_all = self.load_matrices_h5()

            #print ('0 barrier', flush=True, end='\n')
            comm.Barrier()

            #print ('1 barrier', flush=True, end='\n')
            comm.Barrier()

            for spin_all in ["alpha", "beta"]:
                #print ('2 barrier', flush=True, end='\n')
                comm.Barrier()

                for ivg_all in range(self.NVgs):
                    #print ('3 barrier', flush=True, end='\n')
                    comm.Barrier()

                    #print ('4 barrier', flush=True, end='\n')
                    comm.Barrier()

                    #print ('5 barrier', flush=True, end='\n')
                    comm.Barrier()

                    spin          = comm.recv(source=0, tag=rank)
                    Ef            = comm.recv(source=0, tag=rank)
                    n_occ_extra   = comm.recv(source=0, tag=rank)
                    n_occ_extra_1 = comm.recv(source=0, tag=rank)
                    U_cb_mat_o    = comm.recv(source=0, tag=rank)
                    U_cb_mat_u    = comm.recv(source=0, tag=rank)
                    HCg           = comm.recv(source=0, tag=rank)
                    Ef_central    = comm.recv(source=0, tag=rank)
                    Vg            = comm.recv(source=0, tag=rank)

                    #print ('6 barrier', flush=True, end='\n')
                    comm.Barrier()

                    dimC = HCg.shape[0]

                    HL  = np.array(H_all.get("HL-"+spin))
                    SL  = np.array(S_all.get("SL"))
                    HR  = np.array(H_all.get("HR-"+spin))
                    SR  = np.array(S_all.get("SR"))
                    VCL = np.array(H_all.get("VCL-"+spin))
                    SCL = np.array(S_all.get("SCL"))
                    VCR = np.array(H_all.get("VCR-"+spin))
                    SCR = np.array(S_all.get("SCR"))
                    TL  = np.array(H_all.get("TL-"+spin))
                    STL = np.array(S_all.get("STL"))
                    TR  = np.array(H_all.get("TR-"+spin))
                    STR = np.array(S_all.get("STR"))
                    SC  = np.array(S_all.get("SC"))
                    PC  = np.array(PC_all.get("PC-"+spin))

                    #info05 = ("Name of the Processor(Node): {}"
                    #          "Process: {} \nEnergies per rank:{}")
                    #print (moprint.ibordered(info05.format(name, rank,\
                    #       Energies_p_rank)), flush=True, end='\n')

                    #Vgm_HL  = SL  * Vg
                    #Vgm_HR  = SR  * Vg
                    Vgm_VCL = SCL * Vg
                    Vgm_VCR = SCR * Vg
                    #Vgm_TL  = STL * Vg
                    #Vgm_TR  = STR * Vg

                    #HLg  = HL  - Vgm_HL
                    #HRg  = HR  - Vgm_HR
                    VCLg = VCL - Vgm_VCL
                    VCRg = VCR - Vgm_VCR
                    #TLg  = TL  - Vgm_TL
                    #TRg  = TR  - Vgm_TR

                    Energies_r = Energies_p_rank
                    dos_cb, trans_cb = self.dos_t_calculation(Energies_r,\
                                                              Ef, dimC, HL, TL,\
                                                              SL, STL, HR,\
                                                              TR, SR, STR,\
                                                              SCL, VCLg, SCR,\
                                                              VCRg,\
                                                              n_occ_extra,\
                                                              U_cb_mat_o,\
                                                              U_cb_mat_u, SC,\
                                                              HCg,\
                                                              n_occ_extra_1,\
                                                              Ef_central)

                    #Energies = Energies_p_rank
                    #dos_cb, trans_cb = self.dos_t_calculation(Energies, Ef,\
                    #                                          dimC, HL, TL, SL,\
                    #                                          STL, HR, TR, SR,\
                    #                                          STR, SCL, VCL, SCR,\
                    #                                          VCR, n_occ_extra,\
                    #                                          U_cb_mat_o,\
                    #                                          U_cb_mat_u, SC, HCg,\
                    #                                          n_occ_extra_1,\
                    #                                          Ef_central)
                    comm.send(np.array(list(zip(dos_cb, trans_cb))), dest=0, tag=rank)

                    #print ('7 barrier', flush=True, end='\n')
                    comm.Barrier()
                    #print ('8 barrier', flush=True, end='\n')
                    comm.Barrier()

    #@profile
    def energy_levs_shifted(self, HC, SC, PC, Vg, shift_v):
        # Obtain the X_s matrix that fullfils: 
        # X_s^{dagger} S_overlap X_s = 1
        Xsmatrix = self.orthogonalization_basis(SC)

        # Obtain the matrices: s and U for the orthogonalization procedure
        HCg = HC - (Vg * SC)

        vPC = scipy.linalg.eigvals(PC @ SC)
        n_occ = (df(vPC.real).sort_values(by=0,ascending=False,\
                 ignore_index=True)).rename(columns={0:"occ"})
        
        # Calculate the particle number
        N_part = np.trace(PC @ SC) 

        #############################################################
        # For counting the number of extra electrons, calculate the # 
        # un-gated central Hamiltonian homos and lumos              #
        #############################################################

        # Transform HC using Xsmatrix
        HCg_ortho = np.matrix.getH(Xsmatrix) @ HCg @ Xsmatrix

        # Check if HCg_ortho is Hermitean
        #A = np.allclose(HCg_ortho, np.transpose(HCg_ortho), rtol=1e-05, atol=1e-08)
        #print  (A)
        
        # Get eigenvalues of HC_ortho (HOMO-LUMOS)
        wHCg_ortho, vHCg_ortho = scipy.linalg.eigh(HCg_ortho)
        homos_lumos = (df(wHCg_ortho.real).sort_values(by=0,\
                       ignore_index=True)).rename(columns=\
                       {0: "energy_levels"})
        occ_homlum  = pd.concat([n_occ,homos_lumos], axis=1)

        HCg_diag = np.diag(wHCg_ortho)

        homos = []
        lumos = []

        for ihomlum in occ_homlum.itertuples():
            if ihomlum.Index <= (round(N_part,0)-1):
                homos.append(ihomlum)
            else:
                lumos.append(ihomlum)

        homos = df(homos).drop(["Index"], axis=1)
        lumos = df(lumos).drop(["Index"], axis=1)
        N_homos = homos.energy_levels.size

        return N_part, N_homos, occ_homlum, HCg, vHCg_ortho, HCg_diag

    def energy_levs_shiftedXmat(self, HC, SC, PC, Vg, shift_v, Xsmatrix):
        # Obtain the matrices: s and U for the orthogonalization procedure
        HCg = HC - (Vg * SC)

        vPC = scipy.linalg.eigvals(PC @ SC)
        n_occ    = (df(vPC.real).sort_values(by=0,ascending=False,\
                    ignore_index=True)).rename(columns={0:"occ"})
        
        # Calculate the particle number
        N_part = np.trace(PC @ SC) 

        #############################################################
        # For counting the number of extra electrons, calculate the # 
        # un-gated central Hamiltonian homos and lumos              #
        #############################################################

        # Transform HC using Xsmatrix
        HCg_ortho = np.matrix.getH(Xsmatrix) @ HCg @ Xsmatrix

        # Check if HCg_ortho is Hermitean
        #A = np.allclose(HCg_ortho, np.transpose(HCg_ortho), rtol=1e-05, atol=1e-08)
        #print  (A)
        #B = scipy.sparse.issparse(HCg_ortho)
        #print ('HCg_ortho sparse:', B)
        
        # Get eigenvalues of HC_ortho (HOMO-LUMOS)
        wHCg_ortho, vHCg_ortho = scipy.linalg.eigh(HCg_ortho)
        homos_lumos = (df(wHCg_ortho.real).sort_values(by=0,\
                       ignore_index=True)).rename(columns=\
                       {0: "energy_levels"})
        occ_homlum  = pd.concat([n_occ,homos_lumos], axis=1)

        HCg_diag = np.diag(wHCg_ortho)

        homos = []
        lumos = []

        for ihomlum in occ_homlum.itertuples():
            if ihomlum.Index <= (round(N_part,0)-1):
                homos.append(ihomlum)
            else:
                lumos.append(ihomlum)

        homos = df(homos).drop(["Index"], axis=1)
        lumos = df(lumos).drop(["Index"], axis=1)
        N_homos = homos.energy_levels.size

        return N_part, N_homos, occ_homlum, HCg, vHCg_ortho, HCg_diag
        
    def energy_levs_HCg_pU(self, HC, SC, Xsmatrix, Umod):
        # Obtain the matrices: s and U for the orthogonalization procedure
        HC_U = HC + (Umod @ SC)

        # Transform HC using Xsmatrix
        HC_U_ortho = np.matrix.getH(Xsmatrix) @ HC_U @ Xsmatrix
        
        # Get eigenvalues of HC_U_ortho (HOMO-LUMOS)
        wHC_U_ortho, vHC_U_ortho = scipy.linalg.eigh(HC_U_ortho)
        homos_lumos = (df(wHC_U_ortho.real).sort_values(by=0,\
                       ignore_index=True)).rename(columns=\
                       {0: "energy_levels"})
        #occ_homlum  = pd.concat([n_occ,homos_lumos], axis=1)

        HC_U_diag = np.diag(wHC_U_ortho)
        return HC_U, HC_U_diag, vHC_U_ortho
        
    def orthogonalization_basis(self, SC):
        vs,Us  = scipy.linalg.eigh(SC)
        seig   = np.diag(vs)
        seigsq = np.sqrt(np.linalg.inv(seig))
        Xsmatrix = Us @ seigsq @ np.matrix.getH(Us)
        return Xsmatrix

    def dos_t_calculation(self, Energies, Ef, dimC, HL, TL, SL, STL,\
                          HR, TR, SR, STR, SCL, VCL, SCR, VCR, n_occ_extra,\
                          U_cb_mat_o, U_cb_mat_u, SC, HCg, n_occ_extra_1,\
                          mu):
        # init DOS and transmission
        dos_cb   = np.zeros(len(Energies))
        trans_cb = np.zeros(len(Energies))
        #Energies += 1j*self.eta

        for iE, energy in enumerate(Energies):
            startE    = time.time()
            timeE_ave = 0

            # Accuracy of sancho method 
            eps     = 1E-4
            energy1 = energy + 1.j*self.eta + Ef
            
            sigmaL  = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            sigmaR  = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            sigmacb = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            Omega_m = np.zeros(shape=(dimC,dimC),dtype=np.complex)

            HCg_effective_cb = np.zeros(shape=(dimC,dimC),\
                                        dtype=np.complex)

            # Greens function of semi infnite left/right electrode
            gL = mo.sancho(energy1,HL,TL,SL,STL,eps)
            gR = mo.sancho(energy1,HR,TR,SR,STR,eps)
            
            # Compute self-energy of left/right electrode
            sigmaL = (energy1*SCL-VCL) \
                     @ gL @ np.matrix.getH(energy1*SCL-VCL)
            sigmaR = (energy1*SCR-VCR) \
                     @ gR @ np.matrix.getH(energy1*SCR-VCR)
            
            kB   = phys.physical_constants["Boltzmann constant in eV/K"][0]
            temp = 3 
            vsd  = 0.005
            #n_occ_extra   = 1./(np.exp(energy-(mu+vsd/2.))**(1./(kB*temp))+1.)
            #n_occ_extra_1 = 1. - (1./(np.exp(energy-(mu+vsd/2.))\
            #                     **(1./(kB*temp))+1.))
            #print ('n_occ_extra:', n_occ_extra.real,   flush=True)
            #print ('n_occ_extra_1:', n_occ_extra_1.real, flush=True)
            # Omega matrix
            Omega_m = n_occ_extra.real * U_cb_mat_o \
                      @ scipy.linalg.inv(energy1*SC - HCg\
                      - n_occ_extra.real * U_cb_mat_o\
                      - n_occ_extra_1.real * U_cb_mat_u)

            # 'Self' energy of the Coulomb blockade
            sigma_cb = ((energy1*SC - HCg) @ Omega_m) \
                       @ scipy.linalg.inv(np.eye(dimC) @ SC + Omega_m)

            # Effective interactive Hamiltonian for the Greens function
            HCg_effective_cb = HCg + sigma_cb + sigmaL + sigmaR

            # Compute the interacting central Greens function with 
            # parameter U and self energy of the leads
            G_cb = np.linalg.solve(energy1*SC - HCg_effective_cb,\
                                   np.identity(dimC)) 
            
            # Calculate broadening matrices 
            gammaL  = 1j*(sigmaL-np.matrix.getH(sigmaL))
            gammaR  = 1j*(sigmaR-np.matrix.getH(sigmaR))
            #gammaLR = 2 * gammaL @ gammaR / (gammaL + gammaR)
            
            dos_cb[iE]   = -1/np.pi * (np.trace(G_cb @ SC)).imag
            trans_cb[iE] = np.trace(gammaL @ np.matrix.getH(G_cb) \
                           @ gammaR @ G_cb).real

            endE       = time.time()
            tempE      = endE - startE
            timeE_ave += tempE
        #ave_houE = (timeE_ave/len(Energies)) // 3600
        #ave_minE = (timeE_ave/len(Energies)) // 60 - ave_houE*60
        #ave_secE = (timeE_ave/len(Energies)) - 60*ave_minE
        #info_exE = ("Average time per energy point: {} "\
        #            "{:.0f}:{:.0f}:{:.1f} h/m/s")
        #print (info_exE.format(5*' ', ave_houE , ave_minE, ave_secE,\
        #                       flush=True, end='\n'))

        return dos_cb, trans_cb

    # Method for getting DOS and T for the single particle approach
    def dos_t_calculation_sp(self, Energies, Ef, dimC, HL, TL, SL, STL,\
                             HR, TR, SR, STR, SCL, VCL, SCR, VCR, SC, HC):
        # init DOS and transmission
        dos_cb   = np.zeros(len(Energies))
        trans_cb = np.zeros(len(Energies))

        for iE, energy in enumerate(Energies):
            startE    = time.time()
            timeE_ave = 0

            # Accuracy of sancho method 
            eps     = 1E-4
            #shift   = -0.05
            energy1 = energy + 1.j*self.eta + Ef
            
            sigmaL  = np.zeros(shape=(dimC,dimC), dtype=np.complex)
            sigmaR  = np.zeros(shape=(dimC,dimC), dtype=np.complex)
            sigmacb = np.zeros(shape=(dimC,dimC), dtype=np.complex)
            Omega_m = np.zeros(shape=(dimC,dimC), dtype=np.complex)

            HC_effective = np.zeros(shape=(dimC,dimC), dtype=np.complex)

            # Greens function of semi infnite left/right electrode
            gL = mo.sancho(energy1,HL,TL,SL,STL,eps)
            gR = mo.sancho(energy1,HR,TR,SR,STR,eps)
            
            # Compute self-energy of left/right electrode
            sigmaL = (energy1*SCL-VCL) @ gL @ np.matrix.getH(energy1*SCL-VCL)
            sigmaR = (energy1*SCR-VCR) @ gR @ np.matrix.getH(energy1*SCR-VCR)
            
            HC_effective = HC + sigmaL + sigmaR

            # Calculate greens function of central system with 
            # effect of left and right electrodes via corrected
            # Hamiltonian
            G = scipy.linalg.solve(energy1*SC - HC_effective, np.identity(dimC))
            
            # Calculate broadening matrices 
            gammaL  = 1j*(sigmaL-np.matrix.getH(sigmaL))
            gammaR  = 1j*(sigmaR-np.matrix.getH(sigmaR))
            
            #Calculate transmission and dos
            dos_cb[iE]   = -1/np.pi * (np.trace(G @ SC)).imag
            trans_cb[iE] = np.trace(gammaL @ np.matrix.getH(G) @ gammaR @ G).real

            endE       = time.time()
            tempE      = endE - startE
            timeE_ave += tempE

        #ave_houE = (timeE_ave/len(Energies)) // 3600
        #ave_minE = (timeE_ave/len(Energies)) // 60 - ave_houE*60
        #ave_secE = (timeE_ave/len(Energies)) - 60*ave_minE
        #info_exE = ("Average time per energy point: {} "\
        #            "{:.0f}:{:.0f}:{:.1f} h/m/s")
        #print (info_exE.format(5*' ', ave_houE , ave_minE, ave_secE,\
        #                       flush=True, end='\n'))

        return dos_cb, trans_cb

class FET_DOST_Spin_fort:
    # Parallel version of the serial FET_DOST_Spin_fort
    def __init__(self,config):
        self.Sname    = config["System name"]
        self.NE       = config["Number of energy points"]
        self.Ea       = config["Lower energy border"]
        self.Eb       = config["Upper energy border"]
        self.path_in  = config["Path to the system 14-matrices"]
        self.eta      = config["Small imaginary part"]
        self.path_out = config["Path of output"]

        try:
            self.restart_calc = config["Restart calculation"]
        except KeyError:
            print ("No restart keyword found.")
            self.restart_calc  = "No"
        if self.restart_calc   in ["Yes","yes","Y","y"]:
            self.restart_file  = config["Restart file"]
        elif self.restart_calc in ["No","no","no","n"]:
            pass
        try:
            self.gate_correction = \
                               config["Gate correction to central Hamiltonian"]
        except KeyError:
            print ("Gating not applied to central Hamiltonian")
            self.gate_correction = "No"

        try:
            self.pDOS   = config["Projected DOS"]
        except KeyError:
            print ("No key value Projected DOS. Set to NO")
            self.pDOS   = "No"
        if self.pDOS in ["Yes","yes","Y","y"]:
            self.mol1    = config["Molecule 1"]
            self.mol2    = config["Molecule 2"]
            self.is1atom = config["System1 first atom"]
            self.fs1atom = config["System1 last atom"]
            self.is2atom = config["System2 first atom"]
            self.fs2atom = config["System2 last atom"]
        elif self.pDOS in ["No","no","N","n"]:
            pass

        try:
            self.CB_regime = config["Coulomb blockade"]
        except KeyError:
            print ("No key value Coulomb blockade. Set to NO")
            self.CB_regime = "No"
        if self.CB_regime in ["Yes","yes","Y","y"]:
            try:
                self.U_cb      = config["Coulomb energy in eVs"]
                self.scaling_a = config["Density matrix re-scaling (alpha)"]
                self.scaling_b = config["Density matrix re-scaling (beta)"] 
            except KeyError:
                print ("No Coulomb energy or re-scaling factor"
                       "for Coulomb blockade found.")
                self.U_cb = 0
                self.scaling_a = 1.
                self.scaling_b = 1.
        else:
            self.U_cb = 0
            self.scaling_a = 1.
            self.scaling_b = 1.

        self.NVgs = config["Number of gs-voltage points"]
        self.Vg_a = config["Lower gate-source voltage"]
        self.Vg_b = config["Upper gate-source voltage"]
        
        self.structure = sdef.SystemDef(config)

    def load_matrices_h5_sp(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the central region to each. Single particle case.
        H_all = h5py.File(self.path_in+"/H_all.h5", "r")
        S_all = h5py.File(self.path_in+"/S_all.h5", "r")
        return H_all, S_all

    def load_matrices_h5_cb(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each.
        H_all = h5py.File(self.path_in+"/H_all.h5", "r")
        S_all = h5py.File(self.path_in+"/S_all.h5", "r")
        PC    = h5py.File(self.path_in+"/PC.h5", "r")
        return H_all, S_all, PC
   
    def load_FermiE(self):
        # Load the chemical potential
        Ef = np.loadtxt(self.path_in+"/Ef")
        return Ef

    def NEGF(self):
        # Parallel Version of NEGF method.

        startT = time.time()

        path = self.path_out

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        all_extra_ranks = [i+1 for i in range(size-1)]

        Energies_p_rank = []

        if rank == 0:
            info00 = ("NEGF+DFT.\nParallel calculation.\nUKS.\nSystem name: {}")
            print (moprint.ibordered_title(info00.format(self.Sname)),\
                   flush=True,end='\n')

            name = MPI.Get_processor_name()

            print ("\n")

            print ("Number of MPI processes: ", size, flush=True, end="\n")

            print ("\n")
            
            if not os.path.exists(path):
                os.makedirs(path)

            # Init Energy range
            Energies = np.linspace(self.Ea, self.Eb, num=self.NE, dtype=complex)

            perrank, resperrank = divmod(Energies.size, size)
            counts = [perrank + 1 if p < resperrank else perrank for p in\
                      range(size)]

            starts = [sum(counts[:p])   for p in range(size)]
            ends   = [sum(counts[:p+1]) for p in range(size)]

            Energies_p_rank = [Energies[starts[p]:ends[p]] for p in range(size)]

        else:
            name     = MPI.Get_processor_name()
            Energies = None

        Energies_p_rank = comm.scatter(Energies_p_rank, root=0)

        H_all, S_all = self.load_matrices_h5_sp()
       
        if self.pDOS in ["No","no","N","n"]:
            if self.gate_correction in ["Yes", "yes", "Y", "y"]:

                Energies_p_rank += 1.j*self.eta
                Vgates = np.linspace(self.Vg_a, self.Vg_b, num=self.NVgs)

                for spin in ["alpha","beta"]: 
                    HL  = np.array(H_all.get("HL-"+spin))
                    SL  = np.array(S_all.get("SL"))
                    HR  = np.array(H_all.get("HR-"+spin))
                    SR  = np.array(S_all.get("SR"))
                    TL  = np.array(H_all.get("TL-"+spin))
                    STL = np.array(S_all.get("STL"))
                    TR  = np.array(H_all.get("TR-"+spin))
                    STR = np.array(S_all.get("STR"))
                    VCL = np.array(H_all.get("VCL-"+spin))
                    SCL = np.array(S_all.get("SCL"))
                    VCR = np.array(H_all.get("VCR-"+spin))
                    SCR = np.array(S_all.get("SCR"))
                    HC  = np.array(H_all.get("HC-"+spin))
                    SC  = np.array(S_all.get("SC"))
                    
                    Ef       = self.load_FermiE()
                    Ef_alpha = Ef[0]
                    Ef_beta  = Ef[1]
                    Ef       = (Ef_alpha + Ef_beta)/2

                    dimC = HC.shape[0]

                    with open(path+"/out-"+spin+"-sp-g.out", "w+") as f1:
                        for ivg, Vg in enumerate(Vgates):
                            if rank == 0:
                                print ("Gatting applied to central Hamiltonian",
                                       flush=True, end="\n")
                                info03 = ("Gate Voltage index: {: >24d}\n" 
                                          "Gate Voltage (V): {: >33.8f}\n")
                                print (info03.format(ivg, Vg), flush=True,
                                       end='\n')

                                homos_lumos, vHCg_ortho, HCg_diag = \
                                       self.energy_levs_shifted(HC, SC, Vg)

                                f2 = open(path+"/energies-"+spin+"-sp-g.out",
                                        "w+")

                                comm.Barrier()
                            else:
                                comm.Barrier()
                                pass

                            Vgm_HC  = SC  * Vg
                            Vgm_VCL = SCL * Vg 
                            Vgm_VCR = SCR * Vg 

                            HCg  = HC  - Vgm_HC
                            VCLg = VCL - Vgm_VCL
                            VCRg = VCR - Vgm_VCR

                            dos   = np.zeros(Energies_p_rank.size)
                            trans = np.zeros(Energies_p_rank.size)

                            ave_timeE = 0.

                            #if rank == 0:
                            #    print ("#Energy" + 7 * ' ' + 'Used time (sec)',
                            #           flush=True, end='\n')
                            #    print (moprint.iprint_line(), flush=True, end='\n')
                            #else:
                            #    pass

                            for iE, energy in enumerate(Energies_p_rank):
                                #print (otmod.get_threads(), flush=True, end='\n')
                                if rank == 2:
                                    startE = time.time()
                                else:
                                    pass

                                eps     = 1E-4
                                #shift   = -0.05
                                #energy1 = energy + Ef + shift
                                energy1 = energy + Ef

                                # Greens function of semi infnite left/right
                                # electrode
                                gL = mo.sancho(energy1, HL, TL, SL, STL, eps)
                                gR = mo.sancho(energy1, HR, TR, SR, STR, eps)
                                
                                # Compute self-energy of left/right electrode
                                sigmaL = (energy1*SCL-VCLg)\
                                         @ gL @ np.matrix.getH(energy1*SCL-VCLg)
                                sigmaR = (energy1*SCR-VCRg)\
                                         @ gR @ np.matrix.getH(energy1*SCR-VCRg)

                                # Compute DOS and T(E) of the central system.
                                dosf, transf = GC.dost(energy1, sigmaL, sigmaR,
                                                       HCg, SC, dimC) 
                                dos[iE]   = dosf
                                trans[iE] = transf

                                #if rank == 2:
                                #    endE       = time.time()
                                #    tempE      = endE - startE
                                #    ave_timeE += tempE
                                #    print (iE, 5*'  ', "Time p.e.p:", int(tempE % 60),
                                #           flush=True, end="\n")
                                #else:
                                #    pass

                            #ave_houE = (ave_timeE/Energies_p_rank.size) // 3600
                            #ave_minE = (ave_timeE/Energies_p_rank.size) // 60 \
                            #           - ave_houE*60
                            #ave_secE = (ave_timeE/Energies_p_rank.size) \
                            #           - 60*ave_minE
                            #info04   = ("Average time per energy point: {} "\
                            #            "{:.0f}:{:.0f}:{:.1f} h/m/s")
                            #print (info04.format(5*' ', ave_houE , ave_minE, ave_secE,
                            #       flush=True, end='\n'))
                            #print (moprint.iprint_line(), flush=True, end='\n')

                            #print ("Reached barrier 0", flush=True, end='\n')
                            comm.Barrier()

                            if rank == 0:
                                dosT = {}
                                dosT[0] = np.array(list(zip(dos, trans)))

                                for idosT, irank in enumerate(all_extra_ranks):
                                    dosT[irank] = comm.recv(source=irank, tag=irank)

                                comm.Barrier()

                                for i in dosT.keys():
                                    if i == 0:
                                        dost2print = dosT[i]
                                    elif i != 0:
                                        if dosT[i] == []:
                                            pass
                                        elif dosT[i] != []:
                                            dost2print = np.concatenate((dost2print,\
                                                                  np.array(dosT[i])))

                                vg2print = Vg * np.ones((self.NE))
                                dost2print = df(data=dost2print, \
                                                columns=["Dos(E)", "T(E)"])
                                dost2print.insert(loc=0, column="Gate voltage", \
                                                  value=vg2print.T)
                                dost2print.insert(loc=1, column='Energy', \
                                                  value=Energies.real)

                                fmt_01 = "% -05.7f", "% -05.7f", "% -05.7f", \
                                         "% -05.7f"
                                np.savetxt(f1, dost2print, delimiter='  ', fmt=fmt_01)

                                f1.write("\n")
                                f1.flush()

                                if rank == 0:
                                    homos_lumos = np.concatenate((homos_lumos,\
                                                  np.array(Vg * np.ones(dimC))))
                                        
                                    eigener2print = df(data=homos_lumos)
                                    fmt_02 = "% -05.7f", "% -05.7f"
                                    np.savetxt(f2, eigener2print, delimiter="  ",
                                               fmt=fmt_02)

                                    f2.write("\n")
                                    f2.flush()
                                else:
                                    pass

                                print (moprint.iprint_line(), flush=True, end='\n')
                                print ("\n", flush=True, end='\n')

                                comm.Barrier()

                            else:
                                comm.send(np.array(list(zip(dos, trans))), \
                                          dest=0, tag=rank)
                                comm.Barrier()
                                comm.Barrier()

                    if rank == 0:
                        if spin == "alpha":
                            stopH    = time.time()
                            tempH    = stopH-startT
                            hoursH   = tempH//3600
                            minutesH = tempH//60 - hoursH*60
                            secondsH = tempH - 60*minutesH
                            info04 = ("Half time for NEGF+DFT method "
                                      "(up(alpha) spin): "
                                      "{:.0f}:{:.0f}:{:.0f} h/m/s")
                            print (moprint.ibordered(info04.\
                                   format(hoursH, minutesH, secondsH)),\
                                   flush=True, end='\n')
                            print (moprint.iprint_line(), flush=True, end='\n')
                            print ("\n", flush=True)
                            f2.close()

                        elif spin == "beta":
                            # Check memory stats
                            memoryHC = HC.size*HC.itemsize
                            shapeHC  = HC.shape
                            sizeofHC = sys.getsizeof(HC)
                            print ("Size / itemsize / shape / "
                                   "sys.getsizeof(Kb) / "
                                   "Memory(Kb) of matrix to invert:\n"
                                   "{:} / {:} / {:} / {:} / {:} \n".\
                                   format(HC.size, HC.itemsize, shapeHC,\
                                   sizeofHC/1000, memoryHC/1000))               

                            stop    = time.time()
                            temp    = stop-startT
                            hours   = temp//3600
                            minutes = temp//60 - hours*60
                            seconds = temp - 60*minutes

                            info05 = ("Entire time for NEGF+DFT method "
                                      "(up(alpha) and down(beta) spin): "
                                      "{:.0f}:{:.0f}:{:.0f} h/m/s")
                            print (moprint.ibordered(info05.\
                                   format(hours, minutes, seconds)),\
                                   flush=True, end='\n')
                    else:
                        pass

            elif self.gate_correction in ["No", "no", "N", "n"]:
                Energies_p_rank += 1.j*self.eta
                Vgates = np.linspace(self.Vg_a, self.Vg_b, num=self.NVgs)

                for spin in ["alpha","beta"]: 
                    HL  = np.array(H_all.get("HL-"+spin))
                    SL  = np.array(S_all.get("SL"))
                    HR  = np.array(H_all.get("HR-"+spin))
                    SR  = np.array(S_all.get("SR"))
                    TL  = np.array(H_all.get("TL-"+spin))
                    STL = np.array(S_all.get("STL"))
                    TR  = np.array(H_all.get("TR-"+spin))
                    STR = np.array(S_all.get("STR"))
                    VCL = np.array(H_all.get("VCL-"+spin))
                    SCL = np.array(S_all.get("SCL"))
                    VCR = np.array(H_all.get("VCR-"+spin))
                    SCR = np.array(S_all.get("SCR"))
                    HC  = np.array(H_all.get("HC-"+spin))
                    SC  = np.array(S_all.get("SC"))
                    
                    Ef       = self.load_FermiE()
                    Ef_alpha = Ef[0]
                    Ef_beta  = Ef[1]
                    Ef       = (Ef_alpha + Ef_beta)/2

                    dimC = HC.shape[0]
                    if rank == 0:
                        f1 = open(path+"/out-"+spin+"-sp-ng.out", "w+")
                        if spin == "alpha":
                            print ("Gatting not applied to central Hamiltonian")
                        else:
                            pass
                    else:
                        pass
                    
                    dos   = np.zeros(self.NE)
                    trans = np.zeros(self.NE)
                    rank_string = str(rank)

                    dos, trans = self.dos_t_calculation_sp(Energies_p_rank,\
                                                           Ef, dimC, HL, TL,\
                                                           SL, STL, HR, TR, \
                                                           SR, STR, SCL, \
                                                           VCL, SCR, VCR, \
                                                           SC, HC)
                    if rank == 0:
                        dosT = {}
                        dosT[0] = np.array(list(zip(dos, trans)))
                        print (dosT)

                        for idosT, irank in enumerate(all_extra_ranks):
                            dosT[irank] = comm.recv(source=irank, tag=irank)
                        print(dosT)

                        comm.Barrier()

                        for i in dosT.keys():
                            if i == 0:
                                dost2print = dosT[i]
                            else:
                                dost2print = np.concatenate((dost2print,\
                                                            np.array(dosT[i])))
                        
                        dost2print = df(data=dost2print, \
                                        columns=["Dos(E)", "T(E)"])
                        dost2print.insert(loc=0, column="Energy", \
                                          value=Energies.real)
                        fmt_01 = "% -05.7f", "% -05.7f", "% -05.7f"
                        np.savetxt(f1, dost2print, delimiter="  ", fmt=fmt_01)

                        f1.write("\n")
                        f1.flush()
                        print (moprint.iprint_line(), flush=True, end='\n')
                        print ("\n", flush=True, end='\n')

                        comm.Barrier()

                    elif rank != 0:
                        comm.send(np.array(list(zip(dos, trans))), \
                                  dest=0, tag=rank)
                        comm.Barrier()
                        comm.Barrier()

                    if spin == "alpha":
                        stopH    = time.time()
                        tempH    = stopH-startT
                        hoursH   = tempH//3600
                        minutesH = tempH//60 - hoursH*60
                        secondsH = tempH - 60*minutesH
                        info04 = ("Half time for NEGF+DFT method "
                                  "(up(alpha) spin): "
                                  "{:.0f}:{:.0f}:{:.0f} h/m/s")
                        print (moprint.ibordered(info04.\
                               format(hoursH, minutesH, secondsH)),\
                               flush=True, end='\n')
                        print (moprint.iprint_line(), flush=True, end='\n')

                    elif spin == "beta":
                        # Check memory stats
                        memoryHC = HC.size*HC.itemsize
                        shapeHC  = HC.shape
                        sizeofHC = sys.getsizeof(HC)
                        print ("Size / itemsize / shape / "
                               "sys.getsizeof(Kb) / "
                               "Memory(Kb) of matrix to invert: "
                               "{:} / {:} / {:} / {:} / {:} \n".\
                               format(HC.size, HC.itemsize, shapeHC,\
                               sizeofHC/1000, memoryHC/1000))               

                        stop    = time.time()
                        temp    = stop-startT
                        hours   = temp//3600
                        minutes = temp//60 - hours*60
                        seconds = temp - 60*minutes

                        info05 = ("Entire time for NEGF+DFT method "
                                  "(up(alpha) and down(beta) spin): "
                                  "{:.0f}:{:.0f}:{:.0f} h/m/s")
                        print (moprint.ibordered(info05.\
                               format(hours, minutes, seconds)),\
                               flush=True, end='\n')

            else:
                pass

        elif self.pDOS in ["Yes","yes","Y","y"]:
            print ("Parallel PDOS-simulation not implemented")
            sys.exit()

    def energy_levs_shifted(self, HC, SC, Vg):
        # Obtain the X_s matrix that fullfils: 
        # X_s^{dagger} S_overlap X_s = 1
        Xsmatrix = self.orthogonalization_basis(SC)

        # Obtain the matrices: s and U for the orthogonalization procedure
        HCg = HC - (Vg * SC)

        # Transform HC using Xsmatrix
        HCg_ortho = np.matrix.getH(Xsmatrix) @ HCg @ Xsmatrix

        # Get eigenvalues of HC_ortho (HOMO-LUMOS)
        wHCg_ortho, vHCg_ortho = scipy.linalg.eigh(HCg_ortho)
        homos_lumos = (df(wHCg_ortho.real).sort_values(by=0,\
                       ignore_index=True)).rename(columns=\
                       {0: "energy_levels"})
        #occ_homlum  = pd.concat([n_occ,homos_lumos], axis=1)

        HCg_diag = np.diag(wHCg_ortho)

        return homos_lumos, vHCg_ortho, HCg_diag

    def orthogonalization_basis(self, SC):
        vs,Us  = scipy.linalg.eigh(SC)
        seig   = np.diag(vs)
        seigsq = np.sqrt(np.linalg.inv(seig))
        Xsmatrix = Us @ seigsq @ np.matrix.getH(Us)
        return Xsmatrix
