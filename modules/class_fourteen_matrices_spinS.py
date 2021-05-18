import generate_matrices.system_def_CP2K as gen_sysdef
import generate_matrices.class_systemdef as sdef
import modules.mo_print as moprint
import matplotlib.pyplot as plt
import scipy.constants as phys
import dask.dataframe as ddf
import modules.mo as mo
import dask.array as da
import modules.GC as GC
import scipy.linalg
import scipy.sparse
import pandas as pd
import numpy as np
import h5py
import time 
import json
import sys
import os
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from pandas import DataFrame as df
from tabulate import tabulate
from modules.OTfor import otmod

class fourteen_matrices_spin:
    ''' This class contains all relevant methods to calculate the DOS and 
    transmission of a quantum system connected to two periodic electrodes.
    The script works with Spin systems
    
    --HC,SC   ----> on-site Central Hamiltonian, Overlap of the central system.
    --VCL,SCL ----> hopping Hamiltonian, Overlap from the center to the left.
    --VCR,SCR ----> hopping Hamiltonian, Overlap from the center to the right.
    --HL,SL   ----> on-site Lead Hamiltonian, Overlap of the left electrode.
    --TL,STL  ----> hopping Hamiltonian, Overlap of the left electrode
    --HR,SR   ----> on-site Lead Hamiltonian, Overlap of the right electrode.
    --TR,STR  ----> hopping Hamiltonian, Overlap of the right electrode.
    --PC      ----> density matrix, density matrix of the central system.
    
    The direction of the hopping matrices have to be from the center to the 
    left/right, i.e. TL & VCL go to the left, TR & VCR go to the right. The 
    same for the hopping overlaps.
    '''

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
            self.pDOS = config["Projected DOS"]
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
        
        self.structure  = sdef.SystemDef(config)
        
    def load_electrodes(self, spin):
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
        # Load the matrices representing the Central Region.
        HC = np.loadtxt(self.path_in+"/HC-"+spin+".dat")
        SC = np.loadtxt(self.path_in+"/SC.dat")
        return HC,SC

    def load_matrices_h5(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each. Coulomb Blockade case.
        H_all = h5py.File(self.path_in+"/H_all.h5", "r")
        S_all = h5py.File(self.path_in+"/S_all.h5", "r")
        PC    = h5py.File(self.path_in+"/PC.h5", "r")
        return H_all, S_all, PC

    def load_matrices_h5_sp(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each. Single particle case.
        H_all = h5py.File(self.path_in+"/H_all.h5", "r")
        S_all = h5py.File(self.path_in+"/S_all.h5", "r")
        return H_all, S_all
   
    def load_FermiE(self):
        # Load the chemical potential
        Ef = np.loadtxt(self.path_in+"/Ef")
        return Ef

    #@profile
    def NEGF(self):
        start = time.time()
        info00 = "Transport NEGF+DFT.Sequential calculation.\nSystem name: {}"

        print (moprint.ibordered_title(info00.format(self.Sname)),
               flush=True, end='\n')
        ''' Tasks of this method:
        (*) Decimation of the semi-infinite electrodes into the self-energies 
            sigmaL and sigmaR using the sancho method.
        (*) Decorating the quantum region hamiltonian with the self-energies.
        (*) Calculating the DOS and transmission of the quantum region.
        '''

        path = self.path_out
        if not os.path.exists(path):
             os.makedirs(path)

        # Load the matrices with h5py.File()
        H_all, S_all = self.load_matrices_h5_sp()
        
        for spin in ["alpha","beta"]: 
            #HL,SL,HR,SR,VCL,SCL,VCR,SCR,TL,STL,TR,STR=self.load_electrodes(spin)
            #HC,SC = self.load_center(spin)
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
            #Ef = max(Ef)
     
            # Init energy range and add small imaginary part calculate retarded
            # quantities
            E    = np.linspace(self.Ea, self.Eb, self.NE,dtype=complex)
            E   += 1j*self.eta
            dimC = HC.shape[0]
        
            #init DOS and transmission
            dos   = np.zeros(self.NE)
            trans = np.zeros(self.NE)
            ##########################################################
            ## init DOS and transmission orthogonalized
            #dos_o   = np.zeros(self.NE)
            #trans_o = np.zeros(self.NE)
            ##########################################################

            if self.pDOS in ["No","no","N","n"]:
                f1 = open (path+"/out-"+spin+".out", "w+")
                print ("#Energy" + 7 * ' ' + 'Used time (sec)', \
                       flush=True, end='\n')
                print (moprint.iprint_line(), flush=True, end='\n')

                ave_timeE = 0.

                for iE,energy in enumerate(E):
                    startE = time.time()
                    
                    # Sancho method
                    # Accuracy of Sancho method
                    eps     = 1E-4
                    #shift   = -0.05
                    energy1 = energy + Ef
                    
                    sigmaL = np.zeros(shape=(dimC, dimC), dtype=np.complex)
                    sigmaR = np.zeros(shape=(dimC, dimC), dtype=np.complex)
                    HC_effective = np.zeros(shape=(dimC, dimC), dtype=np.complex)

                    # Greens function of semi infnite left/right electrode
                    gL = mo.sancho(energy1, HL, TL, SL, STL, eps)
                    gR = mo.sancho(energy1, HR, TR, SR, STR, eps)
                    
                    # Compute self-energy of left/right electrode
                    sigmaL = (energy1*SCL-VCL) \
                             @ gL @ np.matrix.getH(energy1*SCL-VCL)
                    sigmaR = (energy1*SCR-VCR) \
                             @ gR @ np.matrix.getH(energy1*SCR-VCR)

                    # Calculate broadening matrices 
                    gammaL = 1j*(sigmaL - np.matrix.getH(sigmaL))
                    gammaR = 1j*(sigmaR - np.matrix.getH(sigmaR))
                    
                    HC_effective = HC + sigmaL + sigmaR
                    ##########################################################
                    #HC_effective_o = (wHC_ortho @ HC_diag \
                    #                  @ scipy.linalg.inv(wHC_ortho)\
                    #                  + sigmaL + sigmaR)
                    ##########################################################
                    
                    # Calculate greens function of central system with effect 
                    # of left and right electrodes via corrected hamiltonian.
                    #G = scipy.linalg.solve(energy1*SC - HC_effective,\
                    #                       np.identity(dimC))
                    G = scipy.linalg.inv(energy1*SC - HC_effective)
                    ##########################################################
                    ## Calculate greens function of central system with effect 
                    # of left and right electrodes via corrected hamiltonian 
                    # orthogonal.
                    #G_o = np.linalg.inv(energy1*SC - HC_effective_o)
                    ##########################################################
                    
                    #Calculate transmission and dos
                    dos[iE]   = -1/np.pi * (np.trace(G @ SC)).imag
                    trans[iE] = (np.trace(gammaL @ np.matrix.getH(G)
                                 @ gammaR @ G).real)
                    #########################################################
                    ## Calculate transmission and dos orthogonalized
                    #dos_o[iE]   = -1/np.pi * (np.trace(G_o @ SC)).imag
                    #trans_o[iE] = (np.trace(gammaL @ np.matrix.getH(G_o) 
                    #               @ gammaR @ G_o).real)
                    #########################################################

                    endE       = time.time()
                    tempE      = endE - startE
                    ave_timeE += tempE

                    print (iE, 5*'  ', int(tempE % 60), dos, trans, flush=True, end="\n")

                    f1.write("{: 09.7f} {: 09.6f} {: 9.6f}".\
                             format(energy.real, dos[iE], trans[iE]))
                    f1.write("\n")
                    #########################################################
                    #f1.write("{: 09.7f} {: 09.6f} {: 9.6f} {: 09.6f} {: 9.6f}\
                    #         \n".format(energy.real, dos[iE], trans[iE],\
                    #         dos_o[iE], trans_o[iE]))
                    #########################################################
                    f1.flush()

                print (moprint.iprint_line(), flush=True, end='\n')
                ave_houE = (ave_timeE/self.NE) // 3600
                ave_minE = (ave_timeE/self.NE) // 60 - ave_houE*60
                ave_secE = (ave_timeE/self.NE) - 60*ave_minE
                info04 = ("Average time per energy point: {} "\
                          "{:.0f}:{:.0f}:{:.1f} h/m/s")
                print (info04.format(5*' ', ave_houE , ave_minE, ave_secE,\
                                     flush=True, end='\n'))
                print (moprint.iprint_line(), flush=True, end='\n')

                # Check memory stats
                if spin == "alpha":
                    memoryHC_eff = HC_effective.size*HC_effective.itemsize
                    shapeHC_eff  = HC_effective.shape

                    stophalf    = time.time()
                    temphalf    = stophalf-start
                    hourshalf   = temphalf//3600
                    minuteshalf = temphalf//60
                    secondshalf = temphalf - 60*minuteshalf

                    info01 = ("Size / itemsize / shape / Memory(Kb) of " \
                              "largest matrix to invert:\n" 
                              "{:} / {:} / {:} / {:}")
                    print (moprint.ibordered(info01.format(HC_effective.size, 
                           HC_effective.itemsize, shapeHC_eff,\
                           memoryHC_eff/1000)), flush=True, end='\n')

                    print ("\n")

                    info02 = ("Time for half of calculation (up(alpha) spin):"\
                              "{:.0f}:{:.0f}:{:.0f} h/m/s")
                    print (moprint.ibordered(info02.format(hourshalf,\
                           minuteshalf, secondshalf)), flush=True, end='\n')

                    print ("\n")

                elif spin == "beta":
                    stop    = time.time()
                    temp    = stop-start
                    hours   = temp//3600
                    minutes = temp//60 - hours*60
                    seconds = temp - 60*minutes
                    info03  = ("Entire time for NEGF method (up(alpha) and " \
                               "down(beta) spin): {:.0f}:{:.0f}:{:.0f} h/m/s")
                    print (moprint.ibordered(info03.format(hours,minutes,
                           seconds)), flush=True, end='\n')
                f1.close()
            
            elif self.pDOS in ["Yes","yes","Y","y"]:
                structure     = self.structure
                p2f           = structure.path2files
                CP2Koutfile   = structure.CP2Koutfile
                dic_elem, geo = structure.get_region_geometry(p2f+CP2Koutfile)
                iatomC        = structure.iatomC
                fatomL        = structure.fatomLleads

                SC_sys1    = np.copy(SC)
                SC_sys2    = np.copy(SC)
                dos_sys1   = np.zeros(self.NE)
                dos_sys2   = np.zeros(self.NE)
                #trans_sys1 = np.zeros(self.NE)
                #trans_sys2 = np.zeros(self.NE)

                if spin == "alpha":
                    info00 = ("PDOS calculation. \nMolecule 1: {}" \
                              "\nMolecule 2: {} \n\n")
                    print (info00.format(self.mol1,self.mol2), flush=True, 
                            end="\n")
                    '''N_sX: is the number of elements in system X and it 
                             is the number of atoms times the number of
                             spherical basis functions as in CP2K basis sets.
                    N_sXpX:  are the number or atoms of the system 1 before 
                             the first atom of the system 2 and after the 
                             last atom of system 2.
                    N_s1_C:  is the number of atoms of system 1 before the 
                             first atom in system 2 w.r.t the central region. 
                    '''
                    N_s1   = [(i,j) for i,j in geo 
                              if i < self.is2atom or i > self.fs2atom]
                    N_s2   = [(i,j) for i,j in geo 
                              if i >= self.is2atom and i <= self.fs2atom]
                    N_s1p1 = [(i,j) for i,j in geo 
                              if i >= self.is1atom and i < self.is2atom]
                    N_s1p2 = [(i,j) for i,j in geo 
                              if i > self.fs2atom and i <= self.fs1atom]
                    N_s1_C = [(i,j) for i,j in geo 
                              if i > fatomL and i <= (fatomL + (self.is2atom 
                              - iatomC))]

                    elem_s1   = 0
                    elem_s2   = 0
                    elem_s1p1 = 0
                    elem_s1p2 = 0
                    elem_s1_C = 0
                    
                    for k,l in N_s1:
                        if l in dic_elem.keys():
                            elem_s1 += dic_elem[l] 

                    for k,l in N_s2:
                        if l in dic_elem.keys():
                            elem_s2 += dic_elem[l] 
                    
                    for k,l in N_s1p1:
                        if l in dic_elem.keys():
                            elem_s1p1 += dic_elem[l] 
                    
                    for k,l in N_s1p2:
                        if l in dic_elem.keys():
                            elem_s1p2 += dic_elem[l] 
                    
                    for k,l in N_s1_C:
                        if l in dic_elem.keys():
                            elem_s1_C += dic_elem[l] 

                    #print ("Elements in system 1: ",elem_s1, "\n","Elements\
                    #       in system 2: ",elem_s2, sep='')
                    #print ("Elements in system 1, part 1: ",elem_s1p1, "\n",\
                    #       "Elements in system 1, part 2: "\
                    #       ,elem_s1p2, sep='')
                else:
                    pass

                # All the matrix elements that are not part of the 1st system,
                # are set to 0. 
                for i in list(range(elem_s1_C,elem_s1_C+elem_s2+1)):
                    for j in list(range(elem_s1_C,elem_s1_C+elem_s2+1)):
                        SC_sys1[i][j] = 0.

                # All the matrix elements that are not part of the 2nd system,
                # are set to 0. 
                for i in list(range(elem_s1_C+1)):
                    for j in list(range(elem_s1_C+1)):
                        SC_sys2[i][j] = 0.
                
                for i in list(range(elem_s1_C+elem_s2, SC.shape[0])):
                    for j in list(range(elem_s1_C+elem_s2, SC.shape[1])):
                        SC_sys2[i][j] = 0.
                
                f1 = open (path+"/out-"+spin+"-PDOS"+".out", "w+")

                print ("Energy" + 7 * ' ' + 'Used time', flush=True, end='\n')
                print (moprint.iprint_line(), flush=True, end='\n')
                
                for iE,energy in enumerate(E):
                    startE = time.time()
                    
                    # Sancho method
                    # Accuracy of sancho method 
                    eps     = 1E-4
                    #shift   = -0.05
                    energy1 = energy + Ef + shift

                    startsancho = time.time()
                    sigmaL       = np.zeros(shape=(dimC, dimC), dtype=np.complex)
                    sigmaR       = np.zeros(shape=(dimC, dimC), dtype=np.complex)
                    HC_effective = np.zeros(shape=(dimC, dimC), dtype=np.complex)

                    #Greens function of semi infnite left/right electrode
                    gL = mo.sancho(energy1, HL, TL, SL, STL, eps)
                    gR = mo.sancho(energy1, HR, TR, SR, STR, eps)
                    
                    endsancho = time.time()
                    tempS     = endsancho - startsancho
                    hoursS    = tempS//3600
                    minutesS  = tempS//60
                    secondsS  = tempS - 60*minutesS
                    
                    # Compute self-energy of left/right electrode
                    sigmaL = (energy1*SCL-VCL) @ gL \
                             @ np.matrix.getH(energy1*SCL-VCL)
                    sigmaR = (energy1*SCR-VCR) @ gR \
                             @ np.matrix.getH(energy1*SCR-VCR)

                    HC_effective = HC + sigmaL + sigmaR
                    
                    # Calculate greens function of central system with effect 
                    # of left and right electrodes via corrected hamiltonian
                    G = np.linalg.inv(energy1*SC - HC_effective)
                    
                    # Calculate broadening matrices 
                    gammaL = 1j*(sigmaL - np.matrix.getH(sigmaL))
                    gammaR = 1j*(sigmaR - np.matrix.getH(sigmaR))
                    
                    # Calculate transmission and dos
                    trans[iE] = (np.trace(gammaL @ np.matrix.getH(G) @ gammaR 
                                 @ G).real)
                    #trans_sys1[iE] = (np.trace(gammaL @ np.matrix.getH(G) 
                    #                   @ gammaR @ G).real)
                    #trans_sys2[iE] = (np.trace(gammaL @ np.matrix.getH(G) 
                    #                   @ gammaR @ G).real)
                    
                    #dos[iE]       = -1./np.pi * (np.trace(G @ SC)).imag
                    dos_sys1[iE] = -1./np.pi * (np.trace(G @ SC_sys1)).imag
                    dos_sys2[iE] = -1./np.pi * (np.trace(G @ SC_sys2)).imag

                    endE     = time.time()
                    tempE    = endE - startE
                    hoursE   = tempE//3600
                    minutesE = tempE//60 - hoursE*60
                    secondsE = tempE - 60*minutesE
                    print ("{: 3.5f}".format(energy.real) + 5 * ' ' +
                           "{:.0f}:{:.0f}:{:.1f} h/m/s".format(hoursE, \
                           minutesE, secondsE), flush=True, end='\n')
                    f1.write("{: 09.6f} {: 012.6f} {: 012.6f} {: 9.6f}\n"\
                            .format(energy.real, dos_sys1[iE], dos_sys2[iE],\
                            trans[iE]))
                    f1.flush()

                if spin == "alpha":
                    stophalf    = time.time()
                    temphalf    = stophalf-start
                    hourshalf   = temphalf//3600
                    minuteshalf = temphalf//60
                    secondshalf = temphalf - 60*minuteshalf
                    info01 = ("Time for half calculation (up(alpha) spin): "
                              "{:2.0f}:{:2.0f}:{:2.0f} h/m/s")
                    print (info01.format(hourshalf, minuteshalf, secondshalf),
                           flush=True, end="\n")

                    # Check memory stats
                    memoryHC_eff = HC_effective.size*HC_effective.itemsize
                    shapeHC_eff  = HC_effective.shape
                    info02 =("Size / itemsize / shape / Memory(Kb) of largest "
                             "matrix to invert: {:} / {:} / {:} / {:}\n")
                    print (info02.format(HC_effective.size, HC_effective.itemsize,
                           shapeHC_eff, memoryHC_eff/1000), flush=True,\
                           end="\n") 
                if spin == "beta":
                    stop    = time.time()
                    temp    = stop - start
                    hours   = temp//3600
                    minutes = temp//60 - hours*60
                    seconds = temp - 60*minutes
                    info03  = ("Entire time for NEGF-PDOS method: "
                               "{:.0f}:{:.0f}:{:.0f} h/m/s")
                    print (info03.format(hours, minutes, seconds), \
                           flush=True, end='\n')
                else:
                    pass

                f1.close()
    #@profile 
    def NEGF_CB(self):
        ''' Tasks of this method:
        (.) Calculate the non-interacting greens function of the leads as well 
            as the self energies.
        (.) Correct the central Hamiltonian via the Gate Voltage.
        (.) Correct the central Hamiltonian with the U parameter.
        (.) Obtain the occupation number n_i
        (.) Calculate the central retarded Greens function.
        (.) Calculate the central lesser Greens function.
        (.) Calculate the current.
        '''

        start = time.time()
        info00 = ("Coulomb Blockade NEGF+DFT. Sequential calculation"
                  "\nSystem name: {}")
        print (moprint.ibordered_title(info00.format(self.Sname)),\
               flush=True, end='\n')

        path = self.path_out
        if not os.path.exists(path):
             os.makedirs(path)

        # Load the matrices with h5py.File()
        H_all, S_all, PC_all = self.load_matrices_h5()
        
        for spin in ["alpha","beta"]: 
            # Load the matrices with regular numpy.loadtxt()
            #HL,SL,HR,SR,VCL,SCL,VCR,SCR,TL,STL,TR,STR =\
            #        self.load_electrodes(spin)
            #HC,SC = self.load_center(spin)

            if spin == 'alpha':
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
                HC   = np.array(H_all.get("HC-"+spin))
                #HC_s = np.array(H_all.get("HC-"+spin))
                SC   = np.array(S_all.get("SC"))
                PC   = np.array(PC_all.get("PC-"+spin))
            elif spin == 'beta':
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
                HC   = np.array(H_all.get("HC-"+spin))
                #HC_s = np.array(H_all.get("HC-"+spin))
                SC   = np.array(S_all.get("SC"))
                PC   = np.array(PC_all.get("PC-"+spin))

            #HC  = np.array(H_all.get("HC-"+spin))
            #SC  = np.array(S_all.get("SC"))
            #PC  = np.array(PC_all.get("PC-"+spin))

            #print ('HL  = ', df(HL).to_string(index=False),  "\n",\
            #       'SL  = ', df(SL).to_string(index=False),  "\n",\
            #       'HR  = ', df(HR).to_string(index=False),  "\n",\
            #       'SR  = ', df(SR).to_string(index=False),  "\n",\
            #       'VCL = ', df(VCL).to_string(index=False), "\n",\
            #       'SCL = ', df(SCL).to_string(index=False), "\n",\
            #       'VCR = ', df(VCR).to_string(index=False), "\n",\
            #       'SCR = ', df(SCR).to_string(index=False), "\n",\
            #       'TL  = ', df(TL).to_string(index=False),  "\n",\
            #       'STL = ', df(STL).to_string(index=False), "\n",\
            #       'TR  = ', df(TR).to_string(index=False),  "\n",\
            #       'STR = ', df(STR).to_string(index=False), "\n",\
            #       'HC  = ', df(HC).to_string(index=False),  "\n",\
            #       'SC  = ', df(SC).to_string(index=False),  "\n",\
            #       'PC  = ', df(PC).to_string(index=False),  "\n")


            #if spin == 'alpha':
            #    HL   = np.array(([ -1,0],[0,1]))
            #    SL   = np.array(([1,0],[0,1]))
            #    HR   = np.array(([-1,0],[0,1]))
            #    SR   = np.array(([1,0],[0,1]))
            #    VCL  = np.array(([0.5,0.5],[0.5,0.5]))
            #    SCL  = np.array(([0,0],[0,0]))
            #    VCR  = np.array(([0.5,0.5],[0.5,0.5]))
            #    SCR  = np.array(([0,0],[0,0]))
            #    TL   = np.array(([0.5,0.5],[0.5,0.5]))
            #    STL  = np.array(([0,0],[0,0]))
            #    TR   = np.array(([0.5,0.5],[0.5,0.5]))
            #    STR  = np.array(([0,0],[0,0]))
            #    HC   = np.array(([-1,0],[0,1]))
            #    HC_s = np.array(([-1,0],[0,1]))
            #    SC   = np.array(([1,0],[0,1]))
            #    PC   = np.array(([1,0],[0,0]))
            #elif spin == 'beta':
            #    HL   = np.array(([-1,0],[0,1]))
            #    SL   = np.array(([1,0],[0,1]))
            #    HR   = np.array(([-1,0],[0,1]))
            #    SR   = np.array(([1,0],[0,1]))
            #    VCL  = np.array(([0.5,0.5],[0.5,0.5]))
            #    SCL  = np.array(([0,0],[0,0]))
            #    VCR  = np.array(([0.5,0.5],[0.5,0.5]))
            #    SCR  = np.array(([0,0],[0,0]))
            #    TL   = np.array(([0.5,0.5],[0.5,0.5]))
            #    STL  = np.array(([0,0],[0,0]))
            #    TR   = np.array(([0.5,0.5],[0.5,0.5]))
            #    STR  = np.array(([0,0],[0,0]))
            #    HC   = np.array(([-0.5,0],[0,1.5]))
            #    HC_s = np.array(([-0.5,0],[0,1.5]))
            #    SC   = np.array(([1,0],[0,1]))
            #    PC   = np.array(([1,0],[0,0]))

            Ef       = self.load_FermiE()
            Ef_alpha = Ef[0]
            Ef_beta  = Ef[1]
            Ef       = (Ef_alpha + Ef_beta)/2

            if spin == "alpha":
                SC = SC * self.scaling_a
                N_part_alpha_init, N_homos_alpha_init, occ_homlum_alpha_init,\
                HC_alpha_init, vHCg_alpha_init, HC_alpha_diag_init =\
                self.energy_levs_shifted(HC, SC, PC, 0, 0)

                #############################################################
                #N_part_alpha_s_init, N_homos_alpha_s_init,\
                #occ_homlum_alpha_s_init, HC_alpha_s_init, vHCg_alpha_s_init,\
                #HC_alpha_diag_s_init =\
                #self.energy_levs_shifted(HC_s, SC, PC, 0, 0)
                #############################################################

                info01 = ("Average Fermi energy: {: 4.5f} eV\nFermi energy "
                          "alpha/beta: {: 4.5f} eV / {:4.5f} eV")
                print (moprint.ibordered(info01.format(Ef, Ef_alpha, Ef_beta)),\
                       flush=True, end='\n')
                info02 = ("Total number of {:4s} electrons (Tr(P * S)): "\
                          "{: 7.3f}\n"
                          "Scaling factor for alpha density matrix: {}")
                print (moprint.ibordered(info02.format(spin, N_part_alpha_init,\
                       self.scaling_a)), flush=True,end='\n')
            elif spin == "beta":
                SC = SC * self.scaling_b 
                N_part_beta_init, N_homos_beta_init, occ_homlum_beta_init,\
                HC_beta_init, vHCg_beta_init, HC_beta_diag_init =\
                self.energy_levs_shifted(HC, SC, PC, 0, 0)

                #############################################################
                #N_part_beta_s_init, N_homos_beta_s_init,\
                #occ_homlum_beta_s_init, HC_beta_s_init, vHCg_beta_s_init,\
                #HC_beta_diag_s_init =\
                #self.energy_levs_shifted(HC_s, SC, PC, 0, 0)
                #############################################################

                info02 = ("Total number of {:4s} electrons (Tr(P * S)): "\
                          "{: 7.3f}\n"
                          "Scaling factor for alpha density matrix: {}")
                print (moprint.ibordered(info02.format(spin, N_part_beta_init,\
                       self.scaling_b)), flush=True,end='\n')

        occ_homlum_a_b_init = pd.concat([occ_homlum_alpha_init,\
                                         occ_homlum_beta_init], axis=1,\
                                         ignore_index=True)
        occ_homlum_a_b_init.columns = ["occ_alpha",\
                                       "energy_levels_alpha",\
                                       "occ_beta",\
                                       "energy_levels_beta"]

        Xsmatrix = self.orthogonalization_basis(SC)

        print ("\nInitial energy levels per spin and their occupation numbers")
        print (tabulate(occ_homlum_a_b_init, headers="keys", \
               tablefmt="fancy_grid"))

        Vgates = np.linspace(self.Vg_a, self.Vg_b, num=self.NVgs)
        for spin in ["alpha", "beta"]:
            Vg_step = 0

            Ng_homos_alpha = {}
            Ng_homos_beta  = {}
            Ng_lumos_alpha = {}
            Ng_lumos_beta  = {}

            HCg_alpha = HC_alpha_init
            HCg_beta  = HC_beta_init

            #############################################################
            #HCg_alpha_s = HC_alpha_s_init
            #HCg_beta_s  = HC_beta_s_init
            #############################################################

            f1   = open(path+"/out-"+spin+"-cb.out", "w+")
            if spin == 'alpha':
                debf = open(path+"/energy-levels.out", "w+")
            else:
                pass

            N_homos_alpha_vg = N_homos_alpha_init
            N_homos_beta_vg  = N_homos_beta_init

            #############################################################
            #N_homos_alpha_vg_s = N_homos_alpha_s_init
            #N_homos_beta_vg_s  = N_homos_beta_s_init
            #############################################################

            for ivg, Vg in enumerate(Vgates):
                extra_e_alpha = 0
                extra_e_beta  = 0

                Vgm_HL  = SL  * Vg   
                Vgm_HR  = SR  * Vg 
                Vgm_VCL = SCL * Vg 
                Vgm_VCR = SCR * Vg 
                Vgm_TL  = STL * Vg 
                Vgm_TR  = STR * Vg 

                Vgm = SC * Vg 

                #HCg_s = HC_s - (SC * Vg)

                if ivg == 1:
                    Vg_step = Vg
                else:
                    pass

                N_part_alpha, N_homos_alpha, occ_homlum_alpha,\
                HCg_alpha, vHCg_ortho_alpha, HCg_alpha_diag = \
                self.energy_levs_shifted(HCg_alpha, SC, PC, Vg_step, 0)

                #############################################################
                #N_part_alpha_s, N_homos_alpha_s, occ_homlum_alpha_s,\
                #HCg_alpha_s, vHCg_ortho_alpha_s, HCg_alpha_diag_s = \
                #self.energy_levs_shifted(HCg_alpha_s, SC, PC, Vg_step, 0)
                #############################################################

                N_part_beta, N_homos_beta, occ_homlum_beta,\
                HCg_beta, vHCg_ortho_beta, HCg_beta_diag = \
                self.energy_levs_shifted(HCg_beta, SC, PC, Vg_step, 0)

                #############################################################
                #N_part_beta_s, N_homos_beta_s, occ_homlum_beta_s,\
                #HCg_beta_s, vHCg_ortho_beta_s, HCg_beta_diag_s = \
                #self.energy_levs_shifted(HCg_beta_s, SC, PC, Vg_step, 0)
                #############################################################
                ##Debbug Information
                #print ('CB: HCg alpha pre')
                #print (tabulate(df(HCg_alpha_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                #
                #print ('CB')
                #print (tabulate(df(HCg_alpha_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print ('NI')
                #print (tabulate(df(HCg_alpha_diag_s), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print (tabulate(df(HCg_beta_diag_s), tablefmt="grid",\
                #                showindex=False), end='\n')
                #
                #############################################################

                dimC  = HCg_alpha.shape[0]

                occ_homlum_g_a_b =\
                pd.concat([occ_homlum_alpha, occ_homlum_beta],\
                          axis=1, ignore_index=True)
                occ_homlum_g_a_b.columns = ["occ_alpha",\
                                            "energy_levels_alpha",\
                                            "occ_beta",\
                                            "energy_levels_beta"]

                # Fuer Test: Fermi level ist gleich als 0 (Vsd=0)
                Ef_central = 0

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
                # Was zu tun wenn lumos leer ist:
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

                #############################################################
                ##Debbug Information
                #print ('N_homos_alpha pre', N_homos_alpha_vg)
                #print ('Ng_homos_alpha[ivg] pre', Ng_homos_alpha[ivg])
                #############################################################

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
                #if N_homos_alpha > N_homos_alpha_init :
                #    for inew_occa in range(extra_e_alpha):
                #        occ_hom_alpha['occ'].iloc[N_homos_alpha_init+inew_occa]\
                #        = occ_hom_alpha['occ'].iloc[N_homos_alpha_init-1]
                #else:
                #    pass
                #if N_homos_beta > N_homos_beta_init:
                #    for inew_occb in range(extra_e_beta):
                #        occ_hom_beta['occ'].iloc[N_homos_beta_init+inew_occb]\
                #        = occ_hom_beta['occ'].iloc[N_homos_beta_init-1]
                #else:
                #    pass

                #############################################################
                ##Debbug Information
                #print ('N_homos_alpha', N_homos_alpha_vg)
                #print ('Ng_homos_alpha[ivg]', Ng_homos_alpha[ivg])
                #
                #print (tabulate(occ_hom_alpha, tablefmt="fancy_grid",\
                #       showindex=False), end='\n')
                #print (tabulate(occ_lum_alpha, tablefmt="fancy_grid",\
                #       showindex=False), end='\n')
                #print (tabulate(occ_hom_beta, tablefmt="fancy_grid",\
                #       showindex=False), end='\n')
                #print (tabulate(occ_lum_beta, tablefmt="fancy_grid",\
                #       showindex=False), end='\n')
                ##############################################################

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
                        #print ('Energy levels alpha without U: ')
                        #print (tabulate(df(HCg_alpha_diag), tablefmt="grid",\
                        #       showindex=False), end='\n')

                        for ienea in range(occ_hom_alpha.energy.size, dimC):
                            #HCg_alpha_diag[ienea][ienea] += U_cb 
                            Umod_alpha[ienea][ienea] += U_cb 
                        #print ('Energy levels alpha plus U: ')
                        #print (tabulate(df(HCg_alpha_diag), tablefmt="grid", \
                        #                showindex=False), end='\n')

                        #print ('Energy levels beta without U: ')
                        #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                        #                showindex=False), end='\n')
                        for ieneb in range(0,dimC):
                            if HCg_beta_diag[ieneb][ieneb] >\
                                            occ_hom_alpha['energy'].iloc[-1]:
                                #HCg_beta_diag[ieneb][ieneb] += U_cb
                                Umod_beta[ieneb][ieneb] += U_cb
                            else:
                                pass
                        #print ('Energy levels beta plus U: ')
                        #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                        #                showindex=False), end='\n')
                    if extra_e_alpha == 0 and extra_e_beta != 0:
                        #print ('Energy levels beta without U: ')
                        #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                        #       showindex=False), end='\n')

                        for ieneb in range(occ_hom_beta.energy.size, dimC):
                            #HCg_beta_diag[ieneb][ieneb] += U_cb 
                            Umod_beta[ieneb][ieneb] += U_cb 
                        #print ('Energy levels beta plus U: ')
                        #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                        #   showindex=False), end='\n')

                        #print ('Energy levels alpha without U: ')
                        #print (tabulate(df(HCg_alpha_diag),\
                        #                tablefmt="grid", \
                        #                showindex=False), end='\n')
                        for ienea in range(0,dimC):
                            if HCg_alpha_diag[ienea][ienea] >\
                                             occ_hom_beta['energy'].iloc[-1]:
                                #HCg_alpha_diag[ienea][ienea] += U_cb
                                Umod_alpha[ienea][ienea] += U_cb
                            else:
                                pass
                        #print ('Energy levels alpha plus U: ')
                        #print (tabulate(df(HCg_alpha_diag),\
                        #                tablefmt="grid", \
                        #                showindex=False), end='\n')

                    #if extra_e_alpha != 0 and extra_e_beta != 0:
                    #    #print ('Energy levels alpha without U: ')
                    #    #print (tabulate(df(HCg_alpha_diag), tablefmt="grid",\
                    #    #       showindex=False), end='\n')

                    #    for ienea in range(homos_alpha.energy.size, dimC):
                    #        HCg_alpha_diag[ienea][ienea] += U_cb 
                    #    #print ('Energy levels alpha plus U: ')
                    #    #print (tabulate(df(HCg_alpha_diag), tablefmt="grid",\
                    #    #                showindex=False), end='\n')

                    #    #print ('Energy levels beta without U: ')
                    #    #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                    #    #                showindex=False), end='\n')
                    #    for ieneb in range(0,dimC):
                    #        if HCg_beta_diag[ieneb][ieneb] >\
                    #           homos_beta['energy'].iloc[-1]:
                    #            HCg_beta_diag[ieneb][ieneb] += U_cb
                    #        else:
                    #            pass
                    #    #print ('Energy levels beta plus U: ')
                    #    #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                    #    #                showindex=False), end='\n')
                #############################################################
                ##Debbug Information
                #print ('CB: HCg alpha vor')
                #print (tabulate(df(HCg_alpha_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print ('CB: HCg beta vor')
                #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                ##############################################################

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
                    
                #############################################################
                ##Debbug Information
                #print ('CB: HCg alpha')
                #print (tabulate(df(HCg_alpha_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print ('Non-interacting: alpha')
                #print (tabulate(df(HCg_alpha_diag_s), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print ('CB: HCg beta')
                #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print ('Non-interacting: beta')
                #print (tabulate(df(HCg_beta_diag_s), tablefmt="grid",\
                #                showindex=False), end='\n')
                #############################################################

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

                #############################################################
                ##Debbug Information
                #print (tabulate(df(U_cb_mat_o_diag), tablefmt="grid",\
                #       showindex=False), end='\n')
                #print (tabulate(df(U_cb_mat_u_diag), tablefmt="grid",\
                #       showindex=False), end='\n')
                #############################################################

                U_cb_mat_o =  scipy.linalg.inv(np.matrix.getH(Xsmatrix)) \
                              @ scipy.linalg.inv(vHCg_ortho_alpha) \
                              @ U_cb_mat_o_diag @ vHCg_ortho_alpha \
                              @ scipy.linalg.inv(Xsmatrix)
                U_cb_mat_u =  scipy.linalg.inv(np.matrix.getH(Xsmatrix)) \
                              @ scipy.linalg.inv(vHCg_ortho_beta) \
                              @ U_cb_mat_u_diag @ vHCg_ortho_beta \
                              @ scipy.linalg.inv(Xsmatrix)
                
                #############################################################
                ##Debbug Information
                #print ('Nach')
                #print (tabulate(df(HCg_alpha_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print (tabulate(df(HCg_beta_diag), tablefmt="grid",\
                #                showindex=False), end='\n')
                #print (tabulate(df(HCg_s), tablefmt="grid", showindex=False),\
                #       end='\n')
                #############################################################

                # Printing of the "HOMOs-LUMOs" of the central system to file
                if spin == 'alpha':
                    all_gates = df(np.ones(dimC)*Vg) 
                    #energy_a_p_gate_s  = []
                    #energy_b_p_gate_s  = []
                    energy_a_p_gate_cb = []
                    energy_b_p_gate_cb = []
                      
                    #[energy_a_p_gate_s.append(HCg_alpha_diag_s[i][i]) \
                    # for i in range(dimC)]
                    #[energy_b_p_gate_s.append(HCg_beta_diag_s[i][i]) \
                    # for i in range(dimC)]
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
                    #energy_p_gate = (pd.concat([all_gates,\
                    #                            df(energy_a_p_gate_s),\
                    #                            df(energy_b_p_gate_s),\
                    #                            df(energy_a_p_gate_cb),\
                    #                            df(energy_b_p_gate_cb)],\
                    #                 axis=1)).astype(float)
                    #np.savetxt(debf, energy_p_gate,\
                    #           fmt=['%-5.10f', '%-10.10f', '%-10.10f',
                    #                '%-10.10f', '%-10.10f'])
                    debf.write("\n")
                    debf.flush()
                else:
                    pass

                n_occ_extra   = 1
                n_occ_extra_1 = 1

                info03 = ("Gate Voltage index: {: >20d}\n"
                          "Gate Voltage: {: >33.8f}\n"
                          "Coulomb energy (eVs): {: >20.3f}\n"
                          "Extra electrons(holes) alpha/beta : "
                          "{: >3d}/{: >3d}"
                          "\nTotal number of alpha/beta electrons (dim(HOMOS)): "
                          "{: >3d}/{: >3d}\n")
                print (info03.format(ivg, Vg, U_cb, extra_e_alpha,\
                                     extra_e_beta, Ng_homos_alpha[ivg],\
                                     Ng_homos_beta[ivg]), flush=True,end='\n')

                if spin == 'alpha':
                    HCg   = HCg_alpha
                    #HCg_s = HCg_alpha_s
                elif spin == 'beta':
                    HCg   = HCg_beta 
                    #HCg_s = HCg_beta_s

                # Define energy range and add small imaginary part calculate 
                # retarded quantities
                E  = np.linspace(self.Ea,self.Eb,self.NE,dtype=complex)
                E += 1j*self.eta

                # init DOS and transmission
                dos   = np.zeros(self.NE)
                trans = np.zeros(self.NE)

                HLg  = HL  - Vgm_HL
                HRg  = HR  - Vgm_HR
                VCLg = VCL - Vgm_VCL
                VCRg = VCR - Vgm_VCR
                TLg  = TL  - Vgm_TL
                TRg  = TR  - Vgm_TR

                dos_cb, trans_cb = self.dos_t_calculation_cb(E, Ef, dimC, HLg, \
                                                             TLg, SL, STL, HRg,\
                                                             TRg, SR, STR,\
                                                             SCL, VCLg, SCR,\
                                                             VCRg,\
                                                             n_occ_extra,\
                                                             U_cb_mat_o,\
                                                             U_cb_mat_u, SC,\
                                                             HCg,\
                                                             n_occ_extra_1,\
                                                             Ef_central)

                vg2print = Vg * np.ones((self.NE))

                dost2print = np.column_stack((dos_cb, trans_cb))
                dost2print = df(data=dost2print, columns=["Dos(E)", "T(E)"])
                dost2print.insert(loc=0, column="Gate Voltage", value=vg2print)
                dost2print.insert(loc=1, column="Energy", value=E.real)
                print (dost2print)

                fmt_01 = "% -05.7f" "% -05.7f" "% -05.7f" "% -05.7f"
                np.savetxt(f1, dost2print, delimiter='  ', fmt=fmt_01)
                f1.write("\n")
                f1.flush()
                
                print (moprint.iprint_line())
                print (moprint.iprint_line())
                print ("\n",flush=True,end='\n')

            # Check memory stats
            memoryHCg = HCg.size*HCg.itemsize
            shapeHCg  = HCg.shape
            info05 = ("Size / itemsize / shape / Memory(GB) of largest "
                      "matrix to invert: {:} / {:} / {:} / {:}")
            print (moprint.ibordered(info05.format(HCg.size,\
                   HCg.itemsize, shapeHCg, memoryHCg*1e-9)), \
                   flush=True,end='\n')
                
            if spin == "alpha":
                stophalf    = time.time()
                temphalf    = stophalf-start
                hourshalf   = temphalf//3600
                minuteshalf = temphalf//60
                secondshalf = temphalf - 60*minuteshalf

                info06 = ("Time for half of calculation (up(alpha) spin):"
                          "{:.0f}:{:.0f}:{:.0f} h/m/s")
                print (moprint.ibordered(info06.format(hourshalf,minuteshalf,\
                       secondshalf)),flush=True,end='\n')
                print ("\n\n")
            else:
                pass
        if spin == "beta":
            stop    = time.time()
            temp    = stop-start
            hours   = temp//3600
            minutes = temp//60 - hours*60
            seconds = temp - 60*minutes
            info06  = ("Entire time for NEGF+DFT+Coulomb blockade method "
                       "(up(alpha) and down(beta) spin): "
                       "{:.0f}:{:.0f}:{:.0f} h/m/s")
            print (moprint.ibordered(info06.format(hours,minutes,seconds)),\
                   flush=True,end='\n')
        else:
            pass
    #@profile
    def energy_levs_shifted(self, HC, SC, PC, Vg, shift_v):
        # Obtain the matrices: s and U for the orthogonalization procedure
        HCg = HC - (Vg * SC)

        # Obtain the X_s matrix that fullfils: X_s^{dagger} S_overlap X_s = 1
        Xsmatrix = self.orthogonalization_basis(SC)

        vPC   = scipy.linalg.eigvals(PC @ SC)
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

        # Fuer Test: Fermi level ist gleich als 0 (Vsd=0)
        Ef_central = 0

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

        HC_U_diag = np.diag(wHC_U_ortho)
        return HC_U, HC_U_diag, vHC_U_ortho
        
    def orthogonalization_basis(self, SC):
        vs, Us    = scipy.linalg.eigh(SC)
        seig     = np.diag(vs)
        seigsq   = np.sqrt(np.linalg.inv(seig))
        Xsmatrix = Us @ seigsq @ np.matrix.getH(Us)
        return Xsmatrix

    def dos_t_calculation_cb(self, Energies, Ef, dimC, HL, TL, SL, STL,\
                             HR, TR, SR, STR, SCL, VCL, SCR, VCR, n_occ_extra,\
                             U_cb_mat_o, U_cb_mat_u, SC, HCg, n_occ_extra_1,\
                             mu):
        dos_cb   = np.zeros(len(Energies))
        trans_cb = np.zeros(len(Energies))

        ave_timeE = 0

        for iE, energy in enumerate(Energies):
            startE = time.time()

            # Accuracy of sancho method 
            eps     = 1E-4
            energy1 = energy + Ef
            
            sigmaL  = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            sigmaR  = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            sigmacb = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            Omega_m = np.zeros(shape=(dimC,dimC),dtype=np.complex)

            HCg_effective    = np.zeros(shape=(dimC,dimC), \
                                        dtype=np.complex)
            HCg_effective_cb = np.zeros(shape=(dimC,dimC), \
                                        dtype=np.complex)

            # Greens function of semi infnite left/right electrode
            gL = mo.sancho(energy1, HL, TL, SL, STL, eps)
            gR = mo.sancho(energy1, HR, TR, SR, STR, eps)
            
            # Compute self-energy of left/right electrode
            sigmaL = (energy1*SCL-VCL) \
                     @ gL @ np.matrix.getH(energy1*SCL-VCL)
            sigmaR = (energy1*SCR-VCR) \
                     @ gR @ np.matrix.getH(energy1*SCR-VCR)

            # Omega matrix
            Omega_m = n_occ_extra * U_cb_mat_o \
                      @ scipy.linalg.inv(energy1*SC - HCg - \
                      n_occ_extra * U_cb_mat_o - \
                      n_occ_extra_1 * U_cb_mat_u)

            # 'Self' energy of the Coulomb blockade
            sigma_cb = ((energy1*SC - HCg) @ Omega_m) @ \
                       scipy.linalg.inv(np.eye(dimC) @ SC + Omega_m)

            # Effective Hamiltonian for the Greens function
            #HCg_effective = HCg_s + sigmaL + sigmaR

            # Effective interactive Hamiltonian for the Greens function
            #HCg_effective_cb = HCg_cb + sigma_cb + sigmaL + sigmaR
            HCg_effective_cb = HCg + sigma_cb + sigmaL + sigmaR

            # Calculate the central greens function non-interacting
            #G = np.linalg.solve(energy1*SC - HCg_effective, \
            #                    np.identity(dimC)) 
            
            # Compute the interacting central Greens function with 
            # parameter U and self energy of the leads
            G_cb = np.linalg.solve(energy1*SC - HCg_effective_cb, \
                                   np.identity(dimC)) 
            
            # Calculate broadening matrices 
            gammaL = 1j*(sigmaL-np.matrix.getH(sigmaL))
            gammaR = 1j*(sigmaR-np.matrix.getH(sigmaR))
            #gammaLR = 2 * gammaL @ gammaR / (gammaL + gammaR)
            
            #############################################################
            # Calculate transmission and dos
            #dos[iE]   = -1/np.pi * (np.trace(G @ SC)).imag
            #trans[iE] = np.trace(gammaL @ np.matrix.getH(G) @ \
            #            gammaR @ G).real
            #############################################################

            dos_cb[iE]   = -1/np.pi * (np.trace(G_cb @ SC)).imag
            trans_cb[iE] = np.trace(gammaL @ np.matrix.getH(G_cb) \
                                    @ gammaR @ G_cb).real
            endE  = time.time()
        tempE = endE - startE
        ave_timeE += tempE

        ave_houE = (ave_timeE/len(Energies)) // 3600
        ave_minE = (ave_timeE/len(Energies)) // 60 - ave_houE*60
        ave_secE = (ave_timeE/len(Energies)) - 60*ave_minE
        info04 = ("Average time per energy point: {} "\
                  "{:.0f}:{:.0f}:{:.1f} h/m/s")
        print (info04.format(5*' ', ave_houE , ave_minE, ave_secE,\
                             flush=True,end='\n'))

        return dos_cb, trans_cb


class FET_DOST_Spin_fort:
    ''' This class performs the same operations as the regular
    *fourteen_matrices_spin* class but uses fortran subroutines for more
    efficient Linear Algebra.
    '''

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
            self.pDOS = config["Projected DOS"]
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
        
        self.structure  = sdef.SystemDef(config)
    
    def load_matrices_h5_sp(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each. Single particle case.
        H_all = h5py.File(self.path_in+"/H_all.h5", "r")
        S_all = h5py.File(self.path_in+"/S_all.h5", "r")
        return H_all, S_all
   
    def load_FermiE(self):
        # Load the chemical potential.
        Ef = np.loadtxt(self.path_in+"/Ef")
        return Ef

    #@profile
    def NEGF(self):
        ''' Tasks of this method:
        For the python part:
            (*) Decimation of the semi-infinite electrodes into the 
                self-energies sigmaL and sigmaR using the Sancho method.
        In the fortran part:
            (*) Decorating the Central Hamiltonian with the self-energies.
            (*) Calculating the DOS and Transmission of the Central region.
        '''

        start = time.time()

        info00 = ("Transport NEGF+DFT. Sequential calculation. \n"
                  "System name: {}.\nFortran supported")

        print (moprint.ibordered_title(info00.format(self.Sname)),
               flush=True, end='\n')

        path = self.path_out
        if not os.path.exists(path):
             os.makedirs(path)

        # Load the matrices with h5py.File()
        H_all, S_all = self.load_matrices_h5_sp()
            
        Ef       = self.load_FermiE()
        Ef_alpha = Ef[0]
        Ef_beta  = Ef[1]
        Ef       = (Ef_alpha + Ef_beta)/2
     
        # Initialization of energy array and add small imaginary part 
        # calculate retarded quantities
        E  = np.linspace(self.Ea, self.Eb, self.NE,dtype=complex)
        E += 1j*self.eta

        if self.pDOS in ["No","no","N","n"]:
            for spin in ["alpha","beta"]: 
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
                HC  = np.array(H_all.get("HC-"+spin), order='F', copy=False)
                SC  = np.array(S_all.get("SC"), order='F', copy=False)

                dimC = HC.shape[0]

                print ("#Energy" + 7 * ' ' + 'Used time (sec)',
                       flush=True, end='\n')
                print (moprint.iprint_line(), flush=True, end='\n')

                ave_timeE = 0.

                with open(path+"/out-"+spin+"-sp-ng.out", "w+") as f1:
                    for iE,energy in enumerate(E):
                        startE = time.time()

                        # Sancho method
                        # Accuracy of Sancho method
                        eps     = 1E-4
                        energy1 = energy + Ef

                        # Greens function of semi infnite left/right electrode
                        gL = mo.sancho(energy1, HL, TL, SL, STL, eps)
                        gR = mo.sancho(energy1, HR, TR, SR, STR, eps)
                        
                        # Compute self-energy of left/right electrode.
                        sigmaL = (energy1*SCL-VCL)\
                                 @ gL @ np.matrix.getH(energy1*SCL-VCL)
                        sigmaR = (energy1*SCR-VCR)\
                                 @ gR @ np.matrix.getH(energy1*SCR-VCR)
                        # Compute DOS and T(E) of the central system.
                        dos, trans = GC.dost(energy1, sigmaL, sigmaR, HC, SC,
                                             dimC)

                        endE       = time.time()
                        tempE      = endE - startE
                        ave_timeE += tempE

                        print (iE, 5*'  ', int(tempE % 60), dos, trans,
                               flush=True, end="\n")
                        f1.write("{: 09.7f} {: 09.6f} {: 9.6f}".\
                                 format(energy.real, dos, trans))
                        f1.write("\n")
                        f1.flush()

                print (moprint.iprint_line(), flush=True, end='\n')

                ave_houE = (ave_timeE/self.NE) // 3600
                ave_minE = (ave_timeE/self.NE) // 60 - ave_houE*60
                ave_secE = (ave_timeE/self.NE) - 60*ave_minE
                info04 = ("Average time per energy point: {} "\
                          "{:.0f}:{:.0f}:{:.1f} h/m/s")
                print (info04.format(5*' ', ave_houE , ave_minE, ave_secE,
                       flush=True, end='\n'))
                print (moprint.iprint_line(), flush=True, end='\n')
            
                # Check memory stats
                if spin == "alpha":
                    memoryHC = HC.size*HC.itemsize
                    shapeHC  = HC.shape

                    stophalf    = time.time()
                    temphalf    = stophalf-start
                    hourshalf   = temphalf//3600
                    minuteshalf = temphalf//60
                    secondshalf = temphalf - 60*minuteshalf

                    info01 = ("Size / itemsize / shape / Memory(Kb) of " \
                              "largest matrix to invert:\n"
                              "{:} / {:} / {:} / {:}")
                    print (moprint.ibordered(info01.format(HC.size, 
                           HC.itemsize, shapeHC,\
                           memoryHC/1000)), flush=True, end='\n')

                    info02 = ("Time for half of calculation (up(alpha) spin):"\
                              "{:.0f}:{:.0f}:{:.0f} h/m/s")
                    print (moprint.ibordered(info02.format(hourshalf,\
                           minuteshalf, secondshalf)), flush=True, end='\n')

                    print ("\n")

                elif spin == "beta":
                    stop    = time.time()
                    temp    = stop-start
                    hours   = temp//3600
                    minutes = temp//60 - hours*60
                    seconds = temp - 60*minutes
                    info03  = ("Entire time for NEGF method (up(alpha) and " \
                               "down(beta) spin): {:.0f}:{:.0f}:{:.0f} h/m/s")
                    print (moprint.ibordered(info03.format(hours,minutes,
                           seconds)), flush=True, end='\n')
            
        elif self.pDOS in ["Yes","yes","Y","y"]:
            structure     = self.structure
            p2f           = structure.path2files
            CP2Koutfile   = structure.CP2Koutfile
            dic_elem, geo = structure.get_region_geometry(p2f+CP2Koutfile)
            iatomC        = structure.iatomC
            fatomL        = structure.fatomLleads

            SC_sys1    = np.copy(SC)
            SC_sys2    = np.copy(SC)
            dos_sys1   = np.zeros(self.NE)
            dos_sys2   = np.zeros(self.NE)

            if spin == "alpha":
                info00 = ("PDOS calculation. \nMolecule 1: {}" \
                          "\nMolecule 2: {} \n\n")
                print (info00.format(self.mol1,self.mol2), flush=True, 
                        end="\n")
                '''N_sX: is the number of elements in system X and it 
                         is the number of atoms times the number of
                         spherical basis functions as in CP2K basis sets.
                N_sXpX:  are the number or atoms of the system 1 before 
                         the first atom of the system 2 and after the 
                         last atom of system 2.
                N_s1_C:  is the number of atoms of system 1 before the 
                         first atom in system 2 w.r.t the central region. 
                '''
                N_s1   = [(i,j) for i,j in geo 
                          if i < self.is2atom or i > self.fs2atom]
                N_s2   = [(i,j) for i,j in geo 
                          if i >= self.is2atom and i <= self.fs2atom]
                N_s1p1 = [(i,j) for i,j in geo 
                          if i >= self.is1atom and i < self.is2atom]
                N_s1p2 = [(i,j) for i,j in geo 
                          if i > self.fs2atom and i <= self.fs1atom]
                N_s1_C = [(i,j) for i,j in geo 
                          if i > fatomL and i <= (fatomL + (self.is2atom 
                          - iatomC))]

                elem_s1   = 0
                elem_s2   = 0
                elem_s1p1 = 0
                elem_s1p2 = 0
                elem_s1_C = 0
                
                for k,l in N_s1:
                    if l in dic_elem.keys():
                        elem_s1 += dic_elem[l] 

                for k,l in N_s2:
                    if l in dic_elem.keys():
                        elem_s2 += dic_elem[l] 
                
                for k,l in N_s1p1:
                    if l in dic_elem.keys():
                        elem_s1p1 += dic_elem[l] 
                
                for k,l in N_s1p2:
                    if l in dic_elem.keys():
                        elem_s1p2 += dic_elem[l] 
                
                for k,l in N_s1_C:
                    if l in dic_elem.keys():
                        elem_s1_C += dic_elem[l] 

                #print ("Elements in system 1: ",elem_s1, "\n","Elements\
                #       in system 2: ",elem_s2, sep='')
                #print ("Elements in system 1, part 1: ",elem_s1p1, "\n",\
                #       "Elements in system 1, part 2: "\
                #       ,elem_s1p2, sep='')
            else:
                pass

            # All the matrix elements that are not part of the 1st system,
            # are set to 0. 
            for i in list(range(elem_s1_C,elem_s1_C+elem_s2+1)):
                for j in list(range(elem_s1_C,elem_s1_C+elem_s2+1)):
                    SC_sys1[i][j] = 0.

            # All the matrix elements that are not part of the 2nd system,
            # are set to 0. 
            for i in list(range(elem_s1_C+1)):
                for j in list(range(elem_s1_C+1)):
                    SC_sys2[i][j] = 0.
            
            for i in list(range(elem_s1_C+elem_s2, SC.shape[0])):
                for j in list(range(elem_s1_C+elem_s2, SC.shape[1])):
                    SC_sys2[i][j] = 0.
            
            f1 = open (path+"/out-"+spin+"-PDOS"+".out", "w+")

            print ("Energy" + 7 * ' ' + 'Used time', flush=True, end='\n')
            print (moprint.iprint_line(), flush=True, end='\n')
            
            for iE,energy in enumerate(E):
                # Sancho method
                # Accuracy of sancho method 
                eps     = 1E-4
                shift   = -0.05
                energy1 = energy + Ef + shift

                startsancho = time.time()
                startE      = time.time()
                
                sigmaL       = np.zeros(shape=(dimC, dimC), dtype=np.complex)
                sigmaR       = np.zeros(shape=(dimC, dimC), dtype=np.complex)
                HC_effective = np.zeros(shape=(dimC, dimC), dtype=np.complex)

                #Greens function of semi infnite left/right electrode
                gL = mo.sancho(energy1, HL, TL, SL, STL, eps)
                gR = mo.sancho(energy1, HR, TR, SR, STR, eps)
                
                endsancho = time.time()
                tempS     = endsancho - startsancho
                hoursS    = tempS//3600
                minutesS  = tempS//60
                secondsS  = tempS - 60*minutesS
                
                # Compute self-energy of left/right electrode
                sigmaL = (energy1*SCL-VCL) @ gL \
                         @ np.matrix.getH(energy1*SCL-VCL)
                sigmaR = (energy1*SCR-VCR) @ gR \
                         @ np.matrix.getH(energy1*SCR-VCR)

                HC_effective = HC + sigmaL + sigmaR
                
                # Calculate greens function of central system with effect 
                # of left and right electrodes via corrected hamiltonian
                G = np.linalg.inv(energy1*SC - HC_effective)
                
                # Calculate broadening matrices 
                gammaL = 1j*(sigmaL - np.matrix.getH(sigmaL))
                gammaR = 1j*(sigmaR - np.matrix.getH(sigmaR))
                
                # Calculate transmission and dos
                trans[iE] = (np.trace(gammaL @ np.matrix.getH(G) @ gammaR 
                             @ G).real)
                #trans_sys1[iE] = (np.trace(gammaL @ np.matrix.getH(G) 
                #                   @ gammaR @ G).real)
                #trans_sys2[iE] = (np.trace(gammaL @ np.matrix.getH(G) 
                #                   @ gammaR @ G).real)
                
                #dos[iE]       = -1./np.pi * (np.trace(G @ SC)).imag
                dos_sys1[iE] = -1./np.pi * (np.trace(G @ SC_sys1)).imag
                dos_sys2[iE] = -1./np.pi * (np.trace(G @ SC_sys2)).imag

                endE     = time.time()
                tempE    = endE - startE
                hoursE   = tempE//3600
                minutesE = tempE//60 - hoursE*60
                secondsE = tempE - 60*minutesE
                print ("{: 3.5f}".format(energy.real) + 5 * ' ' +
                       "{:.0f}:{:.0f}:{:.1f} h/m/s".format(hoursE, \
                       minutesE, secondsE), flush=True, end='\n')
                f1.write("{: 09.6f} {: 012.6f} {: 012.6f} {: 9.6f}\n"\
                        .format(energy.real, dos_sys1[iE], dos_sys2[iE],\
                        trans[iE]))
                f1.flush()

            if spin == "alpha":
                stophalf    = time.time()
                temphalf    = stophalf-start
                hourshalf   = temphalf//3600
                minuteshalf = temphalf//60
                secondshalf = temphalf - 60*minuteshalf
                info01 = ("Time for half calculation (up(alpha) spin): "
                          "{:2.0f}:{:2.0f}:{:2.0f} h/m/s")
                print (info01.format(hourshalf, minuteshalf, secondshalf),
                       flush=True, end="\n")

                # Check memory stats
                memoryHC_eff = HC_effective.size*HC_effective.itemsize
                shapeHC_eff  = HC_effective.shape
                info02 =("Size / itemsize / shape / Memory(Kb) of largest "
                         "matrix to invert: {:} / {:} / {:} / {:}\n")
                print (info02.format(HC_effective.size, HC_effective.itemsize,
                       shapeHC_eff, memoryHC_eff/1000), flush=True,\
                       end="\n") 
            elif spin == "beta":
                stop    = time.time()
                temp    = stop - start
                hours   = temp//3600
                minutes = temp//60 - hours*60
                seconds = temp - 60*minutes
                info03  = ("Entire time for NEGF-PDOS method: "
                           "{:.0f}:{:.0f}:{:.0f} h/m/s")
                print (info03.format(hours, minutes, seconds), \
                       flush=True, end='\n')
            else:
                pass

            f1.close()

