import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import os
import sys
import modules.mo as mo


class fourteen_matrices:
    '''
    This class contains all relevant methods to calculate the DOS and transmission of a quantum system connected to two periodic electrodes.
    
    --HC,SC is the on-site hamiltonian, overlap of the quantum system.
    --VCL,SCL is the hopping hamiltonian, overlap from the center to the left.
    --VCR,SCR is the hopping hamiltonian, overlap from the center to the right.
    --HL,SL is the on-site hamiltonian, overlap of the left electrode.
    --TL,STL is the hopping hamiltonian, overlap of the left electrode
    --HR,SR is the on-site hamiltonian, overlap of the right electrode.
    --TR,STR is the hopping hamiltonian, overlap of the right electrode.
    
    The direction of the hopping matrices have to be from the center to the left/right, i.e. TL & VCL go to the left, TR & VCR go to the right.
    And the same for the hopping overlaps.
    '''
    
    def __init__(self,config):
        self.Sname    = config["System name"]
        self.NE       = config["Number of energy points"]
        self.Ea       = config["Lower energy border"]
        self.Eb       = config["Upper energy border"]
        self.Restart  = config["Restart calculation"]
        self.path_in  = config["Path to the system 14-matrices"]
        self.eta      = config["Small imaginary part"]
        self.path_out = config["Path of output"]
    
    
    def load_electrodes(self):
        '''
        Loading matrices representing the left/ right electrode and the coupling from the quantum region to each.
        '''

        HL = np.loadtxt(self.path_in+"/HL.dat")
        SL = np.loadtxt(self.path_in+"/SL.dat")
        
        HR = np.loadtxt(self.path_in+"/HR.dat")
        SR = np.loadtxt(self.path_in+"/SR.dat")
        
        VCL = np.loadtxt(self.path_in+"/VCL.dat")
        SCL = np.loadtxt(self.path_in+"/SCL.dat")
        
        VCR = np.loadtxt(self.path_in+"/VCR.dat")
        SCR = np.loadtxt(self.path_in+"/SCR.dat")
        
        TL = np.loadtxt(self.path_in+"/TL.dat")
        STL = np.loadtxt(self.path_in+"/STL.dat")
        
        TR = np.loadtxt(self.path_in+"/TR.dat")
        STR = np.loadtxt(self.path_in+"/STR.dat")
        
        return HL,SL,HR,SR,VCL,SCL,VCR,SCR,TL,STL,TR,STR
    
    
    def load_center(self):
        '''
        Load the matrices representing the quantum region
        '''
        HC = np.loadtxt(self.path_in+"/HC.dat")
        SC = np.loadtxt(self.path_in+"/SC.dat")
        
        return HC,SC
   
    
    def load_FermiE(self):
        '''
        Load the chemical potential
        '''
        Ef = np.loadtxt(self.path_in+"/Ef.dat")
        return Ef

    
    def NEGF(self):
        restart_true  = ["YES","Yes","yes","Y","y","true"]
        restart_false = ["NO","No","no","N","n","false"]
        
        start = time.time() 
        print ("Parallel calculation")
        print ("Restart: ", str(self.Restart))
        print ("System name : ", self.Sname)
        '''
        Tasks of this method:
        i) Decimation of the semi-infinite electrodes into the self-energies sigmaL and sigmaR using the sancho method.
        ii) Decorating the quantum region hamiltonian with the self-energies.
        iii) Calculating the DOS and transmission of the quantum region.
        '''
        start = time.time() 
        path = self.path_out
        if not os.path.exists(path):
             os.makedirs(path)
       

        #Options for parallelization
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
       

        #Options for restarting a calculation
        if self.Restart in restart_false:
            #Init energy range
            E = np.linspace(self.Ea,self.Eb,self.NE,dtype=complex)
            #Init DOS and transmission
            dos = np.zeros(self.NE)
            trans = np.zeros(self.NE)
            
            perrank = ceil(self.NE/size)
            print ("size", size) 
            print ("rank", rank)
            divL = abs(self.Ea-self.Eb)/self.NE
            EapR = rank*perrank*(divL)            
            EbpR = (rank+1)*perrank*(divL)            
            E    = np.linspace(self.Ea+EapR,self.Ea+EbpR,perrank,endpoint=False,dtype=complex)
        
        elif self.Restart in restart_true:
            El2 = np.linspace(self.Ea,self.Eb,self.NE,dtype=complex)
            El2 = list(El2)
            
            with open (path+"/"+"out.dat") as restart_f:
                E_old = [] 
                for line in restart_f.readlines():
                    E_old.append(line.split(' ',1)[0])
            restart_f.close()
            
            for k in range(len(E_old)):
                E_old[k] = float(E_old[k])
            
            E_new =[]
            for i in range(len(El2)):
                if float(round(El2[i].real,5)) not in E_old:
                    E_new.append(El2[i])
            #Init energy range
            E = np.linspace(min(E_new),max(E_new),len(E_new),dtype=complex)
            #init DOS and transmission
            dos = np.zeros(len(E_new))
            trans = np.zeros(len(E_new))
            
            perrank = ceil(len(int(E_new))/size)
            print ("size", size) 
            print ("rank", rank)
            divL = abs(min(E_new)-max(E_new))/len(int(E_new))
            EapR = rank*perrank*(divL)
            EbpR = (rank+1)*perrank*(divL)            
            E    = np.linspace(min(E_new)+EapR,min(E_new)+EbpR,perrank,endpoint=False,dtype=complex)
        
        HL,SL,HR,SR,VCL,SCL,VCR,SCR,TL,STL,TR,STR = self.load_electrodes()
        HC,SC = self.load_center()
        Ef    = self.load_FermiE()
        
        #Add small imaginary part calculate retarded quantities
        
        E   += 1j*self.eta
        dimC = HC.shape[0]
     
        '''
        Initialize DOS and transmission: Each of the arrays has to have the dimensions of the energy points in the
        specific process
        '''
        dos = np.zeros(perrank)
        trans = np.zeros(perrank)
        
        '''
        If the process is a master, create the arrays for allocating the total energy points, dos and
        transmission. The size of each of the all_* arrays has depends on the number of nodes and processes
        per node used. If it is not a integer multiple of the total number of energy points, it will not
        calculate the last points (Typically 3 or 4 points)
        '''
        if rank == 0:
            all_energy = np.zeros(shape=size*perrank, dtype=complex)
            all_dos    = np.zeros(shape=size*perrank, dtype=dos.dtype)
            all_trans  = np.zeros(shape=size*perrank, dtype=trans.dtype)
        else:
            all_energy = None
            all_dos    = None
            all_trans  = None


        if self.Restart in restart_false:
            f1 = open (path+"/out.dat", "w+")
        elif self.Restart in restart_true:
            f1 = open (path+"/out.dat", "a+")

        #sancho 
        for iE,energy in enumerate(E):
            #Accuracy of sancho method 
            eps = 1E-4
            energy1 = energy+Ef
            #Init self-energies as functions of energy. They have to have the same dimension as the quantum region hamiltonian.
            sigmaL = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            sigmaR = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            HC_effective = np.zeros(shape=(dimC,dimC),dtype=np.complex)
            
            #Green function of semi infnite left/right electrode
            gL = mo.sancho(energy1,HL,TL,SL,STL,eps)
            gR = mo.sancho(energy1,HR,TR,SR,STR,eps)
            
            #Compute self-energy of left/right electrode
            sigmaL = (energy1*SCL-VCL) @ gL @ np.matrix.getH(energy1*SCL-VCL)
            sigmaR = (energy1*SCR-VCR) @ gR @ np.matrix.getH(energy1*SCR-VCR)
            
            HC_effective = HC + sigmaL + sigmaR
            
            #Calculate greens function of central system with effect of left and right electrodes via corrected hamiltonian
            G = np.linalg.inv(energy1*SC - HC_effective)
 
            #Calculate broadening matrices 
            gammaL = 1j*(sigmaL-np.matrix.getH(sigmaL))
            gammaR = 1j*(sigmaR-np.matrix.getH(sigmaR))
            
            #Calculate transmission and dos
            trans[iE] = np.trace(gammaL @ np.matrix.getH(G) @ gammaR @ G).real
            dos[iE]   = -1/np.pi * (np.trace(G @ SC)).imag
        
        '''
        Check memory stats
        '''
        memoryHC_eff = HC_effective.size*HC_effective.itemsize
        shapeHC_eff = HC_effective.shape
        sizeofHC_eff = sys.getsizeof(HC_effective)
        print ("Size / itemsize / shape / sys.getsizeof(Kb) /  Memory(Kb) of matrix to invert: {:} / {:} / {:} \
                / {:} / {:} \n\n".format(HC_effective.size, HC_effective.itemsize, shapeHC_eff, sizeofHC_eff/1000, memoryHC_eff/1000))
        
        '''
        Get the results from every process into the root (master); Wait until all of them are finished to do
        so
        '''
        
        comm.Gather(E, all_energy, root=0)
        #comm.Barrier()
        comm.Gather(dos, all_dos, root=0)
        #comm.Barrier()
        comm.Gather(trans, all_trans, root=0)
        if rank == 0:
            np.savetxt(path+"/out.dat", np.c_[all_energy.real, all_dos, all_trans], fmt="%.5f")
            stop = time.time()
            temp = stop-start
            hours = temp//3600
            minutes = temp//60 - hours*60
            seconds = temp - 60*minutes
            print ("\n Entire time for NEGF method: {:.0f}:{:.0f}:{:.0f} h/m/s \n\n".format(hours,minutes,seconds))
