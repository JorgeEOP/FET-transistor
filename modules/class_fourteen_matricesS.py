import numpy as np
import matplotlib.pyplot as plt
import time
import os
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
        start = time.time() 
        print ("Sequential calculation")
        print ("System name : ", self.Sname)
        '''
        Tasks of this method:
        i) Decimation of the semi-infinite electrodes into the self-energies sigmaL and sigmaR using the sancho method.
        ii) Decorating the quantum region hamiltonian with the self-energies.
        iii) Calculating the DOS and transmission of the quantum region.
        '''
        path = self.path_out
        if not os.path.exists(path):
             os.makedirs(path)
        
        HL,SL,HR,SR,VCL,SCL,VCR,SCR,TL,STL,TR,STR = self.load_electrodes()
        HC,SC = self.load_center()
        Ef    = self.load_FermiE()
 
        #init energy range and add small imaginary part calculate retarded quantities
        E = np.linspace(self.Ea,self.Eb,self.NE,dtype=complex)
        E+= 1j*self.eta
        dimC = HC.shape[0]

        #init DOS and transmission
        dos = np.zeros(self.NE)
        trans = np.zeros(self.NE)
        
        f1 = open (path+"/out.dat", "w+")
        #sancho 
        for iE,energy in enumerate(E):
            #Accuracy of sancho method 
            eps = 1E-4
            energy1 = energy + Ef
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
            
            f1.write("%.5f %.12f %.12f\n" %(energy.real,dos[iE],trans[iE]))
            #f1.write("\n")
            f1.flush()
        f1.close() 
        stop = time.time()
        temp = stop-start
        hours = temp//3600
        minutes = temp//60 - hours*60
        seconds = temp - 60*minutes
        print ("Entire time for NEGF method: {:.1f}:{:.1f}:{:.1f} h/m/s".format(hours,minutes,seconds))
    

    def plot(self):
        Ef    = self.load_FermiE()
        dos = np.load(self.path_out+"/dos.dat",allow_pickle=True)
        trans = np.load(self.path_out+"/trans.dat",allow_pickle=True)
        
        
        path = self.path_out+"/Plot/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        E = np.linspace(self.Ea,self.Eb,self.NE)
        
        fig = plt.figure(1)

        #(xyz): x is the number of rows, y the number of columns and z the index of the subplot 
        ax1 = fig.add_subplot(211)
        plt.plot(E,dos, 'k', linewidth=1.5, fillstyle='full')
        plt.fill_between(E,dos, color='0.8')
        plt.xlim(left=min(E),right=max(E))
        plt.ylim(bottom=0)
        plt.ylabel(r"$D(E)$ [1/eV]")
        plt.grid(linewidth=0.4, linestyle='--')
        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
        fig.add_subplot(212, sharex=ax1)
        plt.plot(E,trans, 'k', linewidth=1.5)
        plt.xlim(left=min(E),right=max(E))
        plt.ylim(bottom=0)
        plt.xlabel(r"$(E-E_F)$ [eV]")
        plt.ylabel(r"$T(E)$")
        plt.grid(linewidth=0.4, linestyle='--')
        plt.tick_params(axis='x', which='both')
        
        plt.xticks(np.arange(self.Ea,self.Eb+0.5,0.5))
        plt.subplots_adjust(hspace=0.05)
        plt.show(1)
        plt.savefig(path+"/dos_trans.png",dpi=600)
