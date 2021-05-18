# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:32:08 2019

@author: OEM
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import modules.mo as mo


class fourteen_matrices:
    """
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
    """
    
    def __init__(self,config):
        self.NE = config["Number of energy points"]
        self.Ea = config["Lower energy border"]
        self.Eb = config["Upper energy border"]
        self.path_in = config["Path to the system 14-matrices"]
        self.eta = config["Small imaginary part"]
        self.path_out = config["Path of output"]
        
        
        
    def load_electrodes(self):
        """
        Loading matrices representing the left/ right electrode and the coupling from the quantum region to each.
        """

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
        """
        Load the matrices representing the quantum region
        """
        HC = np.loadtxt(self.path_in+"/HC.dat")
        SC = np.loadtxt(self.path_in+"/SC.dat")

        
        return HC,SC
   
    def load_FermiE(self):
        """
        Load the chemical potential
        """
        Ef = np.loadtxt(self.path_in+"/Ef.dat")
        return Ef

    def NEGF(self):
       """
       Tasks of this method:
       i) Decimation of the semi-infinite electrodes into the self-energies sigmaL and sigmaR using the sancho method.
       ii) Decorating the quantum region hamiltonian with the self-energies.
       iii) Calculating the DOS and transmission of the quantum region.
       """
       
       path = self.path_out
       if not os.path.exists(path):
            os.makedirs(path)
       
       
       HL,SL,HR,SR,VCL,SCL,VCR,SCR,TL,STL,TR,STR = self.load_electrodes()
       HC,SC = self.load_center()
       Ef    = self.load_FermiE()
       print (Ef)

       #init energy range and add small imaginary part calculate retarded quantities
       E = np.linspace(self.Ea,self.Eb,self.NE,dtype=complex)
       E+= 1j*self.eta
       dimC = HC.shape[0]
       
       #init self-energies as functions of energy. They have to have the same dimension as the quantum region hamiltonian.
       sigmaL = np.zeros([self.NE,dimC,dimC],dtype=complex)
       sigmaR = np.zeros([self.NE,dimC,dimC],dtype=complex)
       HC_effective = np.zeros([self.NE,dimC,dimC],dtype=complex)
       
       #init DOS and transmission
       dos = np.zeros(self.NE)
       trans = np.zeros(self.NE)
       #sancho 
       
       for iE,energy in enumerate(E):
           #accuracy of sancho method 
           eps = 1E-3
           #green function of semi infnite left electrode
           gL = mo.sancho(energy,HL,TL,SL,STL,eps)
           #green function of semi infnite right electrode
           gR = mo.sancho(energy,HR,TR,SR,STR,eps)
           #compute self-energy of left electrode
           sigmaL[iE] = (energy*SCL-VCL) @ gL @ np.matrix.getH(energy*SCL-VCL)
           
           #compute self-energy of right electrode
           sigmaR[iE] = (energy*SCR-VCR) @ gR @ np.matrix.getH(energy*SCR-VCR)
           
           HC_effective[iE] = HC + sigmaL[iE] + sigmaR[iE]
           
           #Calculate greens function of central system with effect of left and right electrodes via corrected hamiltonian
           G = np.linalg.inv(energy*SC - HC_effective[iE])
           #Calculate broadening matrices 
           gammaL = 1j*(sigmaL[iE]-np.matrix.getH(sigmaL[iE]))
           gammaR = 1j*(sigmaR[iE]-np.matrix.getH(sigmaR[iE]))
           
           #Calculate transmission and dos
           trans[iE] = np.trace(gammaL @ np.matrix.getH(G) @ gammaR @ G).real
           dos[iE]   = -1/np.pi * np.trace(np.imag(G))
           
       dos.dump(path+"/dos.dat")
       trans.dump(path+"/trans.dat")
       Ep = np.linspace(self.Ea -Ef,self.Eb - Ef,self.NE)
       np.savetxt(path+"/out.dat", np.c_[Ep, dos, trans])
       
   
    
    def plot(self):
        dos = np.load(self.path_out+"/dos.dat",allow_pickle=True)
        trans = np.load(self.path_out+"/trans.dat",allow_pickle=True)
        
        
        path = self.path_out+"/Plot/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        E = np.linspace(self.Ea,self.Eb,self.NE)
        
        plt.figure()
        plt.plot(E,dos)
        plt.xlabel(r"$(E-E_F)$ in eV")
        plt.ylabel(r"$D(E)$ in 1/eV")
        plt.grid()
        plt.savefig(path+"/dos.png",dpi=600)
        
        plt.figure()
        plt.plot(E,trans)
        plt.xlabel(r"$(E-E_F)$ in eV")
        plt.ylabel(r"$T(E)$")
        plt.grid()
        plt.savefig(path+"/trans.png",dpi=600)
