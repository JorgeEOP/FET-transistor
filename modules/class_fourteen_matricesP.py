import modules.mo_print as moprint
import matplotlib.pyplot as plt
import modules.mo as mo
import numpy as np
import time
import sys
import os
from mpi4py import MPI
from math import ceil

class fourteen_matrices:
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
    
    def load_electrodes(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each.

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

    def load_matrices_h5(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each. Coulomb Blockade case
        H_all = h5py.File(self.path_in+"/H_all.h5", "r")
        S_all = h5py.File(self.path_in+"/S_all.h5", "r")
        PC    = h5py.File(self.path_in+"/PC.h5", "r")
        return H_all, S_all, PC

    def load_matrices_h5_sp(self):
        # Loading matrices representing the left/right electrode and the 
        # coupling from the quantum region to each. Single Particle case.
        H_all = h5py.File(self.path_in+"/H_all.h5", "r")
        S_all = h5py.File(self.path_in+"/S_all.h5", "r")
        return H_all, S_all
    
    def load_center(self):
        # Load the matrices representing the Central Region.
        HC = np.loadtxt(self.path_in+"/HC.dat")
        SC = np.loadtxt(self.path_in+"/SC.dat")
        return HC,SC
    
    def load_FermiE(self):
        # Load the chemical potential
        Ef = np.loadtxt(self.path_in+"/Ef")
        return Ef
    
    def NEGF(self):
        # Parallel Version of NEGF method.
        start = time.time() 

        path = self.path_out
        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        Energies_p_rank = []

        if rank == 0:
            info00 = ("NEGF+DFT. Parallel calculation. RKS. \nSystem name: {}")
            print (moprint.ibordered_title(info00.format(self.Sname)),\
                   flush=True,end='\n')

            name = MPI.Get_processor_name()
            
            info01 = ("Name of the Processor(Node): {} ; Process: {}")
            print (moprint.ibordered(info01.format(name, rank,\
                   Energies_p_rank)), flush=True, end='\n')
            
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
            info01   = "Name of the Processor(Node): {} ; Process: {}"
            print (moprint.ibordered(info01.format(name, rank,\
                   Energies_p_rank)), flush=True, end='\n')

        Energies_p_rank = comm.scatter(Energies_p_rank, root=0)

        H_all, S_all, PC_all = self.load_matrices_h5()

        PC_all = None

        HL  = np.array(H_all.get("HL"))
        SL  = np.array(S_all.get("SL"))
        HR  = np.array(H_all.get("HR"))
        SR  = np.array(S_all.get("SR"))
        VCL = np.array(H_all.get("VCL"))
        SCL = np.array(S_all.get("SCL"))
        VCR = np.array(H_all.get("VCR"))
        SCR = np.array(S_all.get("SCR"))
        TL  = np.array(H_all.get("TL"))
        STL = np.array(S_all.get("STL"))
        TR  = np.array(H_all.get("TR"))
        STR = np.array(S_all.get("STR"))
        HC  = np.array(H_all.get("HC"))
        SC  = np.array(S_all.get("SC"))
        
        Ef       = self.load_FermiE()

        dimC = HC.shape[0]
        
        if rank == 0:
            info02 = ("Fermi Energy (eVs): {: 4.5f} eV\n")
            print (moprint.ibordered(info02.format(Ef)),\
                   flush=True, end='\n')
        else:
            None

        if self.pDOS in ["No","no","N","n"]:
            dos   = np.zeros(self.NE)
            trans = np.zeros(self.NE)
            rank_string = str(rank)

            with open(path+"/out-RKS"+"_"+rank_string+"-sp.out", "w+")\
            as f1_local:
                # Sancho decimation method
                Energies_p_rank += 1j*self.eta
                for iE, energy in enumerate(Energies_p_rank):
                    startE = time.time()
                    #Accuracy of sancho method 
                    eps = 1E-4

                    energy1 = energy + Ef

                    sigmaL = np.zeros(shape=(dimC,dimC),dtype=np.complex)
                    sigmaR = np.zeros(shape=(dimC,dimC),dtype=np.complex)
                    HC_effective = np.zeros(shape=(dimC,dimC), \
                                            dtype=np.complex)

                    #Green function of semi infnite left/right electrode
                    gL = mo.sancho(energy1,HL,TL,SL,STL,eps)
                    gR = mo.sancho(energy1,HR,TR,SR,STR,eps)
                    
                    #Compute self-energy of left/right electrode
                    sigmaL = (energy1*SCL-VCL) @ gL \
                             @ np.matrix.getH(energy1*SCL-VCL)
                    sigmaR = (energy1*SCR-VCR) @ gR \
                             @ np.matrix.getH(energy1*SCR-VCR)
                                    
                    HC_effective = HC + sigmaL + sigmaR
                    
                    # Calculate greens function of central system with 
                    # effect of left and right electrodes via corrected
                    # Hamiltonian
                    G = scipy.linalg.solve(energy1*SC - HC_effective, \
                                           np.identity(dimC))
                    
                    #Calculate broadening matrices 
                    gammaL = 1j*(sigmaL-np.matrix.getH(sigmaL))
                    gammaR = 1j*(sigmaR-np.matrix.getH(sigmaR))

                    #Calculate transmission and dos
                    trans[iE] = np.trace(gammaL @ np.matrix.getH(G) \
                                @ gammaR @ G).real
                    dos[iE]   = -1/np.pi * (np.trace(G @ SC)).imag

                    endE  = time.time()
                    tempE = endE - startE

                    f1_local.write("{: 09.6f} {: 012.6f} {: 9.6f}\n".\
                                   format(energy.real, dos[iE], trans[iE]))
                    f1_local.flush()

                    #f1_local.write("\n")
                    if rank == 0:
                        houE = tempE // 3600
                        minE = tempE // 60 - houE*60
                        secE = tempE - 60 * minE

                        info04 = ("Time per energy point: {} "\
                                  "{:.0f}:{:.0f}:{:.1f} h/m/s")
                        print (info04.format(5*' ', houE , minE, secE,\
                                             flush=True,end='\n'))
                        print (moprint.iprint_line(), flush=True, end='\n')
                        print ("\n",flush=True,end='\n')
                    else:
                        None
        
            '''
            Check memory stats
            '''
            memoryHC_eff = HC_effective.size*HC_effective.itemsize
            shapeHC_eff  = HC_effective.shape
            sizeofHC_eff = sys.getsizeof(HC_effective)
            print ("rank", rank)
            print ("size", size) 
            print ("Size / itemsize / shape / sys.getsizeof(Kb) / "
                   "Memory(Kb) of matrix to invert: {:} / {:} / "
                   "{:} / {:} / {:} \n".format(HC_effective.size,\
                   HC_effective.itemsize, shapeHC_eff,\
                   sizeofHC_eff/1000, memoryHC_eff/1000))               
            
        elif self.pDOS in ["Yes","yes","Y","y"]:
            print ("Parallel PDOS-simulation not implemented")
            sys.exit()

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
