import scipy.constants as phys
import numpy as np
import itertools
import time
import sys
import re
from pandas import DataFrame as df

class SystemDef:
    def __init__(self, cfg):
        self.Sname              = cfg["System name"]
        self.Spin               = cfg["Spin Polarized system"]
        self.iatomC             = cfg["Central system"][0]["first atom"]
        self.fatomC             = cfg["Central system"][1]["last atom"]
        self.iatomLleads        = cfg["Leads"][0]['first atom left Lead']
        self.fatomLleads        = cfg["Leads"][1]['last atom left Lead']
        self.iatomRleads        = cfg["Leads"][2]['first atom right Lead']
        self.fatomRleads        = cfg["Leads"][3]['last atom right Lead']
        self.Nperiods           = cfg["Number of periods on the Leads"]
        self.CP2K_ao_mat        = cfg["Name of the CP2K ao matrices"]
        self.CP2Koutfile        = cfg["Name of the .out CP2K file"]
        self.CP2Kpdosfile_alpha = cfg["Name of the .pdos alpha CP2K file"]
        self.CP2Kpdosfile_beta  = cfg["Name of the .pdos beta CP2K file"]
        self.path2files         = cfg["Path to CP2K ao matrices"]
        self.path_inp           = cfg["Path to the system 14-matrices"]
        try:
            self.CB_regime      = cfg["Coulomb blockade"]
        except KeyError:
            print ("No key value Coulomb blockade. Set to NO")
            self.CB_regime      = "No"
        try:
            self.pDOS           = cfg["Projected DOS"]
        except KeyError:
            print ("No key value Projected DOS. Set to NO")
            self.pDOS           = "No"

    # Reads the KOHN-SHAM and OVERLAP matrices from the CP2K *.ao file
    # It returns the KOHN-SHAM matrix in eV and the OVERLAP matrix (RKS).
    def readCP2K2file(self, CP2K_ao_mat):
        overlap_elements = []
        ks_elements      = []
        where_to_dump    = overlap_elements
        lines2ignore     = [1,2,3,4]
        with open (CP2K_ao_mat, "r") as f1:
            for line in f1:
                if "OVERLAP MATRIX" in line:
                    continue
                elif "KOHN-SHAM MATRIX" in line:
                    where_to_dump = ks_elements
                    continue
                Mat = line.split() 
                if len(Mat) == 0:
                    continue
                elif len(Mat) in lines2ignore:
                    where_to_dump.append([])
                    continue
                where_to_dump[-1].append(Mat[4:])
                
            Overlap = np.hstack(overlap_elements).astype(float)
            KS      = (phys.physical_constants["Hartree energy in eV"][0])\
                       *(np.hstack(ks_elements)).astype(float)
        return Overlap, KS

    # Reads specific orbital interactions. For example, if one is only
    # interested in pz-pz interaction, just will read those orbitals
    def readCP2K2file_porbit(self,CP2K_ao_mat):
        overlap_elements = []
        ks_elements      = []
        where_to_dump    = overlap_elements
        lines2ignore     = [1,2,3,4]
        with open (CP2K_ao_mat, "r") as f1:
            for line in f1:
                if "OVERLAP MATRIX" in line:
                    continue
                elif "KOHN-SHAM MATRIX" in line:
                    where_to_dump = ks_elements
                    continue
                Mat = line.split() 
                if len(Mat) == 0:
                    continue
                elif len(Mat) in lines2ignore:
                    where_to_dump.append([])
                    continue
                orbit  = re.search(r".*?(3pz) (.*\d)", line)
                orbit1 = re.search(r".*?(4pz) (.*\d)", line)
                if orbit:
                    #print (orbit)
                    where_to_dump[-1].append(list(orbit.group(2).split()))
                #if orbit1:
                    #print (orbit1)
                #    where_to_dump[-1].append(list(orbit1.group(2).split()))
                #else:
                #    continue
            Overlap = np.hstack(overlap_elements).astype(float)
            KS      = (phys.physical_constants["Hartree energy in eV"][0])\
                       *(np.hstack(ks_elements)).astype(float)
        return Overlap, KS
    
# Reads the OVERLAP and KOHN-SHAM MATRICES (spin ALPHA and BETA)
    def readCP2K2fileSpin(self,CP2K_ao_mat):
        overlap_elements  = []
        ks_elements_alpha = []
        ks_elements_beta  = []
        useless           = []
        useless_01        = []
        lines2ignore = [1,2,3,4]
        where_to_dump = overlap_elements
        with open (CP2K_ao_mat, "r") as f1:
            for line in f1:
                if "OVERLAP MATRIX" in line:
                    continue
                elif "DENSITY MATRIX FOR ALPHA SPIN" in line:
                    where_to_dump = useless
                    continue
                elif "DENSITY MATRIX FOR BETA SPIN" in line:
                    where_to_dump = useless_01
                    continue
                elif "KOHN-SHAM MATRIX FOR ALPHA SPIN" in line:
                    where_to_dump = ks_elements_alpha
                    continue
                elif "KOHN-SHAM MATRIX FOR BETA SPIN" in line:
                    where_to_dump = ks_elements_beta
                    continue
                Mat = line.split() 
                if len(Mat) == 0:
                    continue
                elif len(Mat) in lines2ignore:
                    where_to_dump.append([])
                    continue
                where_to_dump[-1].append(Mat[4:])
                
            Overlap  = np.hstack(overlap_elements).astype(float)
            KS_alpha = (phys.physical_constants["Hartree energy in eV"][0])\
                        *(np.hstack(ks_elements_alpha)).astype(float)
            KS_beta  = (phys.physical_constants["Hartree energy in eV"][0])\
                        *(np.hstack(ks_elements_beta)).astype(float)
        f1.close()
        return Overlap, KS_alpha, KS_beta

    def readCP2K_S_Spin(self,CP2K_ao_mat):
        overlap_elements  = []
        ks_elements_alpha = []
        ks_elements_beta  = []
        useless           = []
        useless_01        = []
        lines2ignore      = [1,2,3,4]

        where_to_dump = overlap_elements
        with open (CP2K_ao_mat, "r") as f2:
            for line in f2:
                if "OVERLAP MATRIX" in line:
                    continue
                elif "DENSITY MATRIX FOR ALPHA SPIN" in line:
                    where_to_dump = useless
                    continue
                elif "DENSITY MATRIX FOR BETA SPIN" in line:
                    where_to_dump = useless_01
                    continue
                elif "KOHN-SHAM MATRIX FOR ALPHA SPIN" in line:
                    where_to_dump = ks_elements_alpha
                    continue
                elif "KOHN-SHAM MATRIX FOR BETA SPIN" in line:
                    where_to_dump = ks_elements_beta
                    continue
                Mat = line.split() 
                if len(Mat) == 0:
                    continue
                elif len(Mat) in lines2ignore:
                    where_to_dump.append([])
                    continue
                where_to_dump[-1].append(Mat[4:])
            Overlap  = np.hstack(overlap_elements).astype(float)
        f2.close()
        return Overlap

    # This method identifies the beggining and end of the different
    # matrices to read from *.ao CP2K file
    def readCP2K_lines_Spin(self, CP2K_ao_mat):
        eof = 0
        with open (CP2K_ao_mat) as f1:
            for iline, line in enumerate(f1):
                eof += 1
                if "OVERLAP" in line:
                    line2start_overlap = iline
                elif "DENSITY MATRIX FOR ALPHA SPIN" in line:
                    line2start_p_alpha = iline
                elif "DENSITY MATRIX FOR BETA SPIN" in line:
                    line2start_p_beta = iline
                elif "KOHN-SHAM MATRIX FOR ALPHA SPIN" in line:
                    line2start_ks_alpha = iline
                elif "KOHN-SHAM MATRIX FOR BETA SPIN" in line:
                    line2start_ks_beta = iline
        f1.close()
        return eof, line2start_overlap, line2start_p_alpha, line2start_p_beta,\
               line2start_ks_alpha, line2start_ks_beta

    # This methods reads and returns the Overlap matrix from *.ao CP2K file
    def readCP2K_S_Spin(self, CP2K_ao_mat, eof, line2start_overlap,\
                        line2start_p_alpha, line2start_p_beta,\
                        line2start_ks_alpha, line2start_ks_beta):
        s_elements   = []
        lines2ignore = [1,2,3,4]
        l_s   = 0
        with open (CP2K_ao_mat) as f2:
            for iline, line in enumerate(itertools.islice(f2,\
                                         line2start_overlap+1,\
                                         line2start_p_alpha)):
                l_s += 1
                Mat = line.split() 
                if len(Mat) == 0:
                    continue
                elif len(Mat) in lines2ignore:
                    s_elements.append([])
                    continue
                s_elements[-1].append(Mat[4:])
        f2.close()
        Overlap  = np.hstack(s_elements).astype(float)
        print ("Number of lines S:", l_s, flush=True)
        print ("Size of Overlap Matrix:", df(Overlap).shape,\
               flush=True)
        return Overlap


    # This methods reads and returns the KOHN-SHAM matrix per spin 
    # from *.ao CP2K file
    def readCP2K_KS_Spin(self, CP2K_ao_mat, spin, eof, line2start_overlap,\
                         line2start_p_alpha, line2start_p_beta,\
                         line2start_ks_alpha, line2start_ks_beta):
        ks_elements_alpha = []
        ks_elements_beta  = []
        lines2ignore      = [1,2,3,4]
        if spin == "alpha":
            l_ks_alpha = 0
            with open (CP2K_ao_mat) as f2:
                for iline, line in enumerate(itertools.islice(f2,\
                                             line2start_ks_alpha+1,\
                                             line2start_ks_beta)):
                    l_ks_alpha += 1
                    Mat = line.split() 
                    if len(Mat) == 0:
                        continue
                    elif len(Mat) in lines2ignore:
                        ks_elements_alpha.append([])
                        continue
                    ks_elements_alpha[-1].append(Mat[4:])
            f2.close()
            print ("Number of lines KS", spin, ": ", l_ks_alpha, flush=True)
            KS_spin = (phys.physical_constants["Hartree energy in eV"][0])\
                        *(np.hstack(ks_elements_alpha)).astype(float)
            print ("Size of Kohn-Sham Matrix", spin, ":", df(KS_spin).shape,\
                   flush=True)

        elif spin == "beta":
            l_ks_beta = 0
            with open(CP2K_ao_mat) as f2: 
                for iline,line in enumerate(itertools.islice(f2,\
                                            line2start_ks_beta+1, eof)):
                    l_ks_beta += 1
                    Mat = line.split()
                    if len(Mat) == 0:
                        continue
                    elif len(Mat) in lines2ignore:
                        ks_elements_beta.append([])
                        continue
                    ks_elements_beta[-1].append(Mat[4:])
            f2.close()
            KS_spin  = (phys.physical_constants["Hartree energy in eV"][0])\
                        *(np.hstack(ks_elements_beta)).astype(float)
            print ("Number of lines KS", spin, ":", l_ks_beta, flush=True)
            print ("Size of Kohn-Sham Matrix", spin, ":", df(KS_spin).shape,\
                   flush=True)
        return KS_spin

    # This method reads out the density matrix per spin from *.ao CP2K file
    def readCP2K_P_Spin(self, CP2K_ao_mat, spin, eof, line2start_overlap,\
                         line2start_p_alpha, line2start_p_beta,\
                         line2start_ks_alpha, line2start_ks_beta):
        den_elements_alpha = []
        den_elements_beta  = []
        lines2ignore      = [1,2,3,4]
        if spin == "alpha":
            l_p_alpha = 0
            with open (CP2K_ao_mat) as f2:
                for iline, line in enumerate(itertools.islice(f2,\
                                             line2start_p_alpha+1,\
                                             line2start_p_beta)):
                    l_p_alpha += 1
                    Mat = line.split() 
                    if len(Mat) == 0:
                        continue
                    elif len(Mat) in lines2ignore:
                        den_elements_alpha.append([])
                        continue
                    den_elements_alpha[-1].append(Mat[4:])
            f2.close()
            P_spin = np.hstack(den_elements_alpha).astype(float)
            print ("Number of lines Density Matrix", spin, ":",\
                   l_p_alpha, flush=True)
            print ("Size of Density Matrix", spin, ":", df(P_spin).shape,\
                   flush=True)

        elif spin == "beta":
            l_p_beta = 0
            with open(CP2K_ao_mat) as f2: 
                for iline,line in enumerate(itertools.islice(f2,\
                                            line2start_p_beta+1,\
                                            line2start_ks_alpha)):
                    l_p_beta += 1
                    Mat = line.split()
                    if len(Mat) == 0:
                        continue
                    elif len(Mat) in lines2ignore:
                        den_elements_beta.append([])
                        continue
                    den_elements_beta[-1].append(Mat[4:])
            f2.close()
            P_spin = np.hstack(den_elements_beta).astype(float)
            print ("Number of lines Density Matrix", spin, ":",\
                   l_p_beta, flush=True)
            print ("Size of Density Matrix", spin, ":", df(P_spin).shape,\
                   flush=True)
        return P_spin

    # Reads the DENSITY MATRIX (spin ALPHA and BETA)
    def readCP2K2fileSpin_density(self,CP2K_ao_mat):
        with open (CP2K_ao_mat, "r") as f1:
            for line in f1:
                if "DENSITY MATRIX FOR ALPHA SPIN" in line:
                    where_to_dump = den_elements_alpha
                    continue
                elif "DENSITY MATRIX FOR BETA SPIN" in line:
                    where_to_dump = den_elements_beta
                    continue
                elif "KOHN-SHAM MATRIX FOR ALPHA SPIN" in line:
                    break
                elif "KOHN-SHAM MATRIX FOR BETA SPIN" in line:
                    break
                Mat = line.split()
                if len(Mat) == 0:
                    continue
                elif len(Mat) in lines2ignore:
                    where_to_dump.append([])
                    continue
                where_to_dump[-1].append(Mat[4:])
                
            Density_alpha = np.hstack(den_elements_alpha).astype(float)
            Density_beta  = np.hstack(den_elements_beta).astype(float)
        f1.close()
        return Density_alpha, Density_beta

    # Get the Fermi Energy of the system. It uses as Input file the *.out or 
    # *.pdos file from CP2K 
    def getEf(self,CP2Koutfile):
        Ef = []
        try:
            with open(CP2Koutfile) as f1:
                for line in f1.readlines():
                    Efs  = re.search(r".*?fermi\s+energy\s?.*?(-?\d+\.\d+).*?",\
                                     line, flags=re.IGNORECASE)
                    Efs1 = re.search(r"(step i = 0), (\D*) (.*\d)", line)
                    if Efs:
                        Ef.append(float(Efs.group(1)))
                    if Efs1:
                        Ef.append(float(Efs1.group(3)))
                        Ef = \
                           [(phys.physical_constants["Hartree energy in eV"][0])\
                            * i for i in Ef]
                if self.Spin in ["No","no","N","n"]:
                    Ef = Ef
                elif self.Spin in ["Yes","yes","Y","y"]:
                    Ef = Ef
            return Ef
        except FileNotFoundError:
            print ("Wrong file or path to *.out file!")
            sys.exit()


    # This method substracts two things:
    # 1) atom type and number of orbitals in those atoms
    # 2) geometry of the system
    def get_region_geometry(self,CP2Koutfile):
        atom_kind       = []
        number_orbitals = []
        geometry_atom   = []
        atom_number     = []
        try:
            with open(CP2Koutfile) as f1:
                if self.pDOS in ["No","no","N","n"]:
                    for line in f1.readlines():
                        at_info = re.search("(\d*) (Atomic kind): (.*\D) (Number of atoms): (.*\d)", line)
                        num_orb = re.search(" (Number of spherical basis functions): (.*\d)",line)
                        geo_at  = re.search("(.*\d) (.*\d) ([A-Z]|[A-Z][a-z]) (.*\d) ",line)
                        if at_info:
                            atom_kind.append(str(at_info.group(3)))
                            atom_kind =\
                                    [element.strip() for element in atom_kind]
                        elif num_orb:
                            number_orbitals.append(int(num_orb.group(2)))
                        elif geo_at:
                            atom_number.append(int(geo_at.group(1)))
                            geometry_atom.append(geo_at.group(3))
                        dic_elements = dict(zip(atom_kind,number_orbitals))
                        geometry     = list(zip(atom_number,geometry_atom))
                elif self.pDOS in ["Yes","yes","Y","y"]:
                    for line in f1.readlines():
                        at_info = re.search("(\d*) (Atomic kind): (.*\D) (Number of atoms): (.*\d)", line)
                        num_orb = re.search(" (Number of spherical basis functions): (.*\d)",line)
                        geo_at  = re.search("(.*\d) (.*\d) ([A-Z]|[A-Z][a-z]) (.*\d) ",line)
                        if at_info:
                            atom_kind.append(str(at_info.group(3)))
                            atom_kind =\
                                    [element.strip() for element in atom_kind]
                        elif num_orb:
                            number_orbitals.append(int(num_orb.group(2)))
                        elif geo_at:
                            atom_number.append(int(geo_at.group(1)))
                            geometry_atom.append(geo_at.group(3))
                        dic_elements = dict(zip(atom_kind,number_orbitals))
                        geometry     = list(zip(atom_number,geometry_atom))
                return dic_elements, geometry

        except FileNotFoundError:
            print ("Wrong file or path to *.out file")
            sys.exit()
