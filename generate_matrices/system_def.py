#!/usr/bin/python3

'''
This will define the system into consideration
Works for TB chains
'''

from numpy import *
import argparse
import yaml
import os
import pandas as pd


class System_def:
    def __init__(self, cfg):
        self.t_c      =  cfg["nn hopping central system"]
        self.t_l      =  cfg["nn hopping leads"]
        self.dim_c    =  cfg["Dimension of the central system"]
        self.dim_l    =  cfg["Dimension of the leads"]
        self.eps_c    =  cfg["Onsite energy of central system"]
        self.eps_l    =  cfg["Onsite energy of leads"]
        self.path_inp =  cfg["Path to the system matrices"]
    
    def Inppath(self):
        return self.path_inp  

def mtrx_inp(name,dim_1,dim_2,mat):
    while True:
        try:
            Choise = input ("Is that correct? y/n:  ")
            if Choise == "y":
                break
            if Choise == "n":
                print ("Define the elements of your ",name, "\n")
                for i in arange(dim_1):
                    for j in arange(dim_2):
                        while True:
                            try:
                                print ("[",i,"]","[",j,"]")
                                mat[i][j] = input()
                                mat[i][j] = float(mat[i][j])
                                print ("\n")
                                break
                            except ValueError:
                                print("\n\n","Invalid number, try again","\n\n")
                break
        except ValueError: 
            print("\n\n", "Invalid option, try again","\n\n")



def build_and_dump(cfg):
    dim_central =  System_def(cfg).dim_c
    dim_lead    =  System_def(cfg).dim_l
    HC   = System_def(cfg).eps_c * eye(System_def(cfg).dim_c, k=0, dtype=float) + \
           System_def(cfg).t_c * eye(System_def(cfg).dim_c, k=1,dtype=float) + \
           System_def(cfg).t_c * eye(System_def(cfg).dim_c, k=-1,dtype=float)
    print ('HC: ', '\n\n', pd.DataFrame(data=HC, dtype=float), '\n')
    mtrx_inp("Central-Hamiltonian", dim_central, dim_central, HC)  


    HL   = System_def(cfg).eps_l * eye(System_def(cfg).dim_l, k=0, dtype=float) + \
           System_def(cfg).t_l * eye(System_def(cfg).dim_l, k=1,dtype=float) + \
           System_def(cfg).t_l * eye(System_def(cfg).dim_l, k=-1,dtype=float)
    print ("\n\n"'HL: ', '\n\n', pd.DataFrame(data=HL, dtype=float), '\n')
    mtrx_inp("Left-Hamiltonian",dim_lead, dim_lead, HL)  


    HR   = System_def(cfg).eps_l * eye(System_def(cfg).dim_l, k=0, dtype=float) + \
           System_def(cfg).t_l * eye(System_def(cfg).dim_l, k=1,dtype=float) + \
           System_def(cfg).t_l * eye(System_def(cfg).dim_l, k=-1,dtype=float)
    print ("\n\n"'HR: ', '\n\n', pd.DataFrame(data=HR, dtype=float), '\n')
    mtrx_inp("Right-Hamiltonian",dim_lead, dim_lead, HR)  


    SC   = eye(System_def(cfg).dim_c, dtype=float)  
    SL   = eye(System_def(cfg).dim_l, dtype=float)
    SR   = eye(System_def(cfg).dim_l, dtype=float)


    TL   = zeros((System_def(cfg).dim_l,System_def(cfg).dim_l), dtype=float)
    print ("\n\n"'TL: ', '\n\n', pd.DataFrame(data=TL, dtype=float), '\n')
    mtrx_inp("Left-Coupling",dim_lead, dim_lead, TL)  


    TR   =zeros((System_def(cfg).dim_l,System_def(cfg).dim_l), dtype=float)
    print ("\n\n"'TR: ', '\n\n', pd.DataFrame(data=TR, dtype=float), '\n')
    mtrx_inp("Right-Coupling",dim_lead, dim_lead, TR)  


    STL  = zeros((System_def(cfg).dim_l,System_def(cfg).dim_l), dtype=float)   
    STR  = zeros((System_def(cfg).dim_l,System_def(cfg).dim_l), dtype=float)  


    VCL = zeros((System_def(cfg).dim_c, System_def(cfg).dim_l), dtype=float)
    print ("\n\n"'VCL: ', '\n\n', pd.DataFrame(data=VCL, dtype=float), '\n')
    mtrx_inp("Central-Left-lead-Coupling",dim_central, dim_lead, VCL)  


    VCR =zeros((System_def(cfg).dim_c, System_def(cfg).dim_l), dtype=float)
    print ("\n\n"'VCR: ', '\n\n', pd.DataFrame(data=VCR, dtype=float), '\n')
    mtrx_inp("Central-Right-lead-Coupling",dim_central, dim_lead, VCR)  


    SCL  = zeros((System_def(cfg).dim_c,System_def(cfg).dim_l), dtype=float) 
    SCR  = zeros((System_def(cfg).dim_c,System_def(cfg).dim_l), dtype=float)


    path  =  System_def(cfg).Inppath()
    if not os.path.exists(path):
        os.makedirs(path)
    
    HC.dump(path+"/HC.dat")
    HL.dump(path+"/HL.dat")
    HR.dump(path+"/HR.dat")
    SC.dump(path+"/SC.dat")
    SL.dump(path+"/SL.dat")
    SR.dump(path+"/SR.dat")
    TL.dump(path+"/TL.dat")
    TR.dump(path+"/TR.dat")
    STL.dump(path+"/STL.dat")
    STR.dump(path+"/STR.dat")
    VCL.dump(path+"/VCL.dat")
    VCR.dump(path+"/VCR.dat")
    SCL.dump(path+"/SCL.dat")
    SCR.dump(path+"/SCR.dat")
