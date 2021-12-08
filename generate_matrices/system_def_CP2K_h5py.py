import generate_matrices.class_systemdef as sdef
import modules.mo_print as moprint
import pandas as pd
import numpy as np
import time
import h5py
import sys
import os
from pandas import DataFrame as df

''' Parameters on this functions:
- Spin:        if the calculation is Spin polarized
- iatomX:      first atom of region X
- fatomX:      last atom of region X
- Nperiods:    periods of Unit cell of the leads
- CP2K_ao_mat: the *.ao file name from CP2K
- CP2K_out:    *.out file from CP2K
- p2f:         path to the OVERLAP and K-S matrices and *.out files
- path_inp:    path to dump the 14 matrices extracted from the *.ao file
- Xatom2ign:   If a projected DOS is to be calculated, indicates which atoms 
               have to be ignore to have the isolated molecule for the pDOS
'''
#@profile
def build_and_dump(cfg):
    matrices = sdef.SystemDef(cfg)
    
    start = time.time() 
 
    Sname           = sdef.SystemDef(cfg).Sname
    Spin            = sdef.SystemDef(cfg).Spin
    iatomC          = sdef.SystemDef(cfg).iatomC 
    fatomC          = sdef.SystemDef(cfg).fatomC 
    iatomL          = sdef.SystemDef(cfg).iatomLleads 
    fatomL          = sdef.SystemDef(cfg).fatomLleads 
    iatomR          = sdef.SystemDef(cfg).iatomRleads 
    fatomR          = sdef.SystemDef(cfg).fatomRleads 
    Nperiods        = sdef.SystemDef(cfg).Nperiods
    CP2K_ao_mat     = sdef.SystemDef(cfg).CP2K_ao_mat    
    CP2K_out        = sdef.SystemDef(cfg).CP2Koutfile
    CP2K_pdos_alpha = sdef.SystemDef(cfg).CP2Kpdosfile_alpha
    CP2K_pdos_beta  = sdef.SystemDef(cfg).CP2Kpdosfile_beta
    p2f             = sdef.SystemDef(cfg).path2files
    path_inp        = sdef.SystemDef(cfg).path_inp
    CB_regime       = sdef.SystemDef(cfg).CB_regime
    pDOS            = sdef.SystemDef(cfg).pDOS
    
    path = sdef.SystemDef(cfg).path_inp
    if not os.path.exists(path):
        os.makedirs(path)
    
    Ef = matrices.getEf(p2f+CP2K_out)
    if Ef == []:
        for pdosfiles in [CP2K_pdos_alpha,CP2K_pdos_beta]:
            Ef += matrices.getEf(p2f+pdosfiles) 

    dic_elem, geo = matrices.get_region_geometry(p2f + CP2K_out)

    ###################################################################
    # For pDOS calculations
    #if pDOS in ["Yes","yes","Y","y"]:
    #    at2ig  = np.array(range(iatom2ign, fatom2ign+1), dtype=int)
    #    fatomC = fatomC - len(at2ig)
    #    iatomR = iatomR - len(at2ig)
    #    fatomR = fatomR - len(at2ig)
    #elif pDOS in ["No","no","N","n"]:
    #    pass
    ###################################################################

    Nuc   = [(i,j) for i,j in geo if i >= iatomC and i <= fatomC]
    NuclL = [(i,j) for i,j in geo if i >= iatomL and i <= fatomL]
    NuclR = [(i,j) for i,j in geo if i >= iatomR and i <= fatomR]

    # elemC:  is the number of elements in of the CP2K numbering of the 
    # central system
    # elemlX: is the number of elements in of the CP2K numbering of lead X
    elemC  = 0
    elemlL = 0
    elemlR = 0
    for k,l in Nuc:
        if l in dic_elem.keys():
            elemC += dic_elem[l]
    for k,l in NuclL:
        if l in dic_elem.keys():
            elemlL += dic_elem[l]
    for k,l in NuclR:
        if l in dic_elem.keys():
            elemlR += dic_elem[l]

    # elemppX: stands for elements per period of lead X
    elemppL = int(elemlL/Nperiods)
    elemppR = int(elemlR/Nperiods)

    ''' Lines to use for getting specific orbitals from the KS and S matrices. 
        Uncomment/modify if necessary
    '''
    #elemC    = int(2*100)   
    #elemlL   = int(2*100)   
    #elemlR   = int(2*100)   
    #elemppL  = int(2*100/Nperiods)
    #elemppR  = int(2*100/Nperiods)

    info00 = ("Building the system")
    print (moprint.ibordered_title(info00), flush=True, end='\n\n')

    info01 = ("System name            : {}\n"
              "Central region         : {} , {}\n"
              "Left region            : {} , {}\n"
              "Right region           : {} , {}\n"
              "Fermi energie(s)       : {}\n"
              "Orbitals per atom type : {}\n"
              "Periods on the leads   : {}\n"
              "Number of elements in Central region = {}\n"
              "Number of elements in    Left region = {}\n"
              "Number of elements in   Right region = {}\n"
              "Elements per period on Left lead  : {}\n"
              "Elements per period on Right lead : {}\n"
              "Coulomb blockade : {}")
    print (moprint.ibordered(info01.format(Sname, iatomC, fatomC, iatomL, \
                                           fatomL, iatomR, fatomR, Ef, \
                                           dic_elem, Nperiods, elemC, elemlL, \
                                           elemlR, elemppL, elemppR, \
                                           CB_regime)), flush=True, end='\n')
    
    if Spin in ["No","no","N","n"]:
        S, KS = matrices.readCP2K2file(p2f+CP2K_ao_mat)
        ''' Lines to use for getting specific orbitals from the KS and 
        S matrices. Uncomment/modify if necessary
        '''
        #S,KS  = matrices.readCP2K2file_porbit(p2f+CP2K_ao_mat)
        
        ''' Lines to use for getting specific orbitals from the KS and S matrices. 
        Uncomment/modify if necessary. Just get 3pz orbitals
        '''
        #S   = np.asarray([S[:, [3+12*i+i]] for i in range(300)])
        #KS  = np.asarray([KS[:, [3+12*i+i]] for i in range(300)])
        #S   = S[:,:,0]
        #KS  = KS[:,:,0]
       
        #Get the 3pz and 4pz orbitals
        #indexes1 = []
        #indexes2 = []
        #for i in range(300):
        #    x = 10*i+(10+3*(i-2))-1
        #    indexes1.append(x)  
        #for j in range(1,301):
        #    y = 10*j+3*(j-2)-1
        #    indexes2.append(y)
        #full_indexes = np.sort(np.concatenate((indexes1,indexes2), axis=None))
        #S  =  np.take(S,full_indexes,axis=1)
        #KS =  np.take(KS,full_indexes,axis=1)
        
        print (pd.DataFrame(S))

        np.savetxt(path+"/SC.dat", S[elemlL:elemlL+elemC, elemlL:elemlL+elemC],\
                   fmt = '%.5f')
        np.savetxt(path+"/SL.dat", S[elemlL-elemppL:elemlL, elemlL-elemppL:\
                   elemlL], fmt = '%.5f')
        np.savetxt(path+"/SR.dat", S[elemlL+elemC:elemlL+elemC+elemppR,\
                   elemlL+elemC:elemlL+elemC+elemppR], fmt = '%.5f')
        
        np.savetxt(path+"/SCL.dat", S[elemlL:elemlL+elemC, elemlL-elemppL:\
                   elemlL], fmt = '%.5f')
        np.savetxt(path+"/SCR.dat", S[elemlL:elemlL+elemC, elemlL+elemC:\
                   elemlL+elemC+elemppR], fmt = '%.5f')
        
        if Nperiods == 1: 
            np.savetxt(path+"/STL.dat",S[elemlL:elemlL+elemC, elemlL-elemppL:\
                       elemlL], fmt = '%.5f')
            np.savetxt(path+"/STR.dat",S[elemlL:elemlL+elemC, elemlL+elemC:\
                       elemlL+elemC+elemppR], fmt = '%.5f')
        elif Nperiods > 1:
            np.savetxt(path+"/STL.dat",S[elemlL-elemppL:elemlL, elemlL-\
                       (2*elemppL):elemlL-elemppL], fmt = '%.5f')
            np.savetxt(path+"/STR.dat", S[elemlL+elemC:elemlL+elemC+elemppR,\
                       elemlL+elemC+elemppR:elemlL+elemC+2*(elemppR)],\
                       fmt = '%.5f')
            
        np.savetxt(path+"/HC.dat", KS[elemlL:elemlL+elemC, elemlL:elemlL+elemC],\
                   fmt = '%.5f')
        np.savetxt(path+"/HL.dat", KS[elemlL-elemppL:elemlL, elemlL-elemppL:\
                   elemlL], fmt = '%.5f')
        np.savetxt(path+"/HR.dat", KS[elemlL+elemC:elemlL+elemC+elemppR,\
                   elemlL+elemC:elemlL+elemC+elemppR], fmt = '%.5f')
        np.savetxt(path+"/VCL.dat", KS[elemlL:elemlL+elemC, elemlL-elemppL:\
                   elemlL], fmt = '%.5f')
        np.savetxt(path+"/VCR.dat", KS[elemlL:elemlL+elemC, elemlL+elemC:\
                   elemlL+elemC+elemppR], fmt = '%.5f')
        if Nperiods == 1: 
            np.savetxt(path+"/TL.dat", KS[elemlL:elemlL+elemC, elemlL-elemppL:\
                       elemlL], fmt = '%.5f')
            np.savetxt(path+"/TR.dat", KS[elemlL:elemlL+elemC, elemlL+elemC:\
                       elemlL+elemC+elemppR], fmt = '%.5f')
        elif Nperiods > 1:
            np.savetxt(path+"/TL.dat", KS[elemlL-elemppL:elemlL, elemlL-\
                       (2*elemppL):elemlL-elemppL], fmt = '%.5f')
            np.savetxt(path+"/TR.dat", KS[elemlL+elemC:elemlL+elemC+elemppR,\
                       elemlL+elemC+elemppR:elemlL+elemC+2*(elemppR)],\
                       fmt = '%.5f')
        
        np.savetxt(path+"/KS.dat", KS, fmt = '%.5f')

    elif Spin in ["Yes","yes","Y","y"]:  
        # Identify beggining and end of the KS, OVERLAP and DENSITY matrices on
        # a *.ao CP2K file
        eof, line2start_overlap, line2start_p_alpha, line2start_p_beta,\
        line2start_ks_alpha, line2start_ks_beta =\
        matrices.readCP2K_lines_Spin(p2f + CP2K_ao_mat)

        with h5py.File(path+"/S_all.h5", "w") as S_all_h5:
            matrix = []
            matrix = matrices.readCP2K_S_Spin(p2f+CP2K_ao_mat, eof,\
                             line2start_overlap, line2start_p_alpha,\
                             line2start_p_beta, line2start_ks_alpha,\
                             line2start_ks_beta)
            S_all_h5.create_dataset("SC",\
                                    data=matrix[elemlL:elemlL+elemC,\
                                    elemlL:elemlL+elemC])
            S_all_h5.create_dataset("SL",\
                                    data=matrix[elemlL-elemppL:elemlL,\
                                    elemlL-elemppL:elemlL])
            S_all_h5.create_dataset("SR",\
                                    data=matrix[elemlL+elemC:\
                                    elemlL+elemC+elemppR, elemlL+elemC:\
                                    elemlL+elemC+elemppR])
            S_all_h5.create_dataset("SCL",\
                                    data=matrix[elemlL:elemlL+elemC,\
                                    elemlL-elemppL:elemlL])
            S_all_h5.create_dataset("SCR",\
                                    data=matrix[elemlL:elemlL+elemC,\
                                    elemlL+elemC:elemlL+elemC+elemppR])
            
            if Nperiods == 1: 
                S_all_h5.create_dataset("STL",
                                        data=matrix[elemlL:elemlL+elemC,
                                                    elemlL-elemppL:elemlL])
                S_all_h5.create_dataset("STR",
                                        data=matrix[elemlL:elemlL+elemC,
                                                    elemlL+elemC:
                                                    elemlL+elemC+elemppR])
            elif Nperiods > 1:
                S_all_h5.create_dataset("STL",\
                                        data=matrix[elemlL-elemppL:elemlL,\
                                        elemlL-2*elemppL:elemlL-elemppL])
                S_all_h5.create_dataset("STR",\
                                        data=matrix[elemlL+elemC:\
                                                    elemlL+elemC+elemppR,\
                                                    elemlL+elemC+elemppR:\
                                                    elemlL+elemC+2*(elemppR)])
            print ("Fertig mit Overlap", flush=True, end='\n')

        with h5py.File(path+"/H_all.h5", "w") as H_all_h5:
            for i in ["alpha", "beta"]:
                matrix = []
                matrix = matrices.readCP2K_KS_Spin(p2f+CP2K_ao_mat, i, eof,\
                           line2start_overlap, line2start_p_alpha,\
                           line2start_p_beta, line2start_ks_alpha,\
                           line2start_ks_beta)
                H_all_h5.create_dataset("HC-"+i,\
                                        data=matrix[elemlL:elemlL+elemC,\
                                        elemlL:elemlL+elemC])
                H_all_h5.create_dataset("HL-"+i,\
                                        data=matrix[elemlL-elemppL:elemlL,\
                                        elemlL-elemppL:elemlL])
                H_all_h5.create_dataset("HR-"+i,\
                                        data=matrix[elemlL+elemC:elemlL+elemC\
                                        +elemppR, elemlL+elemC:\
                                        elemlL+elemC+elemppR])
                H_all_h5.create_dataset("VCL-"+i,\
                                        data=matrix[elemlL:elemlL+elemC,\
                                        elemlL-elemppL:elemlL])
                H_all_h5.create_dataset("VCR-"+i,\
                                        data=matrix[elemlL:elemlL+elemC,\
                                        elemlL+elemC:elemlL+elemC+elemppR])
                if Nperiods == 1: 
                    H_all_h5.create_dataset("TL-"+i,
                                            data=matrix[elemlL:elemlL+elemC,
                                                        elemlL-elemppL:elemlL])
                    H_all_h5.create_dataset("TR-"+i,\
                                            data=matrix[elemlL:elemlL+elemC,
                                                        elemlL+elemC:
                                                        elemlL+elemC+elemppR])
                elif Nperiods > 1:
                    H_all_h5.create_dataset("TL-"+i,\
                                            data=matrix[elemlL-elemppL:elemlL,\
                                            elemlL-2*elemppL:elemlL-elemppL])
                    H_all_h5.create_dataset("TR-"+i,\
                                            data=matrix[elemlL+elemC:elemlL+\
                                            elemC+elemppR, elemlL+elemC+elemppR:\
                                            elemlL+elemC+2*elemppR])
                print ("Fertig mit KS", i, flush=True, end='\n')

        if CB_regime in ["Yes","yes","Y","y"]:
            with h5py.File(path+"/PC.h5", "w") as PC_all_h5:
                for i in ["alpha", "beta"]:
                    matrix = []
                    matrix = matrices.readCP2K_P_Spin(p2f+CP2K_ao_mat, i, eof,\
                               line2start_overlap, line2start_p_alpha,\
                               line2start_p_beta, line2start_ks_alpha,\
                               line2start_ks_beta)
                    PC_all_h5.create_dataset("DENSITY-"+i, data=matrix)
                    PC_all_h5.create_dataset("PC-"+i,\
                                             data=matrix[elemlL:elemlL+elemC,\
                                             elemlL:elemlL+elemC])
                print ("Fertig mit Density", i, flush=True, end='\n')
        else:
            pass

    else:
        print ("Unknown Spin option")
        sys.exit()
    
    np.savetxt(path+"/Ef", Ef)

    stop = time.time()
    temp = stop-start
    hours = temp//3600
    minutes = temp//60
    seconds = temp - 60*minutes
    info02 = ("Entire time for Generating 16 Matrices: {:.1f}:{:.1f}:{:.1f}"
              "h/m/s")
    print (info02.format(hours,minutes,seconds))
