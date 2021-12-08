import numpy as np
import pandas as pd
import time
import os
import sys
import generate_matrices.class_systemdef as sdef

#Diese Linie wird 79 Zeichen enthalten ----------------------------------------


''' Parameters on this methods:
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

def build_and_dump(cfg):
    
    start = time.time() 
 
    Sname            = sdef.SystemDef(cfg).Sname
    Spin             = sdef.SystemDef(cfg).Spin
    iatomC           = sdef.SystemDef(cfg).iatomC 
    fatomC           = sdef.SystemDef(cfg).fatomC 
    iatomL           = sdef.SystemDef(cfg).iatomLleads 
    fatomL           = sdef.SystemDef(cfg).fatomLleads 
    iatomR           = sdef.SystemDef(cfg).iatomRleads 
    fatomR           = sdef.SystemDef(cfg).fatomRleads 
    Nperiods         = sdef.SystemDef(cfg).Nperiods
    CP2K_ao_mat      = sdef.SystemDef(cfg).CP2K_ao_mat    
    CP2K_out         = sdef.SystemDef(cfg).CP2Koutfile
    CP2K_pdos_alpha  = sdef.SystemDef(cfg).CP2Kpdosfile_alpha
    CP2K_pdos_beta   = sdef.SystemDef(cfg).CP2Kpdosfile_beta
    p2f              = sdef.SystemDef(cfg).path2files
    path_inp         = sdef.SystemDef(cfg).path_inp
    CB_regime        = sdef.SystemDef(cfg).CB_regime
    pDOS             = sdef.SystemDef(cfg).pDOS
    
    path  = sdef.SystemDef(cfg).path_inp
    if not os.path.exists(path):
        os.makedirs(path)

    matrices = sdef.SystemDef(cfg)
    
    Ef       = matrices.getEf(p2f+CP2K_out)
    if Ef == []:
        for pdosfiles in [CP2K_pdos_alpha,CP2K_pdos_beta]:
            Ef += matrices.getEf(p2f+pdosfiles) 

    dic_elem, geo  = matrices.get_region_geometry(p2f+CP2K_out)
###################################################################
# For pDOS calculations
#    if pDOS in ["Yes","yes","Y","y"]:
#        at2ig  = np.array(range(iatom2ign,fatom2ign+1),dtype=int)
#        fatomC = fatomC - len(at2ig)
#        iatomR = iatomR - len(at2ig)
#        fatomR = fatomR - len(at2ig)
#    elif pDOS in ["No","no","N","n"]:
#        pass
###################################################################

    Nuc   = [(i,j) for i,j in geo if i >= iatomC and i <= fatomC]
    NuclL = [(i,j) for i,j in geo if i >= iatomL and i <= fatomL]
    NuclR = [(i,j) for i,j in geo if i >= iatomR and i <= fatomR]

    # elemC:  is the number of elements in of the CP2K numbering of the central 
    #         system
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
    elemppL  = int(elemlL/Nperiods)
    elemppR  = int(elemlR/Nperiods)

    ''' Lines to use for getting specific orbitals from the KS and S matrices. 
    Uncomment/modify if necessary
    '''
    #elemC    = int(2*100)   
    #elemlL   = int(2*100)   
    #elemlR   = int(2*100)   
    #elemppL  = int(2*100/Nperiods)
    #elemppR  = int(2*100/Nperiods)


    print ("System name : ", Sname)
    print ("Central region : ",iatomC, ",", fatomC)
    print ("Left region    : ",iatomL, ",", fatomL)
    print ("Right region   : ",iatomR, ",", fatomR)
    print ("Fermi energie(s) : " , Ef)
    print ("Orbitals per atom type : " , dic_elem)
    print ("Number of elements in Central  region = ",elemC)
    print ("Number of elements in    Left  region = ",elemlL)
    print ("Number of elements in   Right  region = ",elemlR)
    print ("Periods on the leads : ",Nperiods)
    print ("Elements per period on Left lead : ",elemppL)
    print ("Elements per period on Right lead : ",elemppR)
    print ("Coulomb blockade : ", CB_regime)

    if Spin in ["No","no","N","n"]:
        S,KS  = matrices.readCP2K2file(p2f+CP2K_ao_mat)
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
        S,KS_alpha,KS_beta = matrices.readCP2K2fileSpin(p2f+CP2K_ao_mat)
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
                       2*elemppL:elemlL-elemppL], fmt = '%.5f')
            np.savetxt(path+"/STR.dat", S[elemlL+elemC:elemlL+elemC+elemppR,\
               elemlL+elemC+elemppR:elemlL+elemC+2*(elemppR)], fmt = '%.5f')

#Diese Linie wird 79 Zeichen enthalten ----------------------------------------
        
        for i,matrix in zip(["alpha","beta"],[KS_alpha,KS_beta]):
            np.savetxt(path+"/HC-"+i+".dat", matrix[elemlL:elemlL+elemC,\
                       elemlL:elemlL+elemC], fmt = '%.5f')
            np.savetxt(path+"/HL-"+i+".dat", matrix[elemlL-elemppL:elemlL,\
                       elemlL-elemppL:elemlL], fmt = '%.5f')
            np.savetxt(path+"/HR-"+i+".dat", matrix[elemlL+elemC:elemlL+elemC\
                       +elemppR, elemlL+elemC:elemlL+elemC+elemppR], fmt = '%.5f')
            np.savetxt(path+"/VCL-"+i+".dat", matrix[elemlL:elemlL+elemC,\
                       elemlL-elemppL:elemlL], fmt = '%.5f')
            np.savetxt(path+"/VCR-"+i+".dat", matrix[elemlL:elemlL+elemC,\
                       elemlL+elemC:elemlL+elemC+elemppR], fmt = '%.5f')
            if Nperiods == 1: 
                np.savetxt(path+"/TL-"+i+".dat", matrix[elemlL:elemlR+elemC,\
                           0:elemlL], fmt = '%.5f')
                np.savetxt(path+"/TR-"+i+".dat", matrix[elemlR:elemlR+elemC,\
                           elemlR+elemC:2*elemlR+elemC], fmt = '%.5f')
            elif Nperiods > 1:
                np.savetxt(path+"/TL-"+i+".dat", matrix[elemlL-elemppL:elemlL,\
                           elemlL-2*elemppL:elemlL-elemppL], fmt = '%.5f')
                np.savetxt(path+"/TR-"+i+".dat", matrix[elemlL+elemC:elemlL+\
                           elemC+elemppR, elemlL+elemC+elemppR:elemlL+elemC+\
                           2*elemppR], fmt = '%.5f')
            np.savetxt(path+"/KS-"+i+".dat", matrix, fmt = '%.5f')

        if CB_regime in ["Yes","yes","Y","y"]:
            Density_alpha, Density_beta = matrices.readCP2K2fileSpin_density(p2f+CP2K_ao_mat)
            for i,matrix in zip(["alpha","beta"],[Density_alpha,Density_beta]):
                np.savetxt(path+"/PC-"+i+".dat", matrix[elemlL:elemlL+elemC,\
                           elemlL:elemlL+elemC], fmt = '%.5f')
                np.savetxt(path+"/DENSITY-"+i+".dat", matrix, fmt = '%.5f')
        else:
            pass

    else:
        print ("Unknown Spin option")
        sys.exit()
    
    np.savetxt(path+"/OVERLAP.dat", S, fmt = '%.5f')
    np.savetxt(path+"/Ef.dat", Ef)
    stop = time.time()
    temp = stop-start
    hours = temp//3600
    minutes = temp//60
    seconds = temp - 60*minutes
    info00 = ("Entire time for Generating 14 Matrices: {:.1f}:{:.1f}:{:.1f}"
              "h/m/s")
    print (info00.format(hours,minutes,seconds))

