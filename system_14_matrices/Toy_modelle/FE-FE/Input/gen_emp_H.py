from pandas import DataFrame as df
from tabulate import tabulate
import numpy as np
import h5py

eps_a_1 =  0.0
eps_a_2 =  0.0
eps_b_1 =  0.0
eps_b_2 =  0.0
E_a_1   = -0.3
E_a_2   =  0.3
E_b_1   = -0.3
E_b_2   =  0.3

t_a   = 0.2
t_b   = 0.2
tau_a = 0.1
tau_b = 0.1

v_a   = 0.25
v_b   = 0.25
del_a = 0.1
del_b = 0.1

def emp_H():

    ## Zentraler Hamiltonian
    HC_alpha = np.zeros(shape=(16,16))
    HC_beta  = np.zeros(shape=(16,16))

    dimH = HC_alpha.shape[0]
    for i in [0, 4]:
            HC_alpha[i,i]    = E_a_1
            HC_beta[i+1,i+1] = E_b_1
    for i in [8, 12]:
            HC_alpha[i,i]    = eps_a_1
            HC_beta[i+1,i+1] = eps_b_1
    for i in [2, 6]:
            HC_alpha[i,i]    = E_a_2
            HC_beta[i+1,i+1] = E_b_2
    for i in [10, 14]:
            HC_alpha[i,i]    = eps_a_2
            HC_beta[i+1,i+1] = eps_b_2

    # Kette Elemente
    for i,j in zip([4,6],[0,2]):
        HC_alpha[i,j] = tau_a
        HC_alpha[j,i] = tau_a
    for i,j in zip([5,7],[1,3]):
        HC_beta[i,j] = tau_b
        HC_beta[j,i] = tau_b

    # 1te und 2te SMM Alpha spin
    for i,j in zip([8,10,12,14],[0,2,4,6]):
        HC_alpha[i,j] = t_a
        HC_alpha[j,i] = t_a
        #HC_alpha[i,j] = t_a
        #HC_alpha[j,i] = t_a

    ## Links/Rechts Hamiltonian
    HL_alpha = np.zeros(shape=(8,8))
    HL_beta  = np.zeros(shape=(8,8))

    dimHe = HL_alpha.shape[0]
    for i in [0,4]:
        HL_alpha[i,i]    = E_a_1
        HL_beta[i+1,i+1] = E_b_1
    for i in [2,6]:
        HL_alpha[i,i]    = E_a_2
        HL_beta[i+1,i+1] = E_b_2

    # Hopping Elemente
    for i,j in zip([4,6],[0,2]):
        HL_alpha[i,j] = tau_a
        HL_alpha[j,i] = tau_a
    for i,j in zip([5,7],[1,3]):
        HL_beta[i,j] = tau_b
        HL_beta[j,i] = tau_b

    HR_alpha = HL_alpha
    HR_beta  = HL_beta

    ## Links/Rechts-Zentrale Matrizen
    VCL_alpha = np.zeros(shape=(16,8))
    VCL_beta  = np.zeros(shape=(16,8))
    VCR_alpha = np.zeros(shape=(16,8))
    VCR_beta  = np.zeros(shape=(16,8))

    for i,j in zip([0,2],[4,6]):
        VCL_alpha[i,j] = v_a
        VCR_alpha[j,i] = v_a
    for i,j in zip([1,3],[5,7]):
        VCL_beta[i,j] = v_b
        VCR_beta[j,i] = v_b

    ## Links/Rechts Kueplungsmatrizen
    TL_alpha = np.zeros(shape=(8,8))
    TL_beta  = np.zeros(shape=(8,8))
    TR_alpha = np.zeros(shape=(8,8))
    TR_beta  = np.zeros(shape=(8,8))

    for i,j in zip([0,2],[4,6]):
        TL_alpha[i,j] = del_a
        TR_alpha[j,i] = del_a
    for i,j in zip([1,3],[5,7]):
        TL_beta[i,j] = del_b
        TR_beta[j,i] = del_b

    ## Ueberlappungsmatrizen
    SC  = np.identity(16)
    SL  = np.identity(8)
    SR  = np.identity(8)
    SCL = np.zeros(shape=(16,8))
    SCR = np.zeros(shape=(16,8))
    STL = np.zeros(shape=(8,8))
    STR = np.zeros(shape=(8,8))

    ## Speichern
    with h5py.File("H_all.h5", "w") as H_all_h5:
        H_all_h5.create_dataset("HC-alpha" , data=HC_alpha)
        H_all_h5.create_dataset("HL-alpha" , data=HL_alpha)
        H_all_h5.create_dataset("HR-alpha" , data=HR_alpha)
        H_all_h5.create_dataset("VCL-alpha", data=VCL_alpha)
        H_all_h5.create_dataset("VCR-alpha", data=VCR_alpha)
        H_all_h5.create_dataset("TL-alpha" , data=TL_alpha)
        H_all_h5.create_dataset("TR-alpha" , data=TR_alpha)
        H_all_h5.create_dataset("HC-beta"  , data=HC_beta)
        H_all_h5.create_dataset("HL-beta"  , data=HL_beta)
        H_all_h5.create_dataset("HR-beta"  , data=HR_beta)
        H_all_h5.create_dataset("VCL-beta" , data=VCL_beta)
        H_all_h5.create_dataset("VCR-beta" , data=VCR_beta)
        H_all_h5.create_dataset("TL-beta"  , data=TL_beta)
        H_all_h5.create_dataset("TR-beta"  , data=TR_beta)
    with h5py.File("S_all.h5", "w") as S_all_h5:
        S_all_h5.create_dataset("SC" , data=SC)
        S_all_h5.create_dataset("SL" , data=SL)
        S_all_h5.create_dataset("SR" , data=SR)
        S_all_h5.create_dataset("SCL", data=SCL)
        S_all_h5.create_dataset("SCR", data=SCR)
        S_all_h5.create_dataset("STL", data=STL)
        S_all_h5.create_dataset("STR", data=STR)

    with open("Ef", "w") as E_fermi:
        Ef_alpha = 0
        Ef_beta  = 0
        Ef = []
        Ef.append(Ef_alpha)
        Ef.append(Ef_beta)
        np.savetxt(E_fermi, Ef)

    head  = []
    for i in range(dimH):
        head.append(i+1)
    head_el = []
    for i in range(dimHe):
        head_el.append(i+1)

    print (tabulate( HC_alpha, tablefmt="fancy_grid", showindex=head,
           headers=head) )
    print (tabulate( HC_beta, tablefmt="fancy_grid", showindex=head,
           headers=head) )
    #print (tabulate( HL_alpha, tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( HL_beta,  tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( HR_alpha, tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( HR_beta,  tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( VCL_alpha, tablefmt="fancy_grid", showindex=head,
    #       headers=head_el) )
    #print (tabulate( VCL_beta,  tablefmt="fancy_grid", showindex=head,
    #       headers=head_el) )
    #print (tabulate( VCR_alpha, tablefmt="fancy_grid", showindex=head,
    #       headers=head_el) )
    #print (tabulate( VCR_beta,  tablefmt="fancy_grid", showindex=head,
    #       headers=head_el) )
    #print (tabulate( TL_alpha, tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( TL_beta,  tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( TR_alpha, tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( TR_beta,  tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )

def emp_H_1():

    ## Zentraler Hamiltonian
    HC_alpha = np.zeros(shape=(12,12))
    HC_beta  = np.zeros(shape=(12,12))

    dimH = HC_alpha.shape[0]
    for i in [0, 4]:
            HC_alpha[i,i]    = E_a_1
            HC_beta[i+1,i+1] = E_b_1
    for i in [8]:
            HC_alpha[i,i]    = eps_a_1
            HC_beta[i+1,i+1] = eps_b_1
    for i in [2, 6]:
            HC_alpha[i,i]    = E_a_2
            HC_beta[i+1,i+1] = E_b_2
    for i in [10]:
            HC_alpha[i,i]    = eps_a_1
            HC_beta[i+1,i+1] = eps_b_1

    # Kette Elemente
    for i,j in zip([4,6],[0,2]):
        HC_alpha[i,j] = tau_a
        HC_alpha[j,i] = tau_a
    for i,j in zip([5,7],[1,3]):
        HC_beta[i,j] = tau_b
        HC_beta[j,i] = tau_b

    # 1te und 2te SMM Alpha spin
    for i,j in zip([8,10],[0,4]):
        HC_alpha[i,j] = t_a
        HC_alpha[j,i] = t_a
        HC_alpha[i,j] = t_a
        HC_alpha[j,i] = t_a


    ## Links/Rechts Hamiltonian
    HL_alpha = np.zeros(shape=(8,8))
    HL_beta  = np.zeros(shape=(8,8))

    dimHe = HL_alpha.shape[0]
    for i in [0,4]:
        HL_alpha[i,i]    = E_a_1
        HL_beta[i+1,i+1] = E_b_1
    for i in [2,6]:
        HL_alpha[i,i]    = E_a_2
        HL_beta[i+1,i+1] = E_b_2

    # Hopping Elemente
    for i,j in zip([4,6],[0,2]):
        HL_alpha[i,j] = tau_a
        HL_alpha[j,i] = tau_a
    for i,j in zip([5,7],[1,3]):
        HL_beta[i,j] = tau_b
        HL_beta[j,i] = tau_b

    HR_alpha = HL_alpha
    HR_beta  = HL_beta

    ## Links/Rechts-Zentrale Matrizen
    VCL_alpha = np.zeros(shape=(12,8))
    VCL_beta  = np.zeros(shape=(12,8))
    VCR_alpha = np.zeros(shape=(12,8))
    VCR_beta  = np.zeros(shape=(12,8))

    for i,j in zip([0,2],[4,6]):
        VCL_alpha[i,j] = v_a
        VCR_alpha[j,i] = v_a
    for i,j in zip([1,3],[5,7]):
        VCL_beta[i,j] = v_b
        VCR_beta[j,i] = v_b

    ## Links/Rechts Kueplungsmatrizen
    TL_alpha = np.zeros(shape=(8,8))
    TL_beta  = np.zeros(shape=(8,8))
    TR_alpha = np.zeros(shape=(8,8))
    TR_beta  = np.zeros(shape=(8,8))

    for i,j in zip([0,2],[4,6]):
        TL_alpha[i,j] = del_a
        TR_alpha[j,i] = del_a
    for i,j in zip([1,3],[5,7]):
        TL_beta[i,j] = del_b
        TR_beta[j,i] = del_b

    ## Ueberlappungsmatrizen
    SC  = np.identity(12)
    SL  = np.identity(8)
    SR  = np.identity(8)
    SCL = np.zeros(shape=(12,8))
    SCR = np.zeros(shape=(12,8))
    STL = np.zeros(shape=(8,8))
    STR = np.zeros(shape=(8,8))

    ## Speichern
    with h5py.File("H_all.h5", "w") as H_all_h5:
        H_all_h5.create_dataset("HC-alpha" , data=HC_alpha)
        H_all_h5.create_dataset("HL-alpha" , data=HL_alpha)
        H_all_h5.create_dataset("HR-alpha" , data=HR_alpha)
        H_all_h5.create_dataset("VCL-alpha", data=VCL_alpha)
        H_all_h5.create_dataset("VCR-alpha", data=VCR_alpha)
        H_all_h5.create_dataset("TL-alpha" , data=TL_alpha)
        H_all_h5.create_dataset("TR-alpha" , data=TR_alpha)
        H_all_h5.create_dataset("HC-beta"  , data=HC_beta)
        H_all_h5.create_dataset("HL-beta"  , data=HL_beta)
        H_all_h5.create_dataset("HR-beta"  , data=HR_beta)
        H_all_h5.create_dataset("VCL-beta" , data=VCL_beta)
        H_all_h5.create_dataset("VCR-beta" , data=VCR_beta)
        H_all_h5.create_dataset("TL-beta"  , data=TL_beta)
        H_all_h5.create_dataset("TR-beta"  , data=TR_beta)
    with h5py.File("S_all.h5", "w") as S_all_h5:
        S_all_h5.create_dataset("SC" , data=SC)
        S_all_h5.create_dataset("SL" , data=SL)
        S_all_h5.create_dataset("SR" , data=SR)
        S_all_h5.create_dataset("SCL", data=SCL)
        S_all_h5.create_dataset("SCR", data=SCR)
        S_all_h5.create_dataset("STL", data=STL)
        S_all_h5.create_dataset("STR", data=STR)

    with open("Ef", "w") as E_fermi:
        Ef_alpha = 0.0
        Ef_beta  = 0.0
        Ef = []
        Ef.append(Ef_alpha)
        Ef.append(Ef_beta)
        np.savetxt(E_fermi, Ef)

    head  = []
    for i in range(dimH):
        head.append(i+1)
    head_el = []
    for i in range(dimHe):
        head_el.append(i+1)

    print (tabulate( HC_alpha, tablefmt="fancy_grid", showindex=head,
           headers=head) )
    print (tabulate( HC_beta, tablefmt="fancy_grid", showindex=head,
           headers=head) )
    #print (tabulate( HL_alpha, tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( HL_beta,  tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( HR_alpha, tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( HR_beta,  tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( VCL_alpha, tablefmt="fancy_grid", showindex=head,
    #       headers=head_el) )
    #print (tabulate( VCL_beta,  tablefmt="fancy_grid", showindex=head,
    #       headers=head_el) )
    #print (tabulate( VCR_alpha, tablefmt="fancy_grid", showindex=head,
    #       headers=head_el) )
    #print (tabulate( VCR_beta,  tablefmt="fancy_grid", showindex=head,
    #       headers=head_el) )
    #print (tabulate( TL_alpha, tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( TL_beta,  tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( TR_alpha, tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )
    #print (tabulate( TR_beta,  tablefmt="fancy_grid", showindex=head_el,
    #       headers=head_el) )

if __name__ == "__main__":
    emp_H()
    #emp_H_1()
