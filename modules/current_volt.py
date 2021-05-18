import scipy.integrate as integrate
import modules.mo_print as moprint
import matplotlib.pyplot as plt
import scipy.constants as phys
import scipy.interpolate
import modules.mo as mo
import pandas as pd
import numpy as np
import decimal
import math 
import time 
import sys
import os
from pandas import DataFrame as df
from functools import partial
from mpmath import mp


class iv_characteristics:
    ''' The goals of this module are:
    (.) Calculate the IV characteristics using the T(E) from one of the 
        methods in class fourteen matrices.
    (.) If desired, calculate the differential conductance maps (stability 
        maps)
    '''

    def __init__(self,config):
        self.Sname    = config["System name"]
        self.Spin     = config["Spin Polarized system"]
        self.NE       = config["Number of energy points"]
        self.Ea       = config["Lower energy border"]
        self.Eb       = config["Upper energy border"]
        self.G_calc   = config["Differential conductance calculation"]
        self.NVsd     = config["Number of sd-voltage points"]
        self.NVgs     = config["Number of gs-voltage points"]
        self.Vsd_a    = config["Lower source-drain voltage"]
        self.Vsd_b    = config["Upper source-drain voltage"]
        self.Vg_a     = config["Lower gate-source voltage"]
        self.Vg_b     = config["Upper gate-source voltage"]
        self.temp     = config["Temperature in Kelvin"]
        self.path_out = config["Path of output"]

        try:
            self.gate_correction = \
                               config["Gate correction to central Hamiltonian"]
        except KeyError:
            print ("Gating not applied to central Hamiltonian")
            self.gate_correction = "No"

        try:
            self.CB_regime = config["Coulomb blockade"]
        except KeyError:
            print ("No key value Coulomb blockade. Set to NO",\
                   flush=True, end='\n')
            self.CB_regime = "No"

    def iv_curves(self):
        info00 = ("Current Volt and differential conductance calculation.\n"
                  "NEGF+DFT\nSystem name: {}")
        print (moprint.ibordered_title(info00.format(self.Sname)),
               flush=True, end='\n')

        path = self.path_out
        if not os.path.exists(path):
            os.makedirs(path)
        
        espace = '   '

        VSD = np.linspace(self.Vsd_a,self.Vsd_b,self.NVsd,dtype=float)
        VG  = np.linspace(self.Vg_a,self.Vg_b,self.NVgs,dtype=float)
        
        # Units eV,s
        e_char = phys.elementary_charge
        hbar   = phys.physical_constants["Planck constant over 2 pi in eV s"][0]
        Planck = phys.physical_constants["Planck constant in eV s"][0]
        kB     = phys.physical_constants["Boltzmann constant in eV/K"][0]
        G0     = phys.physical_constants["conductance quantum"][0]
        info01 = ("Electron charge: {: .9e}\n"
                  "hbar (eV*s): {: .9e}\n"
                  "Boltzmann constant (eV/K): {: .9e}\n"
                  "Planck constant (eV*s): {: .9e}\n"
                  "Quantum conductance (S): {: .9e}")
        print (moprint.ibordered(info01.format(e_char, hbar, kB, Planck,G0)),
               flush=True, end='\n')

        # Electrochemical potential in the left electrode
        mu = 0
        # Temperature in Kelvin
        temp    = self.temp 
        tempstr = str(temp)
        
        if self.Spin in ["No","no","N","n"]:

            f1 = open(path+"/IVsd"+"-"+tempstr+"K"+".out","+w")
            f1.write("Current" + espace + "Vsd" + espace + espace + "Vg" +\
                     espace + "Conductance \n")

            # Read out the energies and transmission values
            f2 = mo.read_files(path+"/out.out",0,2,0)
            f2.write("# Vsd"+2*espace+"IVsd"+"\n")
            f2.write("# Temperature[K]: "+tempstr+"\n")
            transmission = f2[2]
            E = f2[0]
            
            for voltage in VSD:
                
                landau  = np.zeros(self.NE)
                '''
                Integrate Landau equation (using trapezoidal formula)
                '''
                for iE,(energy,trans) in enumerate(zip(E,transmission)):
                    try:
                        fermi_source = mo.fermi_func(energy, mu ,temp)
                        fermi_drain  = mo.fermi_func(energy, mu-voltage,temp)
                        landau[iE]   = trans * (fermi_source - fermi_drain)
                    except OverflowError:
                        fermi_source = float("inf")
                        fermi_drain  = float("inf")
                
                I  = (e_char**2)/Planck * np.trapz(landau,E) 
                
                f1.write("%.8f %.8f\n" %(voltage,I))
                f1.flush()

        elif self.Spin in ["Yes","yes","Y","y"]:
            if self.G_calc in ["No","no","N","n"]:
                #For a+b current
                for spin in ["alpha","beta"]:
                    #Read out the energies and transmission values
                    f2 = mo.read_files(path+"/out-"+spin+"-.out",0,2,0)
                    if spin == "alpha":
                        transmission_alpha = f2[2]
                    if spin == "beta":
                        transmission_beta  = f2[2]
                    
                E = f2[0]
                transmission_tot = transmission_alpha + transmission_beta 
                ener_trans   = df({'Energy':E, 'Transmission':transmission_tot})
                N_iv         = [0,10,60,160]
                voltage      = []
                result_apb   = []
                result_apbG  = []
                
                f3 = open(path+"/IVsd_a+b.out","+w")
                f3.write("# Vsd"+2*espace+"IVsd"+"\n")
                f3.write("# Temperature[K]: "+tempstr+"\n")
                
                for i, Nvg in enumerate(N_iv):
                    endE             = transmission_tot[len(E)-1] 
                    gated_trans      = ener_trans.Transmission.\
                                       shift(-Nvg * i).fillna(endE)
                    gated_ener_trans = df({"Energy":E,\
                                          "Transmission":gated_trans})
                    vg = round(i * Nvg *\
                               (gated_ener_trans.at[1, "Energy"] -\
                                gated_ener_trans.at[0, "Energy"]), 8)

                    print ("Vg(i): ", i, "\n" ,"vg(eV): ", vg, "\n\n")
                
                    landau      = np.zeros(len(E))
                    der_landau  = np.zeros(len(E))
                    current_apb = []
                    difcond_apb = []
                        
                    for vsdi,vsd in enumerate(VSD):
                        for row in gated_ener_trans.itertuples():
                            '''
                            Integrate Landau equation 
                            (using trapezoidal formula)
                            '''
                            der_I  = 0.
                            I      = 0.
                            energy = row[1]
                            trans  = row[2]

                            fermi_source = mo.fermi_func(energy,mu,temp)
                            fermi_drain  = mo.fermi_func(energy,mu-vsd/2.,temp)
                            der_I        = mo.exponent_fermi(energy,\
                                           mu-vsd/2. ,temp)/(1. +\
                                           mo.exponent_fermi(energy,\
                                           mu-vsd/2. ,temp))**2  
                            
                            landau[row[0]]     = trans*(fermi_source-fermi_drain)
                            der_landau[row[0]] = trans * der_I
                        
                        I      =  (e_char/Planck) * np.trapz(landau,E) 
                        G_cond = (1./(2. * kB * temp)) * np.trapz(der_landau,E)
                            
                        current_apb.append(I)
                        difcond_apb.append(G_cond)
                        if i == 0:
                            voltage.append(vsd)
                        elif i != 0:
                            pass
                    if i == 0:
                        result_apb.append(voltage)
                    elif i != 0:
                        pass
                    result_apb.append(current_apb)
                    result_apbG.append(difcond_apb)
                
                results_apb  = df(result_apb).T
                results_apbG = df(result_apbG).T
                
                results_apb = pd.concat([results_apb,results_apbG],axis=1)
                np.savetxt(f3, results_apb, delimiter='    ', fmt='% .13f')
                f3.flush()
                f3.close()

            # Conductance and IV maps
            elif self.G_calc in ["Yes", "yes", "Y", "y"]:
                if self.gate_correction in ["No", "no", "N", "n"]:
                    for spin in ["alpha","beta"]:
                        # Read out the energies and transmission values
                        f2 = mo.read_files(path+"/out-"+spin+"-sp-g-1eV.out", 0, 2, 0)
                        if spin == "alpha":
                            transmission_alpha = f2[2]
                        if spin == "beta":
                            transmission_beta = f2[2]
                        E = f2[0]

                    transmission_tot = transmission_alpha + transmission_beta 
                    landau           = np.zeros(len(E))
                    der_landau       = np.zeros(len(E))
                        
                    f1 = open(path+"/Conductance-sp-1eV.out","+w")
                    f1.write("# Vg"+2*espace+"Vsd"+2*espace+"Conductance"+"\n")
                    f1.write("# Temperature[K]: "+tempstr+"\n")

                    ener_trans = df({'Energy':E, 'Transmission':transmission_tot})
                    
                    #for i in range(-self.NVgs+300,self.NVgs):
                    for i in range(-self.NVgs, self.NVgs):
                        if i >= 0: 
                            endE = transmission_tot[0] 
                        elif i < 0:
                            endE = transmission_tot[len(E)-1] 
                        gated_trans = ener_trans.Transmission.\
                                      shift(periods=i).fillna(endE)
                        gated_ener_trans = df({'Energy':E,\
                                              'Transmission':gated_trans})
                        vg = round(-i * (ener_trans.at[1, 'Energy'] -
                                         ener_trans.at[0, 'Energy']), 8)
                        inter_trans = mo.inter_trans(gated_ener_trans['Energy'],\
                                                     gated_ener_trans['Transmission'])
                        
                        print ("Vg(i): ", i, "\n" ,"vg(eV): ", vg, "\n\n",\
                               flush=True)
                        #Enew = np.linspace(-0.5,0.5,10000)
                        #ynew = inter_trans(Enew)
                        #plt.plot(Enew,ynew,linewidth=1.5)
                        #plt.xticks(np.arange(-0.5,0.5,0.1))
                        #plt.minorticks_on()
                        #plt.show()
                        #sys.exit()
                        for vsd in VSD:
                            ''' Using trapz'''
                            #for row in gated_ener_trans.itertuples():
                            #    der_I  = 0.
                            #    energy = row[1]
                            #    trans  = row[2]
                            #    
                            #    fermi_source = mo.fermi_func(energy,mu,temp)
                            #    fermi_drain  = mo.fermi_func(energy,mu-vsd/2.,temp)
                            #    der_I  = 0.5 * 1./(np.cosh((energy-\
                            #                        (mu-vsd/2.))/(kB *  temp)) + 1.)
                            #    der_I  = 0.25 * (1./np.cosh(0.5 *\
                            #                     (energy-(mu-vsd/2.))/(kB *  temp)))**2

                            #    landau[row[0]] = trans * (fermi_source - fermi_drain)
                            #    der_landau[row[0]]  = trans * der_I
                            
                            #der_landau = int_der_landau(gated_ener_trans)
                            #I       = abs((e_char/Planck) * np.trapz(landau,E))
                            #G_cond  = G0 * (1./(2. * kB * temp)) *\
                            #           np.trapz(der_landau,E)

                            #f1.write("%.10f %.10f %.12f\n" %(voltage_g,vsd,I))

                            '''Using integrate.quad'''
                            int_landau = lambda x,a,b,c,d: inter_trans(x) \
                                         *((1./(np.exp(x-(a+d/2.))**(1./(b*c))+1.)) \
                                         -(1./(np.exp(x-(a-d/2.))**(1./(b*c))+1.))) 
                            
                            int_der_landau = lambda x,a,b,c,d: inter_trans(x) \
                                             * 0.5 * (1./(2.*b*c)) \
                                             * (1./(1.+mp.cosh((x-(a+d/2.))/(b*c))) \
                                             + 1./(1.+mp.cosh((x-(a-d/2.))/(b*c))))

                            I, err_I = integrate.quad(int_landau, np.float64(E[0]),\
                                                      np.float64(E[-1]),\
                                                      args=(mu,kB,temp,vsd,),\
                                                      limit=60,epsabs=0)
                            I = G0 * I  

                            G_cond, G_err = integrate.quad(int_der_landau,\
                                                           np.float64(E[0]),\
                                                           np.float64(E[-1]),\
                                                           args=(mu,kB,temp,vsd,),\
                                                           limit=60, epsabs=0)
                            G_cond = G0 * G_cond
                            
                            f1.write("{: .10f} {: .10f} {: .14f} {: .14f} "
                                     "{: .14f}\n".format(vg, vsd, abs(I), G_cond, I))
                        f1.write("\n")
                        f1.flush()
                    f1.close()

                elif self.gate_correction in ["Yes", "yes", "Y", "y"]:
                    for spin in ["alpha","beta"]:
                        # Read out the gate voltages, energies and transmission 
                        # values
                        f2 = mo.read_files(path+"/out-"+spin+"-sp-g-1eV.out",
                                           0, 3, 0)
                        if spin == "alpha":
                            transmission_alpha_sp = f2[3]
                        elif spin == "beta":
                            transmission_beta_sp = f2[3]
                        # Get just the non-repeated elements of Gate voltages...
                        vgvf = f2[0]
                        VG   = []
                        for i in range(len(vgvf)):
                            if i == 0:
                                VG.append(vgvf[i])
                            else:
                                if vgvf[i] == vgvf[i-1]:
                                    continue
                                else:
                                    VG.append(vgvf[i])
                        VG = np.asarray(VG, dtype=float)

                        # ... and of Energies.
                        evf = f2[1]
                        E   = []
                        for j in range(len(evf)):
                            if j == 0:
                                E.append(evf[j])
                            else:
                                if evf[j] in E:
                                    continue
                                else:
                                    E.append(evf[j])
                        E = np.asarray(E, dtype=float)

                    transmission_tot_sp = transmission_alpha_sp\
                                          + transmission_beta_sp

                    landau_sp     = np.zeros(len(E))
                    der_landau_sp = np.zeros(len(E))

                    f1 = open(path+"/Conductance_sp-g.out","+w")
                    f1.write("# Temperature[K]: " + tempstr + "\n")
                    f1.write("# Vg" + 12*espace + "Vsd" + 11*espace + "|I|" + 
                             15*espace + "Conductance" + 7*espace + "I" + "\n")

                    for ivg, vg in enumerate(VG):
                        # Chunks of Transmission per gate Voltage
                        start_T  = ivg * (len(E) - 1) + ivg 
                        end_T    = (ivg + 1) * len(E)
                        Trans_vg = np.asarray(transmission_tot_sp[start_T:end_T])

                        inter_trans = mo.inter_func(E, Trans_vg)

                        info02 = ("ivg: {}\nGate Voltage(V): {: .8f}") 
                        print (moprint.ibordered(info02.format(ivg, vg)), 
                               flush=True, end='\n')
                        print (moprint.iprint_line(), flush=True, end='\n')
                        print ("ivsd" + 7 * ' ' + 'Source-Drain Voltage(V)',
                               flush=True,end='\n')
                        print (moprint.iprint_line())

                        for ivsd, vsd in enumerate(VSD):
                            info03 = ("{:2d} {:7} {: .8f}") 
                            print (info03.format(ivsd, ' ', vsd),
                                   flush=True, end='\n')

                            # Integration using integrate.quad
                            int_landau = lambda x,a,b,c,d: inter_trans(x)\
                                     *((1./(np.exp(x-(a+d/2.))**(1./(b*c))+1.))\
                                     -(1./(np.exp(x-(a-d/2.))**(1./(b*c))+1.))) 
                            
                            int_der_landau = lambda x,a,b,c,d: inter_trans(x)\
                                                    * 0.5 * (1./(2.*b*c))\
                                                    * (1./(1.+mp.cosh((x\
                                                    - (a+d/2.))/(b*c)))\
                                                    + 1./(1.+mp.cosh((x\
                                                    - (a-d/2.))/(b*c))))

                            I, err_I = integrate.quad(int_landau,np.float64(E[0]),
                                                      np.float64(E[len(E)-1]),\
                                                      args=(mu,kB,temp,vsd,),\
                                                      limit=80,epsabs=0)
                            G_cond, G_err = integrate.quad(int_der_landau,
                                                     np.float64(E[0]),\
                                                     np.float64(E[len(E)-1]),\
                                                     args=(mu, kB, temp, vsd,),\
                                                     limit=80, epsabs=0)
                            I      = G0 * I  
                            G_cond = G0 * G_cond
                            
                            f1.write("{: .10f} {: .10f} {: .14f} {: .14f} "
                                     "{: .14f}\n".format(vg, vsd, abs(I),
                                                         G_cond, I))
                            f1.flush()
                        f1.write("\n")
                        print (moprint.iprint_line())
                        print ("\n",flush=True,end='\n')

                else:
                    print ("Option not recognized")
                    sys.exit()


    def iv_curves_cb(self):
        info00 = ("Current Volt and differential conductance calculation.\n"
                  "Coulomb blockade NEGF+DFT\nSystem name: {}")
        print (moprint.ibordered_title(info00.format(self.Sname)),
               flush=True, end='\n')

        path = self.path_out
        if not os.path.exists(path):
            os.makedirs(path)
        
        espace = ' '

        VSD = np.linspace(self.Vsd_a, self.Vsd_b, self.NVsd, dtype=float)

        # Units eV,s
        e_char = phys.elementary_charge
        hbar   = phys.physical_constants["Planck constant over 2 pi in eV s"][0]
        Planck = phys.physical_constants["Planck constant in eV s"][0]
        kB     = phys.physical_constants["Boltzmann constant in eV/K"][0]
        G0     = phys.physical_constants["conductance quantum"][0]

        info01 = ("Electron charge            : {: >15.9e}\n"
                  "hbar                (eV*s) : {: >15.9e}\n"
                  "Boltzmann constant  (eV/K) : {: >15.9e}\n"
                  "Planck constant     (eV*s) : {: >15.9e}\n"
                  "Quantum conductance (S)    : {: >15.9e}")
        print (moprint.ibordered(info01.format(e_char, hbar, kB, Planck, G0)),
               flush=True, end='\n')
        
        # Initial electrochemical potential of the whole system
        mu = 0 

        # Temperature in Kelvin
        temp    = self.temp 
        tempstr = str(temp)
        
        # Conductance and IV maps for Coulomb blockade
        if self.G_calc in ["No","no","N","n"]:
            None
            sys.exit()

        elif self.G_calc in ["Yes","yes","Y","y"]:
            for spin in ["alpha","beta"]:
                # Read out the gate voltages, energies and transmission 
                # values
                f2 = mo.read_files(path+"/out-"+spin+"-sp-g.out", 0, 3, 0)
                if spin == "alpha":
                    transmission_alpha_cb = f2[3]
                elif spin == "beta":
                    transmission_beta_cb = f2[3]
                
                # Get just the non-repeated elements of Gate voltages ...
                vgvf = f2[0]
                VG   = []
                for i in range(len(vgvf)):
                    if i == 0:
                        VG.append(vgvf[i])
                    else:
                        if vgvf[i] == vgvf[i-1]:
                            continue
                        else:
                            VG.append(vgvf[i])
                VG = np.asarray(VG, dtype=float)

                # ... and of Energies.
                evf = f2[1]
                E   = []   
                for j in range(len(evf)):
                    if j == 0:
                        E.append(evf[j])
                    else:
                        if evf[j] in E:
                            continue
                        else:
                            E.append(evf[j])
                E = np.asarray(E, dtype=float)

            transmission_tot_cb = transmission_alpha_cb + transmission_beta_cb
            landau_cb           = np.zeros(len(E))
            der_landau_cb       = np.zeros(len(E))

            f1 = open(path+"/Conductance_cb.out","+w")
            f1.write("# Temperature[K]: " + tempstr + "\n")
            f1.write("# Vg" + 12*espace + "Vsd" + 11*espace + "|I|" + 
                     15*espace + "Conductance" + 7*espace + "I" + "\n")

            for ivg, vg in enumerate(VG):
                # Chunks of Transmission per gate Voltage
                start_T  = ivg * (len(E) - 1) + ivg 
                end_T    = (ivg + 1) * len(E)
                Trans_vg = np.asarray(transmission_tot_cb[start_T:end_T])

                inter_trans = mo.inter_func(E, Trans_vg)

                info02 = ("ivg: {}\nGate Voltage(V): {: .8f}") 
                print (moprint.ibordered(info02.format(ivg, vg)), 
                       flush=True, end='\n')
                print (moprint.iprint_line(), flush=True, end='\n')

                #Enew = np.linspace(E[0],E[4],10)
                #ynew = inter_trans(Enew)
                #plt.plot(Enew,ynew,linewidth=1.5)
                #plt.xticks(np.arange(E[0],E[4],0.1))
                #plt.minorticks_on()
                #plt.show()
                #plt.savefig("interpol.png")

                print ("ivsd" + 7 * ' ' + 'Source-Drain Voltage(V)',flush=True,end='\n')
                print (moprint.iprint_line())
                for ivsd, vsd in enumerate(VSD):
                    info03 = ("{:2d} {:7} {: .8f}") 
                    print (info03.format(ivsd, ' ', vsd), flush=True, end='\n')
                    # Integration using integrate.quad
                    int_landau = lambda x,a,b,c,d: inter_trans(x) \
                                 *((1./(np.exp(x-(a+d/2.))**(1./(b*c))+1.)) \
                                 -(1./(np.exp(x-(a-d/2.))**(1./(b*c))+1.)) ) 
                    
                    int_der_landau = lambda x,a,b,c,d: inter_trans(x) \
                                            * 0.5 * (1./(2.*b*c)) \
                                            * (1./(1.+mp.cosh((x \
                                            - (a+d/2.))/(b*c))) \
                                            + 1./(1.+mp.cosh((x \
                                            - (a-d/2.))/(b*c))))

                    I, err_I = integrate.quad(int_landau,np.float64(E[0]),
                                              np.float64(E[len(E)-1]), \
                                              args=(mu,kB,temp,vsd,), \
                                              limit=80,epsabs=0)
                    G_cond, G_err = integrate.quad(int_der_landau,
                                                   np.float64(E[0]), \
                                                   np.float64(E[len(E)-1]), \
                                                   args=(mu, kB, temp, vsd,), \
                                                   limit=80, epsabs=0)
                    I      = G0 * I  
                    G_cond = G0 * G_cond
                    
                    f1.write("{: .10f} {: .10f} {: .14f} {: .14f} {: .14f}\n"\
                             .format(vg, vsd, abs(I), G_cond, I))
                    f1.flush()
                f1.write("\n")
                print (moprint.iprint_line())
                print ("\n",flush=True,end='\n')

#############################################################################
# Different ways to integrate the Transmission coefficient. Copy and paste  #
# them if necessary in the integration part.                                #
#############################################################################

#int_der_landau = lambda y,a,b,c,d: inter_trans(y) * 0.25 * \
#                       (\
#                       1. - (mp.tanh((y-(a-d/2.))/(2.*b*c)))**2
#                       ) 
#int_der_landau = lambda y,a,b,c,d: 1. * 0.25 * inter_trans(y) if \
#                                   math.isclose(abs(y),abs((a-d/2.)),rel_tol=1E-1) \
#                                   else 0

#if voltage_sd <= 0:
#    int_der_landau = lambda x,a,b,c,d: inter_trans(x) * 0.5 * (1./(2.*b*c)) * \
#                               (\
#                               1./(1. + mp.cosh((x-(a-d/2.))/(b*c)))\
#                               )
#elif voltage_sd > 0:
#    int_der_landau = lambda x,a,b,c,d: inter_trans(x) * 0.5 * (1./(2.*b*c)) * \
#                               (\
#                               1./(1. + mp.cosh((x-(a+d/2.))/(b*c)))\
#                               )

#int_der_landau = lambda x,a,b,c,d: inter_trans(x) * 0.5 * (1./(2.*b*c)) * \
#                        (\
#                        1./(1. + mp.cosh((x-(a-d/2.))/(b*c)))\
#                        )
