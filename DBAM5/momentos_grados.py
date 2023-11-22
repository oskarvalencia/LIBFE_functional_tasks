# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:26:31 2023

@author: ovale
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/DBAM5/datos5.csv', skiprows=4)

#%%

def rigidez_(data, side):
    if side == 'izquierdo':
        r = [5,11,40,44,50,56,62]
    elif side == 'derecho':
        r = [104,110,139,143,149,155,161]
    lado = r
    cdm_pelvis= data.iloc[:,97].values / 1000
    fuerza_z= data.iloc[:,lado[2]].values
    
    # Buscar el primer índice donde la fuerza_z sea mayor que 0.1 N/kg
    inicio = next((p for p, v in enumerate(fuerza_z) if v > 0.01), None)
    final = np.argmin(cdm_pelvis)
    
    fe_tobillo= data.iloc[inicio:final,lado[0]].values
    mom_tobillo= data.iloc[inicio:final,lado[1]].values / 1000 # Nmm/kg a Nm/kg
    fe_cadera= data.iloc[inicio:final,lado[3]].values
    mom_cadera= data.iloc[inicio:final,lado[4]].values / 1000
    fe_rodilla= data.iloc[inicio:final,lado[5]].values
    mom_rodilla= data.iloc[inicio:final,lado[6]].values / 1000
    
    # fuerza Z vs CDM
    cdm_pelvis_= abs((cdm_pelvis[inicio:final])-cdm_pelvis[inicio])
    fuerza_= fuerza_z[inicio:final]
    max_= np.argmax(fuerza_)+1
    d_pend, d_int= np.polyfit(cdm_pelvis_[:max_],fuerza_[:max_],1)
    
  
    # momento tobillo vs planti/dorsiflexion tobillo
    
    max_t= np.argmax(mom_tobillo)+1
    d_pendt, d_intt= np.polyfit(fe_tobillo[:max_t],mom_tobillo[:max_t],1)
    
   
    # momento rodilla vs F/E rodilla
    
    max_r= np.argmax(mom_rodilla)+1
    d_pendr, d_intr= np.polyfit(fe_rodilla[:max_r],mom_rodilla[:max_r],1)
  
    # momento cadera vs F/E cadera
    
    max_c= np.argmax(mom_cadera)+1
    d_pendc, d_intc= np.polyfit(fe_cadera[:max_c],mom_cadera[:max_c],1)
       
    fig, axs = plt.subplots(2, 2, figsize=(14,6), sharex=False, sharey=False, 
                            gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
    fig.suptitle('Lado ' + str(side))
    

    # fuerza vs desplazamiento
    axs[0,0].plot(cdm_pelvis_,fuerza_,'bo-')
    axs[0,0].plot(cdm_pelvis_[:max_],d_pend*cdm_pelvis_[:max_]+d_int,"r",label=f'pend = {d_pend:.4f}')
    axs[0,0].set_ylabel('Fuerza [N/kg]')
    axs[0,0].set_xlabel('Desplazamiento-cdm [m]')
    axs[0,0].legend(loc='upper right')


    # momento tobillo vs f/e tobillo
    axs[0,1].plot(fe_tobillo,mom_tobillo,'bo-')
    axs[0,1].plot(fe_tobillo[:max_t],d_pendt*fe_tobillo[:max_t]+d_intt,"r",label=f'pend = {d_pendt:.4f}')
    axs[0,1].set_ylabel('Momento [Nm/kg]')
    axs[0,1].set_xlabel('Fle/Ext tobillo [°]')
    axs[0,1].legend(loc='upper right')
    
    # momento rodilla vs f/e rodilla
    axs[1,0].plot(fe_rodilla,mom_rodilla,'bo-')
    axs[1,0].plot(fe_rodilla[:max_r],d_pendr*fe_rodilla[:max_r]+d_intr,"r",label=f'pend = {d_pendr:.4f}')
    axs[1,0].set_ylabel('Momento [Nm/kg]')
    axs[1,0].set_xlabel('Fle/Ext rodilla [°]')
    axs[1,0].legend(loc='upper right')
    
    # momento cadera vs f/e cadera
    axs[1,1].plot(fe_cadera,mom_cadera,'bo-')
    axs[1,1].plot(fe_cadera[:max_c],d_pendc*fe_cadera[:max_c]+d_intc,"r", label=f'pend = {d_pendc:.4f}')
    axs[1,1].set_ylabel('Momento [Nm/kg]')
    axs[1,1].set_xlabel('Fle/Ext cadera [°]')
    axs[1,1].legend(loc='upper right')
    
    return f'''Lado {str(side)}
    \nmomento tobillo vs f/e tobillo {d_pend:.4f} N/kg/m
    \nmomento tobillo vs f/e tobillo {d_pendt:.4f} Nm/kg/°
    \nmomento rodilla vs f/e rodilla {d_pendr:.4f} Nm/kg/°
    \nmomento cadera vs f/e cadera {d_pendc:.4f} Nm/kg/°'''

#%%

# cargando datos y seleccionando lado izquierdo o derecho
rigidez_(data, 'izquierdo')
rigidez_(data, 'derecho')

