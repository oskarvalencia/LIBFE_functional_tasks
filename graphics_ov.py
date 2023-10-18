# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:29:54 2023

@author: ovale
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos= pd.read_csv('https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/gait_intrisec_muscles_and_acc_2000hz/emg_gait.csv')
'''1er Dorsal interossei (DI)-EMG9, Flexor Dallucis Drevis (FHB)-EMG10
Extensor Digitorum Drevis (EDB)-EMG11, Abductor Hallucis (AbH)-EMG12'''
emg1= datos.iloc[:,34].values
emg11= datos.EMG9.values
acc_x= datos.ACCY8.values*0.001 #convertir mm a m

#%%

def t_(d,f):
    '''creando una base temporal de un vector d adquirido a una frecuencia
    de muestreo f'''
    return np.linspace(0, len(d)*(1/f), len(d))

#%%
t= t_(emg1,2000)


fig, axs = plt.subplots(1, 2, figsize=(15, 5),
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.5},dpi=150)

axs[0].plot(t,acc_x,'g', label='acc_x')
axs[0].set_title('Acelerometría', fontsize=20)
axs[0].set_ylabel('Acc [m/s2]', fontsize=14)
axs[0].set_xlabel('Tiempo [s]', fontsize=14)
axs[0].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[0].axvspan(0,1, facecolor='lightcoral', alpha=0.6)
axs[0].grid()

axs[1].plot(t,emg1,'r', label= 'DI')
#axs[1].plot(t,emg1,'b', label= 'Bíceps Femoral') #puede agregar otro músculo
axs[1].set_title('Electromiografía', fontsize=20)
axs[1].set_xlabel('Tiempo [s]', fontsize=14)
axs[1].set_ylabel('Amplitud EMG [V]', fontsize=14)
axs[1].axvspan(0,1, facecolor='lightcoral', alpha=0.6)
#axs[1].axvline((np.max(t)*0.6), color='g', ls= 'dotted', lw=3) #crea una línea vertical
axs[1].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[1].grid()








