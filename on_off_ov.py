# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:29:54 2023

@author: ovalencia
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from detecta import detect_onset

datos= pd.read_csv('https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/on_off/abd_acc.csv')
cvm_bb= pd.read_csv('https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/on_off/cvm_bb.csv')
cvm_dm= pd.read_csv('https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/on_off/cvm_dm.csv')

'''DM (columna 0), BB (columna_1), ACC_emg (columna_2),
ACC_x (columna_3), ACC_y (columna_4), ACC_z (columna_5)
'''
#%%

'''extrar vectores, bíceps braquial funcional y contracción voluntaria máxima'''
Bb= datos.iloc[:10000,0].values #tarea funcional
Bb_cvm= cvm_dm.iloc[:10000,0].values #cvm de di



#%%
'''usando la función ajusta_emg_func, es importante cargar el archivo previamente'''

Bb_fun= ajusta_emg_func(Bb, Bb_cvm, 1000, 20, 2, 'DM',show=True)

#%%

plt.plot(datos.iloc[:,3].values, label='X')
plt.plot(datos.iloc[:,4].values, label='Y')
plt.plot(datos.iloc[:,5].values, label='Z')
plt.legend(loc='best')

acc_z= datos.iloc[:10000,5].values

#%%

fig, axs = plt.subplots(2, 1, figsize=(10, 8),
                        gridspec_kw={'hspace': 0.5, 'wspace': 0.5},dpi=150)

axs[0].plot(acc_z,'g', label='acc_z')
axs[0].set_title('Acelerometría', fontsize=20)
axs[0].set_ylabel('Acc [m/s2]', fontsize=14)
axs[0].set_xlabel('Tiempo [s]', fontsize=14)
axs[0].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
#axs[0].axvspan(0,1, facecolor='lightcoral', alpha=0.6)
axs[0].grid()

axs[1].plot(Bb_fun,'r', label= 'BB')
#axs[1].plot(t,emg1,'b', label= 'Bíceps Femoral') #puede agregar otro músculo
axs[1].set_title('Electromiografía', fontsize=20)
axs[1].set_xlabel('Tiempo [s]', fontsize=14)
axs[1].set_ylabel('Amplitud EMG [V]', fontsize=14)
#axs[1].axvspan(0,1, facecolor='lightcoral', alpha=0.6)
#axs[1].axvline((np.max(t)*0.6), color='g', ls= 'dotted', lw=3) #crea una línea vertical
axs[1].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[1].grid()



#%%

'''La función detect_onset fue extarída desde:
   https://nbviewer.org/github/demotu/detecta/blob/master/docs/detect_onset.ipynb'''

on_acc= detect_onset(acc_z, np.mean(acc_z[500:1000])+(np.std(acc_z[500:1000])*4),
                 n_above=500, n_below=0, show=True)

on_bb= detect_onset(Bb_fun, np.mean(Bb_fun[500:1000])+(np.std(Bb_fun[500:1000])*4),
                 n_above=500, n_below=0, show=True)

time= on_bb[0,0]-on_acc[0,0]
prom= round(np.mean(Bb_fun[on_bb[0,0]:on_bb[0,1]]),2)

#%%

fig, axs = plt.subplots(2, 1, figsize=(10, 8),
                        gridspec_kw={'hspace': 0.5, 'wspace': 0.5},dpi=150)

axs[0].plot(acc_z,'g', label='acc_z')
axs[0].set_title('Acelerometría', fontsize=20)
axs[0].set_ylabel('Acc [m/s2]', fontsize=14)
axs[0].set_xlabel('tiempo [ms]', fontsize=14)
axs[0].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[0].axvline(on_acc[0,0], color='b', ls= '-', lw=2) #crea una línea vertical
axs[0].grid()

axs[1].plot(Bb_fun,'r', label= f'BB_onset = {time} [ms]')
#axs[1].plot(t,emg1,'b', label= 'Bíceps Femoral') #puede agregar otro músculo
axs[1].set_title('Electromiografía', fontsize=20)
axs[1].set_xlabel('tiempo [ms]', fontsize=14)
axs[1].set_ylabel('Amplitud EMG [%CVM]', fontsize=14)
axs[1].axvspan(on_acc[0,0],on_bb[0,0], facecolor='black', alpha=0.2)
axs[1].axvline(on_bb[0,0], color='k', ls= '-', lw=2) #crea una línea vertical
axs[1].axvline(on_acc[0,0], color='b', ls= '-', lw=2)
axs[1].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[1].grid()
#%%
'''guardando datos en .csv'''

DK=str(input('ID voluntario= ')) #cambiar nombre 1P por nombre de la persona (iniciales)

Res=np.vstack([time,prom])
df1=pd.DataFrame(Res.T, columns=['time','prom'])
df1.to_csv(DK + '.csv')







