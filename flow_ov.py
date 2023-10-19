# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:29:54 2023

@author: ovalencia
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos= pd.read_csv('https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/gait_intrisec_muscles_and_acc_2000hz/emg_gait.csv')
cvm_di= pd.read_csv('https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/gait_intrisec_muscles_and_acc_2000hz/EMG_CVM_DIEDB.csv')


'''1er Dorsal interossei (DI)-EMG9, Flexor Dallucis Drevis (FHB)-EMG10
Extensor Digitorum Drevis (EDB)-EMG11, Abductor Hallucis (AbH)-EMG12'''
#%%

'''extrar vectores'''
di= datos.EMG11.values #tarea funcional
di_cvm= cvm_di.EMG11.values #cvm de di

#%%
'''usando la función ajusta_emg_func'''

edb= ajusta_emg_func(di, di_cvm, 2000, 20, 2, 'DI',show=True) 

prom= np.mean(edb)
maxi= np.max(edb)

#%%
'''guardando datos en .csv'''

DK=str(input('ID voluntario= ')) #cambiar nombre 1P por nombre de la persona (iniciales)

Res=np.vstack([prom,maxi])
df1=pd.DataFrame(Res.T, columns=['prom','maxi'])
df1.to_csv(DK + '.csv')







