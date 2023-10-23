# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:29:54 2023

@author: ovalencia
"""

#%%
# En esta sección hay 3 errores

import numpy as
import as pd
import matplotlib.pyplot as plt

datos_cine= pd.read_csv("https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/gait_bagnoli_vicon/CINE_GAIT",
                    sep=';')

'''L= Left, R= Right, A= Ankle, K= Knee, H= Hip, X= sagittal plane,
 Y= frontal plane, Z= transverse plane'''
#%%
# En este apartado hay 2 errores

'''extrar dos vectores, F-E y ABD-ADD de rodilla, izquierda y derecha '''
F_E_K_left= datos_cine.LHX.values #izquierda
F_E_K_right= datos_cine.RKX.values #derecha

#Abd_Add_K_left= datos_cine.LKY.values #izquierda
Abd_Add_K_right= datos_cine.RKY.values #derecha

t= np.linspace(0,len(datos_cine.LKX)*(1/100),len(datos_cine.LKX))
#%%

# En esta sección hay 1 error repetido

'''usando la función ajusta_emg_func'''

fig, axs = plt.subplots(2, 2, figsize=(10, 8),
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.2},dpi=150)

axs[0,0].plot(t,F_E_K_right,'g', label=f'ROM = {np.max(F_E_K_right)-np.min(F_E_K_right):.2f} [°]')
axs[0,0].set_title('Rodilla der', fontsize=16)
axs[0,0].set_ylabel('Ext/Fle [°]', fontsize=14)
axs[0,0].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[0,0].axhline(np.max(F_E_K_right), color='b', ls= '--', lw=1) #crea una línea vertical
axs[0,0].axhline(np.min(F_E_K_right), color='b', ls= '--', lw=1)
axs[0,0].grid()

axs[0,1].plot(t,F_E_K_left,'r', label=f'ROM = {np.max(F_E_K_left)-np.min(F_E_K_left):.2f} [°]')
axs[0,1].set_title('Rodilla izq', fontsize=16)
axs[0,1].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[0,1].axhline(np.max(F_E_K_left), color='b', ls= '--', lw=1) #crea una línea vertical
axs[0,1].axhline(np.min(F_E_K_left), color='b', ls= '--', lw=1)
axs[0,1].grid()

axs[1,0].plot(t,Abd_Add_K_right,'g', label=f'ROM = {np.max(Abd_Add_K_right)-np.min(Abd_Add_K_right):.2f} [°]')
axs[1,0].set_ylabel('Abd/Add [°]', fontsize=14)
axs[1,0].set_xlabel('tiempo [s]', fontsize=14)
axs[1,0].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[1,0].axhline(np.max(Abd_Add_K_right), color='b', ls= '--', lw=1) #crea una línea vertical
axs[1,0].axhline(np.min(Abd_Add_K_right), color='b', ls= '--', lw=1)
axs[1,0].grid()

axs[1,0].plot(t,Abd_Add_K_left,'r', label=f'ROM = {np.max(Abd_Add_K_left)-np.min(Abd_Add_K_left):.2f} [°]')
axs[1,0].set_xlabel('tiempo [s]', fontsize=14)
axs[1,0].legend(loc= 'upper left', fontsize= 'medium', borderpad=0.9)
axs[1,0].axhline(np.max(Abd_Add_K_left), color='b', ls= '--', lw=1) #crea una línea vertical
axs[1,0].axhline(np.min(Abd_Add_K_left), color='b', ls= '--', lw=1)
axs[1,0].grid()

plt.savefig('fig_cine_error.png')

#%%
rom_rk_fe= round(np.max(F_E_K_right)-np.min(F_E_K_right),2)
rom_lk_fe= round(np.max(F_E_K_left)-np.min(F_E_K_left),2)
rom_rk_abd= round(np.max(Abd_Add_K_right)-np.min(Abd_Add_K_right),2)
rom_lk_abd= round(np.max(Abd_Add_K_left)-np.min(Abd_Add_K_left),2)

#%%

# En esta sección hay 1 error

'''guardando datos en .csv'''

DK=str(input('ID voluntario= ')) #cambiar nombre 1P por iniciales del voluntario (a)

Res=np.vstack([rom_rk_fe,rom_lk_fe,rom_rk_abd,rom_lk_abd])
df1=pd.DataFrame(Res, columns=['f_e_rod_d','f_e_rod_i','abd_rod_d','abd_rod_i'])
df1.to_csv(DK + '.csv')







