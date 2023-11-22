# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:26:31 2023

@author: ovale
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('datos5.csv', skiprows=4)
cdm_pelvis= data.iloc[:,97].values
fuerza_z= data.iloc[:,40].values

# Buscar el primer índice donde la fuerza_z sea mayor que 0.1 N/kg
inicio = next((p for p, v in enumerate(fuerza_z) if v > 0.01), None)
final = np.argmin(cdm_pelvis)


fe_tobillo= data.iloc[inicio:final,5].values
mom_tobillo= data.iloc[inicio:final,11].values / 1000
fe_cadera= data.iloc[inicio:final,44].values
mom_cadera= data.iloc[inicio:final,50].values / 1000
fe_rodilla= data.iloc[inicio:final,56].values
mom_rodilla= data.iloc[inicio:final,62].values / 1000

#%%



#%%

# fuerza Z vs CDM
cdm_pelvis_= abs((cdm_pelvis[inicio:final])-cdm_pelvis[inicio])
fuerza_= fuerza_z[inicio:final]
max_= np.argmax(fuerza_)+1
d_pend, d_int= np.polyfit(cdm_pelvis_[:max_],fuerza_[:max_],1)
plt.figure()
plt.plot(cdm_pelvis_,fuerza_,'b',label=f'pend = {d_pend:.2f}')
plt.plot(cdm_pelvis_[:max_],d_pend*cdm_pelvis_[:max_]+d_int,"r")
plt.ylabel('Fuerza [N/kg]')
plt.xlabel('Desplazamiento [mm]')
plt.legend(loc='best')

# momento tobillo vs planti/dorsifelxion tobillo

max_t= np.argmax(mom_tobillo)+1
d_pendt, d_intt= np.polyfit(fe_tobillo[:max_t],mom_tobillo[:max_t],1)

plt.figure()
plt.plot(fe_tobillo,mom_tobillo,'b',label=f'pend = {d_pendt:.2f}')
plt.plot(fe_tobillo[:max_t],d_pendt*fe_tobillo[:max_t]+d_intt,"r")
plt.ylabel('Fuerza [N/kg]')
plt.xlabel('Desplazamiento [mm]')
plt.legend(loc='best')

#%%


import matplotlib.pyplot as plt
import numpy as np

def encontrar_punto_maximo(variable1, variable2):
    # Combina las dos variables en una matriz
    matriz_combinada = np.array([variable1, variable2])
    
    # Encuentra el índice del máximo en la matriz
    indices_maximos = np.argmax(matriz_combinada, axis=1)
    
    return indices_maximos

# Ejemplo de uso
variable1 = [1, 3, 5, 2, 8]
variable2 = [2, 4, 1, 6, 7]

indices_maximos = encontrar_punto_maximo(variable1, variable2)

# Gráfico de dispersión de las dos variables
plt.scatter(range(len(variable1)), variable1, label='Variable 1')
plt.scatter(range(len(variable2)), variable2, label='Variable 2')

# Resalta los puntos máximos en rojo
for indice_maximo in indices_maximos:
    plt.scatter(indice_maximo, variable1[indice_maximo], color='red', marker='x', s=100)
    plt.scatter(indice_maximo, variable2[indice_maximo], color='red', marker='x', s=100)

# Configuración del gráfico
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Gráfico con Dos Variables')
plt.legend()
plt.show()
