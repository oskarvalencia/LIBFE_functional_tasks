# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:49:00 2024

@author: ovale
"""

import pandas as pd
import seaborn as sns #es necesario instalar librería: pip install seaborn
import pingouin as pg #es necesario instalar la libreriia: pip install pingouin
import matplotlib.pyplot as plt


#%%
#Cargando base de datos
'''Palma FH, Kozma PJ, Soler Chamorro C. 2023. Valores de referencia
 de los cambios de presiones plantares entre condición estática a 
 dinámica en personas sanas [en proceso de publicación]'''

base = pd.read_csv('Base_DBAM_.csv', sep=';')

#%%
#análisis descriptivo usando librería pandas

des_datos= base.describe() #descripción de datos usando dos decimales

#%%
#gráfica usando...
plt.figure()
plt.boxplot(base.estatura)
plt.xlabel('leyenda X')
plt.ylabel('leyenda Y')

#%%
plt.figure()
sns.swarmplot(y=base.estatura, color=".10", size=6, linestyle=':')
sns.boxplot(y=base.estatura, fliersize=2, color='gray',palette='Blues_d', linewidth=2)
#%%
#agregar boxplot con dos
plt.figure()
sns.boxplot(data= base, x="tipo_pie",y="IECP",hue="sexo")
plt.figure()
sns.boxplot(data= base, x="tipo_pie",y="IECP")

#%%
#gráfica usando...

plt.figure()
plt.scatter(base.masa, base.estatura)
plt.xlabel('leyenda X')
plt.ylabel('leyenda Y')

#%%

#Shapiro-Wilk usando libreria pingouin
prueba_normalidad= pg.normality(data=base)

#%%
#histograma de variable l_eeii

plt.hist(base.l_eeii, density=True, bins = 10)
plt.title('Histograma de variable -l_eeii-')
plt.ylabel('Frecuencia')
plt.grid(color='k', linestyle='-', linewidth=.1)
plt.show()

#%%
#chi2 y prueba exacta de fisher usando libreria pg

chi2= pg.chi2_independence(base, x= 'pisada', y= 'tipo_pie')

#%%
#t-student o wilcoxon para datos pareados usando librería pg

t_test_pareado= pg.ttest(x=base.area_e, y=base.area_d, paired= True, alternative='two-sided')

wilcoxon_pareado= pg.wilcoxon(x=base.area_e, y=base.area_d, alternative='two-sided')

#%%
#t-student o Mann Whitney para datos independientes usando librería pg

hombres =base[(base.sexo == 0)] #sacando todos los datos para hombres
mujeres =base[(base.sexo == 1)] #sacando todos los datos para mujeres

t_test_nopareado= pg.ttest(x=hombres.fpi_total, y=mujeres.fpi_total, paired= False, alternative='two-sided')

mann_whitney_pareado= pg.mwu(x=hombres.fpi_total, y=mujeres.fpi_total, alternative='two-sided')
#%%
#ejemplo de gráfica para una correlación
sns.regplot(x=base.masa, y=base.rpm_d)

#%%
#pruebas de correlación
#Pearson y Spearman usando librería pg

pearson= pg.corr(base.masa, base.rpm_d, method="pearson")

spearman= pg.corr(base.masa, base.rpm_d, method="spearman")









