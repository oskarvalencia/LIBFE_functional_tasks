# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:48:15 2023

@author: ovalencia
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from scipy.integrate import trapz
from scipy.signal import butter, filtfilt
get_ipython().magic('reset -sf')

#%%

def rms_(x,Fs,ws,show=True):
    """ Author: O. Valencia, LIBFE
    Root mean square, considering the following variables:
    
    Parameters
    ----------
    EMG_N (x) : float
        signal EMG
    Fs : sampling frequency
    
    ws : window size
    
    show : bool
        optional (default = False)
        True (1) plots data in a matplotlib figure.
        False (0) to not plot.
        
    Returns
    -------
    Vector with RMS values : float
    
    """
    x1= x-np.mean(x) #delete offset
    b, a = butter(2, [(20/(Fs/2)), (500/(Fs/2))], btype = 'bandpass') #Low= 20Hz (b) High=500Hz (a), orden 2.
    x2= filtfilt(b, a, x1)
    W1= int((Fs/1000)*ws)
    AA=[]
    for i in range(0,len(x2),W1):
        r= np.sqrt(np.mean(x2[i:i+(W1+1)]**2))
        AA.append(r)
        Ar= np.array(AA)
        
    max_value= np.max(Ar)
    
    if show:
        fig1(x2,Ar,ws)
    return max_value

def fig1(EMG1,Ar,ws):
    t1= np.linspace(0, len(EMG1), len(Ar))
    plt.figure()
    plt.plot(EMG1,"royalblue", label= "Raw signal")
    plt.plot(t1,Ar, 'ro',lw=2, label= "RMS signal "+ "[" + str(ws)+ " ms]")
    plt.xlabel("Sample number")
    plt.ylabel("sEMG amplitude [mV]")
    plt.axvspan((np.argmax(Ar)*200)-150,(np.argmax(Ar)*200)+150, facecolor='red', alpha=0.6)
    plt.xlim(0,t1[-1])
    plt.legend(loc="best")
    plt.grid()
    plt.show()

#%%

def rms_nor(x,y,Fs,ws,show=True):
    """ Author: O. Valencia, LIBFE
    Calculating the root mean square 
    of both the functional and MVC EMG signals.
    
    Parameters
    ----------
    Functional_EMG (x) : float
        signal EMG
    
    MVC_EMG (y) : float
        signal EMG
        
    Fs : sampling frequency
    
    ws : window size
    
    show : bool
        optional (default = False)
        True (1) plots data in a matplotlib figure.
        False (0) to not plot.
        
    Returns
    -------
    Vector with RMS values normalized : float
    
    """
    x1= x-np.mean(x) #delete offset
    y1= y-np.mean(y) #delete offset
    b, a = butter(2, [(20/(Fs/2)), (500/(Fs/2))], btype = 'bandpass') #bandpass filter
    x2= filtfilt(b, a, x1)
    y2= filtfilt(b, a, y1)
    
    W1= int((Fs/1000)*ws) #calculating the window size
    AA=[]
    for i in range(0,len(x2),W1): #RMS without ovelap
        r= np.sqrt(np.mean(x2[i:i+(W1+1)]**2))
        AA.append(r)
        Ar= np.array(AA)
    
    AA1=[]
    for i in range(0,len(y2),W1): #RMS without ovelap
        r1= np.sqrt(np.mean(y2[i:i+(W1+1)]**2))
        AA1.append(r1)
        Ar1= np.array(AA1)
    max_value= np.max(Ar1)
    #max_value=np.mean(AA1[(np.argmax(AA1)-5):(np.argmax(AA1)+5)]) #with 10 data set
    
    signal_f= (Ar/max_value)*100 #signal normalized
    
    if show:
        fig(x,AA,ws)
    return signal_f

def fig(x,AA,ws):
    t1= np.linspace(0, len(x), len(AA))
    plt.figure()
    plt.plot(x,"royalblue", label= "Raw signal")
    plt.plot(t1,AA, 'r',lw=2, label= "RMS signal "+ "[" + str(ws)+ " ms]")
    plt.xlabel("Sample number")
    plt.ylabel("sEMG amplitude [V]")
    plt.xlim(0,t1[-1])
    plt.legend(loc="best")
    plt.grid()
    plt.show()
#%%

"""Ajustando una señal electromiográfica funcional:

-Laboratorio Integrativo de Biomecánica y Fisiología del Esfuerzo,
Escuela de Kinesiología, Universidad de los Andes, Chile-
-Escuela de Ingeniería Biomédica, Universidad de Valparaíso, Chile-
        --Profesores: Oscar Valencia & Alejandro Weinstein--

Valencia O, de la Fuente C., Guzmán-Venegas R., Salas R., Weinstein A. (2021) 
Propuesta de Flujo de Procesamiento utilizando Python para ajustar la Señal 
Electromiográfica Funcional a la Contracción Voluntaria Máxima.
Revista de Kinesiología, 40(3): 171-175.

"""
def ajusta_emg_func(emg_fun, emg_cvm, fs, f_c, f_orden, nombre,show=True):
    """Ajusta EMG funcional según contracción voluntaria máxima.

    La función utiliza una señal EMG funcional y otra basada en la
    solicitación de una contracción isométrica voluntaria máxima. Ambas señales
    son procesadas considerando su centralización (eliminación de
    "offset"), rectificación y filtrado (pasa bajo con filtfilt).

    Parameters
    ----------
    emg_fun : array_like
        EMG funcional del músculo a evaluar
    emg_cvm : array_like
        EMG vinculada a la contracción voluntaria máxima del mismo músculo
    fs : float
       Frecuencia de muestreo, en hertz, de la señal EMG. Debe ser la misma
       para ambas señales.
    f_c : float
        Frecuencia de corte, en hertz, del filtro pasa-bajos.
    f_orden : int
        Orden del filtro pasa bajos
    nombre : string
        nombre del músculo entre ""

    Return
    ------
    emg_fun_norm : array_like
        EMG funcional filtrada y  normalizada
    emg_fun_env_f : array_like
        Envolvente de EMG funcional filtrada
    emg_cvm_envf_ : array_like
        Envolvente de EMG CVM filtrada
    """
    #centralizando y rectificando las señales EMG
    emg_fun_env = abs(emg_fun - np.mean(emg_fun))
    emg_cvm_env = abs(emg_cvm - np.mean(emg_cvm))

    # Filtrado pasa-bajo de las señales
    b, a = butter(int(f_orden), (int(f_c)/(fs/2)), btype = 'low')
    emg_fun_env_f = filtfilt(b, a, emg_fun_env)
    emg_cvm_env_f = filtfilt(b, a, emg_cvm_env)

    #calculando el valor máximo de emg_cvm y ajustando la señal EMG funcional
    emg_cvm_I = np.max(emg_cvm_env_f)
    #emg_cvm_I=np.mean(emg_cvm_env_f[(np.argmax(emg_cvm_env_f)-5):(np.argmax(emg_cvm_env_f)+5)]) #con 10 datos
    emg_fun_norm = (emg_fun_env_f / emg_cvm_I) * 100
    
    if show:
        plot_emgs(emg_fun, emg_fun_env, emg_fun_norm, emg_cvm, emg_cvm_env,
                      fs, f_c, f_orden, nombre)
    return emg_fun_norm

def plot_emgs(emg_fun, emg_fun_env, emg_fun_norm, emg_cvm, emg_cvm_env,
              fs, f_c, f_orden,
              nombre):
    """Grafica señales de EMG funcional y CVM.

    Parameters
    ----------
    emg_fun : array_like
        EMG funcional.
    emg_fun_env : array_like
        Envolvente del EMG funcional.
    emg_fun_norm : array_like
        EMG funcional normalizada según CVM.
    emg_cvm : array_like
        EMG contracción voluntaria máxima.
    fs : float
        Frecuencia de muestreo, en hertz.
    f_c : float
        Frecuencia de corte del filtro pasa-bajo, en hertz.
    f_orden : int
        Orden del filtro.
    nombre : str
        Nombre del músculo.
    """

    # Vectores de tiempo
    t1 = np.arange(0, len(emg_fun) / fs, 1 / fs)
    t2 = np.arange(0, len(emg_cvm) / fs, 1 / fs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize = (8, 7))

    ax1.plot(t1, emg_fun, 'b', label='Señal bruta')
    ax1.set_title(f'Músculo: {nombre}; filtro aplicado: f_c={f_c} [Hz] y '
                  f'orden {f_orden}')

    ax1.plot(t1, emg_fun_env, 'r', lw=2, label='Señal filtrada')
    ax1.set_ylabel(f'{nombre} Funcional\nAmplitud [V]',fontsize=9)
    ax1.set_ylim(emg_fun.min() - (emg_fun.min()*0.1), emg_fun.max() + (emg_fun.max()*0.1))
    ax1.set_xlim(0, t1.max())
    ax1.grid()
    ax1.legend(loc='upper center', fontsize='x-small', borderpad=None)

    ax2.plot(t2, emg_cvm, 'b', label='Señal bruta')
    ax2.plot(t2, emg_cvm_env, 'r', lw=2, label='Señal filtrada')
    ax2.set_ylabel(f'{nombre} CVM\nAmplitud [V]',fontsize=9)
    ax2.axvline((np.argmax(emg_cvm_env) / fs), color='maroon')
    ax2.text(0.85, 0.95 ,f'Max = {emg_cvm_env.max():.2f}',
             transform=ax2.transAxes, ha="left", va="top")
    ax2.set_ylim(emg_cvm.min() - (emg_cvm.min()*0.1), emg_cvm.max() + (emg_cvm.max()*0.1))
    ax2.set_xlim(0, t2.max())
    ax2.grid()
    ax2.legend(loc='upper center', fontsize='x-small', borderpad=None)

    ax3.plot(t1, emg_fun_norm, 'g',label='Señal ajustada según CVM')
    ax3.set_ylim(emg_fun_norm.min(), emg_fun_norm.max() + 2)
    ax3.set_xlim(0, t1.max())
    ax3.set_xlabel('Tiempo [s]', fontsize=9)
    ax3.set_ylabel('% EMG CVM')
    ax3.grid()
    ax3.legend(loc='upper center', fontsize='x-small', borderpad=None)

    plt.tight_layout(h_pad=.1)

#%%

def coactivation_index(emg_A, emg_B, label_musc_A, label_musc_B, fs, show= True):
    """Calcula el índice de coactivación.

    La función calcula la coactivación entre músculos antagonistas [1,2]. El
    cálculo se realiza a partir de dos señales de EMG previamente procesadas y
    normalizadas (en base al valor RMS o promedio de la señal rectificada).

    Ambas señales deben ser adquiridas con la misma frecuencia de
    muestreo y tener el mismo largo.
    
    Valencia O., Varas M., Moya C., Besomi M., Weinstein A., Guzmán-Venegas R. (2022) 
    Coactivación muscular. Una propuesta para calcular su índice utilizando Python.
    41(2): 120-123

    Parameters
    ----------
    emg_a : array_like
        EMG funcional de uno de los músculos antagonistas (A).
    emg_b : array_like
        EMG funcional del otro músculo antagonista (B).
    label_musc_A, label_musc_B : string
            Etiqueta a utilizar para cada músculo.
    fs : float
        Frecuencia de muestreo de la señal EMG, en hertz. Debe ser la misma
        para ambas señales.

    Return
    -------
    Coeficiente de activación (porcentaje)

    Referencia
    ----------
    .. [1] Falconer K, Winter DA. Quantitative assessment of cocontraction at
    the ankle joint in walking.  Electromyogr Clin Neurophysiol 1985; 25:
    135-149.
    .. [2] Guilleron, C., Maktouf, W., Beaune, B., Henni, S., Abraham, P., &
    Durand, S. (2021). Coactivation pattern in leg muscles during treadmill
    walking in patients suffering from intermittent claudication. Gait &
    Posture, 84, 245-253.
    """
    I_antagonist = trapz(np.minimum(emg_A, emg_B)) / fs
    I_total = trapz(emg_A + emg_B) / fs
    ci= 2 * I_antagonist / I_total * 100
    
    if show:
        plot_coactivacion(emg_A, emg_B, label_musc_A, label_musc_B, fs, ci)
    return ci


def plot_coactivacion(emg_A, emg_B, label_musc_A, label_musc_B, fs, ci):
    """Gráfica la coactivación entre dos músculos.

    La función grafica dos señales de EMG. Se muestra el área correspondiente a
    la región de coactivación.

    Parameters
    ----------
    ax : Matplotlib axes
        Ejes sobre los cuales se grafica la coactivación.
    emg_A, emg_B : array_like
        Vectores con la señal de EMG de los dos músculos a graficar.
    label_musc_A, label_musc_B : string
        Etiqueta a utilizar para cada músculo.
    ci : float
        Valor del índice de coactivación.
    """
    t = np.arange(0, len(emg_A) / fs, 1 / fs)
    emg_min = np.minimum(emg_A, emg_B).min()
    emg_max = np.maximum(emg_A, emg_B).max()
    plt.plot(t, emg_A, lw=2, label=label_musc_A)
    plt.plot(t, emg_B, lw=2, label=label_musc_B)
    plt.plot([], [], ' ', label=f'IC = {ci:0.2f}%')
    plt.fill_between(t, np.minimum(emg_A, emg_B), alpha=0.3, color='k')
    plt.ylim(0.8 * emg_min, 1.2 * emg_max)
    plt.xlim(0, t[-1])
    plt.xlabel('Tiempo [s]', fontsize=12)
    plt.ylabel('Amplitud EMG [%CVM]', fontsize=12)
    plt.legend()

#%%

def select(fl,side,F,FM):
    """Función para seleccionar eventos.
    
    Autor: O. Valencia, LIBFE, 2023
    
    Parameters
    ----------
    fl: matriz con datos (dataframe).
    
    side: lado izquierdo (¨Right¨) o derecho ("Left").
    
    F: hay tres alternativas, "ALL": considera todos los eventos (foot strike y foot off),"FS": solo seleciona los foot strike, "FO": solo selecciona los foot off.
        
    FM: corresponde a la frecuencia de captura de las cámaras
    
    Returns
    -------
    Dos componentes: matriz con los datos seleccionados y un vector con los valores ajustados a la frecuencia de captura
    """
    fl2=fl[(fl.Context ==str(side))]
    if F == 'ALL':
        L=fl2
    elif F == 'FS':
        L=fl2[(fl2.Name=='Foot Strike')]
    elif F == 'FO':
        L=fl2[(fl2.Name=='Foot Off')]
    result= L.iloc[:,3].values* FM
    return L, result
