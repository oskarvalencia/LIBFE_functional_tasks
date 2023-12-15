# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:09:02 2023

@author: ovale
"""
import spm1d #instalar previamente usando pip install spm1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
def select_events(data, side, event_type, capture_frequency):
    """
    Function to select events.

    Parameters
    ----------
    data : pandas.DataFrame
        Matrix of data.
    side : str
        Left side ('Right') or right side ('Left').
    event_type : str
        Type of event. Can be 'ALL' for all events,
        'FS' for foot strike, 'FO' for foot off.
    capture_frequency : float
        Capture frequency of the cameras.

    Returns
    -------
    Tuple[pandas.DataFrame, numpy.ndarray]
        Matrix with the selected data and a vector with values
        adjusted to the capture frequency.
    """
    # Filter by side
    filtered_data = data[data.Context == side]

    # Filter by event type
    if event_type == 'ALL':
        selected_data = filtered_data
    elif event_type == 'FS':
        selected_data = filtered_data[filtered_data.Name == 'Foot Strike']
    elif event_type == 'FO':
        selected_data = filtered_data[filtered_data.Name == 'Foot Off']
    else:
        raise ValueError("Event type must be 'ALL', 'FS', or 'FO'.")

    # Calculate adjusted values
    adjusted_values = selected_data.iloc[:, 3].values * capture_frequency

    return selected_data, adjusted_values


normal = pd.read_csv('Curvas_promedio_Fukuchi.csv', sep=';')

#%%

cine = pd.read_csv('Cine_postfatiga.csv', skiprows=4) #cambiar Cine_postfatiga
cinef_= cine.iloc[:,2::].values

ev = pd.read_csv('Ev_postfatiga.csv', skiprows=2) # cambiar Ev_postfatiga

sel_datosR, adj_valoresR= select_events(ev, 'Right', 'FS', 200)
sel_datosL, adj_valoresL= select_events(ev, 'Left', 'FS', 200)

#%%

datos_R=[] 
for o in range(0,24):
    eje= cinef_[:,o]
    for i in range(0,10): 
        cut= eje[int(adj_valoresR[i]):int(adj_valoresR[i+1])]
        datos_R.append(cut)

ciclosR= spm1d.util.interp(datos_R, Q=101)

datos_L=[] 
for r in range(0,24):
    eje= cinef_[:,r]
    for t in range(0,10): 
        cut1= eje[int(adj_valoresL[t]):int(adj_valoresL[t+1])]
        datos_L.append(cut1)

ciclosL= spm1d.util.interp(datos_L, Q=101)

def ave(GRpx):
    F=[]
    for y in range(0,101):
        h= np.mean(GRpx[:,y])
        F.append(h)
    return F

izquierda= ciclosL[30:120,:]
derecha= ciclosR[150:240,:]

#%%
f= derecha # cambiar por derecha o izquierda

t1= np.linspace(0,101,101)

fig, axs = plt.subplots(3, 3,figsize=(6, 4), sharex='all', sharey='row',
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, dpi=200)
(ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
fig.suptitle('')

ax1.plot(t1,normal.lhipangz,'r',markersize=0.8, linewidth=1)
ax1.plot(t1,ave(f[0:10,:]),'r--',markersize=0.8, linewidth=1)
ax1.set_title('Cadera',fontsize=7)
#ax1.set_xlim(0, len(GRpx[u,:]))
ax1.fill_between(t1,normal.lhipangz + normal.lhipangz_sd,normal.lhipangz - normal.lhipangz_sd, color='silver')

ax2.plot(t1,normal.lkneeangz,'b',markersize=0.8, linewidth=1)
ax2.plot(t1,ave(f[30:40,:]),'b--',markersize=0.8, linewidth=1)
ax2.set_title('Rodilla',fontsize=7)
ax2.fill_between(t1,normal.lkneeangz + normal.lkneeangz_sd,normal.lkneeangz - normal.lkneeangz_sd, color='silver')

ax3.plot(t1,normal.lankleangz,'g',markersize=0.8, linewidth=1)
ax3.plot(t1,ave(f[60:70,:]),'g--',markersize=0.8, linewidth=1)
ax3.set_title('Tobillo',fontsize=7)
ax3.fill_between(t1,normal.lankleangz + normal.lankleangz_sd,normal.lankleangz- normal.lankleangz_sd, color='silver')

ax4.plot(t1,normal.lhipangx,'r',markersize=0.8, linewidth=1)
ax4.plot(t1,ave(f[10:20,:]),'r--',markersize=0.8, linewidth=1)
ax4.fill_between(t1,normal.lhipangx + normal.lhipangx_sd,normal.lhipangx - normal.lhipangx_sd, color='silver')


ax5.plot(t1,normal.lkneeangx,'b',markersize=0.8, linewidth=1)
ax5.plot(t1,ave(f[40:50,:]),'b--',markersize=0.8, linewidth=1)
ax5.fill_between(t1,normal.lkneeangx + normal.lkneeangx_sd,normal.lkneeangx - normal.lkneeangx_sd, color='silver')

#ax5.set_xlim(0,47)
ax6.plot(t1,normal.lankleangx,'g',markersize=0.8, linewidth=1)
ax6.plot(t1,ave(f[70:80,:]),'g--',markersize=0.8, linewidth=1)
ax6.fill_between(t1,normal.lankleangx + normal.lankleangx_sd,normal.lankleangx - normal.lankleangx_sd, color='silver')

#ax6.set_xlim(0,47)
ax7.plot(t1,normal.lhipangy,'r',markersize=0.8, linewidth=1)
ax7.plot(t1,ave(f[20:30,:]),'r--',markersize=0.8, linewidth=1)
ax7.fill_between(t1,normal.lhipangy + normal.lhipangy_sd,normal.lhipangy - normal.lhipangy_sd, color='silver')

#ax7.set_xlim(0,47)
ax8.plot(t1,normal.lkneeangy,'b',markersize=0.8, linewidth=1)
ax8.plot(t1,ave(f[50:60,:]),'b--',markersize=0.8, linewidth=1)
ax8.fill_between(t1,normal.lkneeangy + normal.lkneeangy_sd,normal.lkneeangy - normal.lkneeangy_sd, color='silver')
#ax8.set_xlim(0,47)

ax9.plot(t1,normal.lankleangy,'g',markersize=0.8, linewidth=1)
ax9.plot(t1,ave(f[80:90,:]),'g--',markersize=0.8, linewidth=1)
ax9.fill_between(t1,normal.lankleangy + normal.lankleangy_sd,normal.lankleangy - normal.lankleangy_sd, color='silver')




ax1.set_ylabel('Plano sagital\n E / F [°]',fontsize=7)
ax4.set_ylabel('Plano frontal\n ABD / ADD [°]',fontsize=7)
ax7.set_ylabel('Plano transversal\n RE / RI [°]',fontsize=7)

ax7.set_xlabel('Ciclo de carrera [%]',fontsize=7)
ax8.set_xlabel('Ciclo de carrera [%]',fontsize=7)
ax9.set_xlabel('Ciclo de carrera [%]',fontsize=7)


ax1.tick_params(axis='y',width=1 , colors='k', labelsize='xx-small',length=2, pad=2)
ax4.tick_params(axis='y',width=1 , colors='k', labelsize='xx-small',length=2, pad=2)
ax7.tick_params(axis='both',width=1 , colors='k', labelsize='xx-small',length=2, pad=2)

ax7.tick_params(axis='x',width=1 , colors='k', labelsize='xx-small',length=2, pad=2)
ax8.tick_params(axis='x',width=1 , colors='k', labelsize='xx-small',length=2, pad=2)
ax9.tick_params(axis='x',width=1 , colors='k', labelsize='xx-small',length=2, pad=2)