# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:41:34 2023

@author: ovalencia, LIBFE, Escuela de Kinesiologia, 
Universidad de los Andes.
"""

import numpy as np # https://numpy.org/
import pandas as pd # https://pandas.pydata.org/
import matplotlib.pyplot as plt # https://matplotlib.org/

#%%

'''ejemplo 1'''

print("int: ", int(10), ", float: ", float(4.6),
      ", str: ", str("buen día"), ", bool: ", bool(True))

'''nota: La función "print()" imprime las variables que se establecen en
su interior.''' 

#%%

'''ejemplo  2'''

A=2
a=3

print('Result= ',A>=a)
#%%
'''Ejemplo 3: Solicitar que el usuario ingrese dos números y entregar
los resultados de los diferentes operadores

Operadores Aritméticos

| Símbolo | Significado       |
|---------|-------------------|
|   +     | Suma              |
|   -     | Resta             |
|   -     | Negativo          |
|   *     | Multiplicación    |
|   **    | Exponente         |
|   /     | División          |
|   //    | División entera   |
|   %     | Módulo            |
'''

n1 = float(input("Ingrese un número: ")) # Solicita el primer número
n2 = float(input("Ingrese un número: ")) # Solicita el segundo número


print("Suma =" , n1 + n2)
print("Resta = ", n1 - n2)
print("Multiplicación = ", n1 * n2)
print("División = ", n1 / n2)
print("Exponente = ", n1 ** n2)
print("División entera = ", n1 // n2)
print("Módulo = ", n1 % n2)

'''nota: La función "input()" solicita una información previamente 
estipulada. Esta puede ser de cualquier naturaleza (considerando 
el tipo de variable)'''
#%%

'''Ejemplo 4: Solicitar al usuario que ingrese valores lógicos aplicar 
los operadores lógicos.'''

p = bool(int(input("p (ingrese 0 o 1)= ")))
q = bool(int(input("q (ingrese 0 o 1)= ")))

print(p == q)
print(not p == q)
print(1 < p > 1)

#%%
'''Ejemplo 5: Precedencia de los operadores. Cuando en una misma línea de código
existen más de un tipo de operador, también existe cierto nivel de 
jerarquía. La prioridad en que se aplican estos operadores, considerando 
la prioridad se puede resumir en la sigueinte tabla.

| Prioridad | Operador    | Descripción                        |
|-----------|-------------|------------------------------------|
| 1         | **          | Exponente                          |
| 2         | + -         | Símbolo unario positivo o negativo |
| 3         | * / % //    | Multiplicación, división, módulo   |
| 4         | + -         | Suma y resta                       |
| 5         | &           | operador lógico bitwise and        |
| 6         |  ^          | operador lógico bitwise or, xor    |
| 7         | <= < > >=   | operadores de comparación          |
| 8         | <> == ! =   | operadores de igualdad.            |
| 9         | = %= /= \*=  | operadores de asignación          |
| 9         | //= -= += \*\*= | operadores de asignación       |
| 10        | is is not   | operadores de identidad            |
| 11        | in in not   | operadores de membresía            |
| 12        | not or and  | operadores de lógicos              |
'''
e= 3 * 5 / 2 - 4 ** 0.0007
print('e= ', e)
o=5 - (-3 / 2 ** 2)
print('o= ', o)
x = e <= o
print('x= ', x)

#%%

'''Ejemplo 6: Utilizando la función help()'''

help(plt) #información 

#%%
'''Ejemplo 7: Uso de for

- El operador *for* funciona similar a while, sin embargo,
está optimizado para repetir funciones un determinado 
número de veces (también llamado *loop*). Para poder realizar esto,
debemos acompañar *for* con *in range(int):*. 
El int dentro del paréntesis permite indicar la cantidad de veces 
que se repetirá la función for. Algo importante a notar, es que la 
función se repite un total del *int* veces, pero al comenzar desde 
el valor 0, este solo llegará al valor int – 1. Tomando el ejemplo 
anterior, podríamos realizar el mismo cálculo, pero un total de 4 
iteraciones.'''

print("f° c°")
for temp in range(4):
    print(temp, "", int((temp - 32)))

#%%

'''Ejemplo 8: Una función en lenguaje de programación es una pieza de código 
que desarrolla una tarea específica (muchas veces clarifica y reduce
un código). Esta se puede crear en Python escribiendo *def*. Las 
funciones se utilizan considerando los siguientes puntos:

- Brindar la oportunidad de nombrar un grupo de declaraciones 
(flujo de procesamiento y variables implícitas), lo que hace 
que su programa sea más fácil de leer y depurar.

- Hacer que un programa sea más pequeño al eliminar el código 
repetitivo. Posteriormente, si realiza un cambio, solo tiene 
que hacerlo en un lugar (es decir, donde escribió la función).

Toda función contiene un nombre específico luego de escribir
*def* y parámetros establecidos, los cuales deben escribirse 
al interior de un paréntesis. Es importante destacar que cada 
función retorna un elemento, el cual debe ser delimitado posterior
a la palabra *return*. Esta última debe ser parte de la *identación*
(es decir, un espacio supeditado luego de la segunda línea que contiene
la función) necesaria al construir una función.'''

# Desarrollando y aplicando una función
def imc(estatura, masa):
    resultado_imc = round(masa / estatura**2,3) # "round" ajusta a un cierto n de decimales 
    return print('IMC= ',resultado_imc, 'kg/m2')

# llamando la función
imc(1.68,67)

import math

# función

def h(a,b):
    y=math.sqrt() #tarea
    return print('Hipotenusa= ',round(y,2))

# llamando la función
h(100,300) #cambiar valores

#%%

'''Ejemplo : Cargar datos en formato .csv

- La sigla CSV proviene del inglés: *Comma Separated Values*
 (valores separados por comas). Estos archivos son identificados
 por Excel y otros programas, y se caracterizan por poseer una serie 
 de datos en distintas filas y columnas. Normalmente las columnas 
 corresponden a distintos tipos de datos (variables), mientras que 
 las filas son las distintas mediciones de cada uno de estos tipos 
 de datos (usualmente en distintas personas o en distintos periodos 
 de tiempo).


- Para trabajar los archivos .csv en Python hay dos partes importantes:
  importar el archivo .csv y luego trabajar con sus valores. Para importar
  el archivo .csv se suele utilizar la librería pandas, la cual debe importar
  de la siguiente forma: *import pandas as pd*
  
- Al importar estas librerías, Python no entrega ningún output, pero es 
  necesario hacerlo para utilizar las funciones de la librería. Se le 
  indicó *as pd*, ya que cada vez que utilicemos alguna función de dicha
  librería desde aquí en adelante, se utilizará el comando pd. Si bien 
  estas letras son modificables, se recomienda mantener el *pd*, ya que 
  es universalmente utilizado.


- Tras tener la librería importada, importaremos el archivo .csv como
 una tabla de datos de dos dimensiones, lo cual también es llamado 
 matriz o *array like*. Una matriz en Python es un arreglo de datos 
 en dos dimensiones, en otras palabras, es una lista que posee listas 
 dentro de ella, con N cantidad de filas y M cantidad de columnas. En 
 este caso se llamará a la ‘matriz’ del archivo .csv de la siguiente 
 forma: *matriz = pd.read_csv(‘ubicación y nombre del
 archivo’, sep=‘,’)*. Otra forma de importar datos en Spyder es cambiar
 la dirección del *path o directory*.

**Importante:** la ruta de acceso a los archivos guardados en tu 
computador personal depende del tipo de sistema operativo, Windows 
o Mac. En el primer caso, se requiere abrir la dirección exacta (carpeta)
donde se encuentran lo archivos .csv; luego solo debes copiar la ruta
y pegarla entre las cremillas. En el segundo caso, se debe presionar 
*con dos dedos sobre el archivo* y seleccionar **obtener información*.
Posteriormente, es necesario ver las propiedades del archivo, copiar la
ruta acceso (*copiar ubicación*), y agragar el *nombre del archivo*. Por
último, en jupyter notebook debe escribir entre las cremillas *Users/...*,
sin embargo, si al copiar tu ruta esto ya aparece, no es necesario \
repetirlo.'''

import pandas as pd
datos = pd.read_csv("https://raw.githubusercontent.com/oskarvalencia/M4_DBAM/main/CINE_GAIT.csv",
                    sep=';')

#%%

'''Ejemplo 10: Creando figuras con matplotlib.pyplot

- Graficar señales es una de las funciones más importantes al momento
 observar el comportamiento de los datos que posteriormente deben ser
 analizados. Formas para hacerlo hay muchas, pero la librería 
 *matplotlib.pyplot* es una de las más utilizadas, dada la facilidad 
 de su uso y versatilidad en las funciones que contiene. Para utilizar 
 esta librería lo primero que debes realizar es importar (llamar) la 
 siguiente línea de códigos: *import matplotlib.pyplot as plt*'''
 
import matplotlib.pyplot as plt

plt.figure(figsize = (12, 6))
plt.plot(datos.LHX, color='green', marker='.', linestyle='-', 
         linewidth=4, label='plano sagital')
plt.xlabel('n datos', fontsize=12)
plt.ylabel('F/E cadera [°]', fontsize=12)
plt.xlim(0,len(datos.LHX))
plt.legend(loc='best', fontsize=12)
plt.grid()

#%%

'''Anexos 
- https://matplotlib.org/stable/gallery/animation/double_pendulum.html#sphx-glr-gallery-animation-double-pendulum-py
- https://matplotlib.org/stable/gallery/mplot3d/surface3d.html#sphx-glr-gallery-mplot3d-surface3d-py
'''

from collections import deque

from numpy import cos, sin

import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 2.5  # how many seconds to simulate
history_len = 500  # how many trajectory points to display


def derivs(t, state):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.01
t = np.arange(0, t_stop, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate the ODE using Euler's method
y = np.empty((len(t), 4))
y[0] = state
for i in range(1, len(t)):
    y[i] = y[i - 1] + derivs(t[i - 1], y[i - 1]) * dt

# A more accurate estimate could be obtained e.g. using scipy:
#
#   y = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()


#%%%

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#%%

'''Bibliografía

- https://numpy.org/
- https://matplotlib.org/stable/index.html
- https://pandas.pydata.org/
- https://greenteapress.com/thinkpython2/thinkpython2.pdf'''