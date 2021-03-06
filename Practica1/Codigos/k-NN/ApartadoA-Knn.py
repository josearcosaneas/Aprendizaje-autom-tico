# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:46:48 2013

@author: jose-ia
"""

print __doc__

import numpy as np 
import pylab as pl 
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
#Fijamos un tamño de 101 por el bucle for utilizado
numero_vecinos=101
# Cargamos los datos de la base de datos 
iris = datasets.load_iris()
iris_X,iris_y = iris.data,iris.target
#partimos los datos de forma similar al ejemplo de la web de la asignatura.
training=0.3
lt=np.round(training*len(iris_X))
# particion aleatoria de los datos para training y test
np.random.seed()
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-lt]]
iris_y_train = iris_y[indices[:-lt]]
iris_X_test  = iris_X[indices[-lt:]]
iris_y_test  = iris_y[indices[-lt:]]
# Creo cuatro listas para almacerna la precision de cada clase y la media de las tres
#  para dibujar las gragicas. El elemento auxiliar es el utilizado para calcular la media.
clase1=[]
clase2=[]
clase3=[]
mediaclases = []
aux = 0
    
for i in range(1,numero_vecinos,2):# numero_vecinos=101  
    neigh= KNeighborsClassifier(i, weights='uniform')
    #Entrenamos y predecimos
    y_ = neigh.fit(iris_X_train, iris_y_train).predict(iris_X_test);    
    #calculamos la precision en a     
    a = precision_score(iris_y_test, y_, average=None)
    #añadimos a cada clase el elemento que le pertenece de a
    clase1.append(a[0])
    clase2.append(a[1])
    clase3.append(a[2])
    aux = float (a[0]+a[1]+a[2])/(3)#aqui calculo la media y la añado en media clases
    mediaclases.append(aux)
#Dibujo la graficas
pl.plot(range(len(range(1,numero_vecinos,2))),clase1, label="clase 1")
pl.plot(range(len(range(1,numero_vecinos,2))),clase2, label="clase 2")
pl.plot(range(len(range(1,numero_vecinos,2))),clase3, label="clase 3")
pl.plot(range(len(range(1,numero_vecinos,2))),mediaclases,linewidth=1.5, label="mediaclases")
pl.legend(loc="best")
# Imprimo los resultaod de las clases y de la media para valorar mejor la precision
print clase1
print clase2
print clase3
print mediaclases
nombre = "Incidencia de k:1-"+str(numero_vecinos/2)
pl.title(nombre)
pl.xlim(0,numero_vecinos/2)
pl.ylim(-0.05,1.05)
pl.savefig(nombre)

pl.show() 
