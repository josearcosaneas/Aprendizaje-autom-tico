# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:39:31 2013

@author: jose-ia
"""

print __doc__

import numpy as np 
import pylab as pl 
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
#from sklearn.preprocessing import StandardScaler

numero_vecinos=101
#leo datos
iris = datasets.load_iris()
iris_X,iris_y = iris.data,iris.target
#partimos los datos
training=0.3
lt=np.round(training*len(iris_X))
# particion aleatoria de los datos para training y test
np.random.seed()
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-lt]]
iris_y_train = iris_y[indices[:-lt]]
iris_X_test  = iris_X[indices[-lt:]]
iris_y_test  = iris_y[indices[-lt:]]
# Creo las clases necesarias para almacenar los valores de test y de training
# de las diferentes clases.
clase1_test = []
clase2_test = []
clase3_test = []
clase1_train = []
clase2_train = []
clase3_train = []

for i in range(1,numero_vecinos,2):  
     neigh = KNeighborsClassifier(i, weights='distance')
     # Este este caso se utiliza la distancia inversa.     
     y_ = neigh.fit(iris_X_train, iris_y_train).predict(iris_X_test);    
     yy_ = neigh.fit(iris_X_train, iris_y_train).predict(iris_X_train)    
     #calculo el error como en el ejrcicio anterior (1-precision)
     #calculo en a la precion para test y en  b la precion para el training      
     a = 1-precision_score(iris_y_test, y_, average=None)
     b = 1-precision_score(iris_y_train, yy_, average=None)
     #añado a casa clase el valor que le corresponde de a y b
     clase1_test.append(a[0])
     clase2_test.append(a[1])
     clase3_test.append(a[2])
     clase1_train.append(b[0])
     clase2_train.append(b[1])
     clase3_train.append(b[2])
#Dibujo la graficas para el test
pl.plot(range(1,numero_vecinos,2),clase1_test, label="clase 1")
pl.plot(range(1,numero_vecinos,2),clase2_test, label="clase 2")
pl.plot(range(1,numero_vecinos,2),clase3_test, label="clase 3")
pl.legend(loc="best")
pl.xlim(0,numero_vecinos/2)
pl.ylim(-0.5,1.05)
nombre = "Valorando ponderacion: Error de ajuste datos test"
pl.title(nombre)
title1 = nombre 
pl.savefig(title1)
pl.show() 
#Dibujo la graficas par ael training
pl.plot(range(1,numero_vecinos,2),clase1_train, label="clase1")
pl.plot(range(1,numero_vecinos,2),clase2_train, label="clase2")
pl.plot(range(1,numero_vecinos,2),clase3_train, label="clase3")
pl.legend(loc="best")    
pl.xlim(0,numero_vecinos/2)
pl.ylim(-0.5,1.05)
nombre = "Valorando ponderacion: Error de ajuste datos entrenamiento"
pl.title(nombre)
title2 = nombre 
pl.savefig(title2)

pl.show() 

