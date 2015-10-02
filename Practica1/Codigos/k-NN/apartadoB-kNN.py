# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:27:04 2013

@author: jose-ia
"""

print __doc__

import numpy as np 
import pylab as pl 
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
#Se inicializa el numero de vecinos a 101 porque el bucle for itera de dos en dos desde 1.
numero_vecinos=101
#leo datos
iris = datasets.load_iris()
iris_X,iris_y = iris.data,iris.target
#partimos los datos de igual forma que el documento de ayuda de la pagina web de la asignatura.
training=0.3
lt=np.round(training*len(iris_X))
# particion aleatoria de los datos para training y test se hace de igual forma 
# que en el documento de ayuda de la pagina web de la asignatura.
np.random.seed()
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-lt]]
iris_y_train = iris_y[indices[:-lt]]
iris_X_test  = iris_X[indices[-lt:]]
iris_y_test  = iris_y[indices[-lt:]]
# Se crean la clases necesarias para almacenar los datos de test y training para las clases.
clase1_test = []
clase2_test = []
clase3_test = []
clase1_training = []
clase2_training = []
clase3_training = []

for i in  range(1,numero_vecinos,2):
    neigh = KNeighborsClassifier(i, weights='uniform')
    y_ = neigh.fit(iris_X_train, iris_y_train).predict(iris_X_test);    
    yy_ = neigh.fit(iris_X_train, iris_y_train).predict(iris_X_train)    
    #Calculamos el error como (1-la precicion)
    a = 1-precision_score(iris_y_test, y_, average=None)
    b = 1-precision_score(iris_y_train, yy_, average=None)
    # Lo añadimos a cada lista los elementos de la posicino 0 en la clase1 , posicion 1 clase 2,
    #Posicon 2 clase3    
    clase1_test.append(a[0])
    clase2_test.append(a[1])
    clase3_test.append(a[2])
    clase1_training.append(b[0])
    clase2_training.append(b[1])
    clase3_training.append(b[2])

# Dibujamos las graficas para el test
pl.plot(range(1,numero_vecinos,2),clase1_test, label="clase 1")
pl.plot(range(1,numero_vecinos,2),clase2_test, label="clase 2")
pl.plot(range(1,numero_vecinos,2),clase3_test, label="clase 3")
pl.legend(loc="best")
pl.xlim(0,numero_vecinos/2)
pl.ylim(-0.5,1.05)
nombre = "Error de ajuste para datos de test de k:1-"+str(numero_vecinos/2)
pl.title(nombre)    
pl.savefig(nombre)

pl.show() 

# Dibujamos la graficas para el training
pl.plot(range(1,numero_vecinos,2),clase1_training, label="clase 1")
pl.plot(range(1,numero_vecinos,2),clase2_training, label="clase 2")
pl.plot(range(1,numero_vecinos,2),clase3_training, label="clase 3")
pl.legend(loc="best")
pl.xlim(0,numero_vecinos/2)
pl.ylim(-0.5,1.05)
nombre = "Error de ajuste para datos de entrenamiento de k:1-"+str(numero_vecinos/2)
pl.title(nombre)
pl.savefig(nombre)

pl.show() 

