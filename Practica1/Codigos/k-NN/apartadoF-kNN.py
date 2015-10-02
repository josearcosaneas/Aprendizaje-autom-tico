# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:25:23 2013

@author: jose-ia
"""
"""
Sigo ulizando la funcion de particionar datos de los apartados anteriores
Incluyo una funcion normalizar para normalizar los datos de test y traning.
"""
print __doc__

import numpy as np 
import pylab as pl 
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

numero_vecinos=101
#leo datos
iris = datasets.load_iris()
iris_X,iris_y = iris.data,iris.target
n_muestras= (int) (len(iris_X))
#funcion de particion de datos ultilizada en los dos apartados siguientes.
def particionarDatos(numero_muestras,training, X, y):
    #partimos los datos
    lt=np.round(training*len(iris_X))
    # particion aleatoria de los datos para training y test
    np.random.seed(0)
    indices = np.random.permutation(n_muestras)
    iris_X_train = iris_X[indices[:-lt]]
    iris_y_train = iris_y[indices[:-lt]]
    iris_X_test  = iris_X[indices[-lt:]]
    iris_y_test  = iris_y[indices[-lt:]]
    
    return iris_X_train,iris_y_train,iris_X_test,iris_y_test
# Hago una particion de los datos con 0.3 de training
iris_X_train,iris_y_train,iris_X_test,iris_y_test = particionarDatos(n_muestras, 0.3 ,iris_X, iris_y)
# Funcion para normalizar los datos tiene como argumentos los valores del test y de 
# trainig ademas dle numero de muestras
def Normalizar(training_x, training_y, test_x, test_y, muestras):
    # Uilizo la funcion StandardScaler para normalizar y entrenar los datos.
    # en trainin_norm se guarda el valor normalizado de los datos de entrenamiento
    # y en test_normal el valor normalizado de los datos de test.
    s = StandardScaler(copy=True, with_mean=False, with_std=True).fit(training_x)
    training_norm = s.transform(training_x)
    test_norm = s.transform(test_x)
    # Se devuelven los datos de test y entrenamiento normalizados 
    return training_norm, test_norm
# Creamos dos variables para los datos normacilizados de x e y
training_norm, test_norm = Normalizar(iris_X_train,iris_y_train,iris_X_test,iris_y_test, n_muestras)    

clase1=[]
clase2=[]
clase3=[]
mediaclases=[];
aux = 0
# Realizamos la misma operacion que el apartado 1 pero con los datos normalizados
for i in range(1,numero_vecinos,2):  
    neigh = KNeighborsClassifier(i, weights='uniform')
    y_ = neigh.fit(training_norm, iris_y_train).predict(test_norm);    
    a = precision_score(iris_y_test, y_, average=None)
    clase1.append(a[0])
    clase2.append(a[1])
    clase3.append(a[2])
    aux = float (a[0]+a[1]+a[2])/(3)
    mediaclases.append(aux)
# Dibujo la graficas
pl.plot(range(len(range(1,numero_vecinos,2))),clase1, label="clase 1")
pl.plot(range(len(range(1,numero_vecinos,2))),clase2, label="clase 2")
pl.plot(range(len(range(1,numero_vecinos,2))),clase3, label="clase 3")
pl.plot(range(len(range(1,numero_vecinos,2))),mediaclases, label="mediaclases")
pl.legend(loc="best")
nombre = "Incidencia de k:1-"+str(numero_vecinos)
pl.title(nombre)
pl.xlim(0,numero_vecinos/2)
pl.ylim(-0.05,1.05)
pl.savefig(nombre)

pl.show()