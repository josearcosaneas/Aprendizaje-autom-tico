# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:22:45 2013

@author: jose-ia
"""

print __doc__

import numpy as np 
import pylab as pl 
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
#from sklearn.preprocessing import StandardScaler
x=1
#numero_vecinos=52
#leo datos
iris = datasets.load_iris()
iris_X,iris_y = iris.data,iris.target
n_muestras= (int) (len(iris_X))
def particionarDatos(numero_muestras,training, X, y):
    #partimos los datos
    lt=np.round(training*len(iris_X))
    # particion aleatoria de los datos para training y test
    np.random.seed()
    indices = np.random.permutation(n_muestras)
    iris_X_train = iris_X[indices[:-lt]]
    iris_y_train = iris_y[indices[:-lt]]
    iris_X_test  = iris_X[indices[-lt:]]
    iris_y_test  = iris_y[indices[-lt:]]
    
    return iris_X_train,iris_y_train,iris_X_test,iris_y_test

n_muestras=len(iris_X)  
sum1=0
sum2=0
sum3=0
clase1 = []
clase2 = []
clase3 = []
media1 = []
media2 = []
media3 = []
training = []
#Creo las particiones de test y training necesarias para este apartado desde 
#50 a 100% de 10 en 10 en caso de ser 90 se coje como valor (n_muestras-1)*100/n_muestras
for i in range(50,100,10):
    if(i==90):
        i= (float) ((n_muestras-1)*100)/n_muestras
    training.append(i) 
    
    for j in range(0,10):
        iris_X_train,iris_y_train,iris_X_test,iris_y_test = particionarDatos(n_muestras, i, iris_X,iris_y)
        # En esta ocacion utiliso la distancia p=1 para especificar en la funcion 
        # que utiliso la distancia de Manhatan        
        neigh = KNeighborsClassifier(1, weights='uniform',p=1)
        y_ = neigh.fit(iris_X_train, iris_y_train).predict(iris_X_test);    
        a = precision_score(iris_y_test, y_, average=None)
    
        # Meto los valores de precision para cada clase
        clase1.append(a[0])
        clase2.append(a[1])
        clase3.append(a[2])
        
    #calculo un valor para la media
    for i in range(0,len(clase1)):
        sum1 = (float) (sum1) + (float) (clase1[i])
    for i in range(0,len(clase2)):
        sum2 = (float) (sum2) + (float) (clase2[i])
    for i in range(0,len(clase3)):
        sum3 = (float) (sum3) + (float) (clase3[i])
                #calculo el valor real de la media
    sum1 = (float) (sum1) / len(clase1)
    sum2 = (float) (sum2) / len(clase2)
    sum3 = (float) (sum3) / len(clase3)
    #añado los valores a una lista para una lista para poder dibujarlos
    media1.append(sum1)
    media2.append(sum2)
    media3.append(sum3)
    #vuelvo a inicialisar los valores de la variables a utilisar.
    sum1=0
    sum2=0
    sum3=0
    clase1 = []
    clase2 = []
    clase3 = []
#Dibujo las graficas para las particiones de training respecto a las medias de las 
# clases
pl.plot(training,media1, label="Media clase 1")
pl.plot(training,media2, label="Media clase 2")
pl.plot(training,media3, label="Media clase 3")
pl.legend(loc="best")
indice = len(training)
pl.xlim(training[0],training[indice])
pl.ylim(-0.5,1.05)
pl.title("Size aprendizaje distancia de Manhattan")    
pl.savefig('size_aprendizaje_l2')
pl.show() 