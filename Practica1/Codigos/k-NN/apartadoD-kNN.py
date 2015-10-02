# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:18:33 2013

@author: jose-ia
"""

print __doc__

import numpy as np 
import pylab as pl 
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score

#leo datos
iris = datasets.load_iris()
iris_X,iris_y = iris.data,iris.target
n_muestras= (int) (len(iris_X))
def particionarDatos(numero_muestras,training, X, y):
    #partimos los datos
    lt=np.round(training*len(X))
    # particion aleatoria de los datos para training y test
    np.random.seed()
    indices = np.random.permutation(numero_muestras)
    iris_X_train = X[indices[:-lt]]
    iris_y_train = y[indices[:-lt]]
    iris_X_test  = X[indices[-lt:]]
    iris_y_test  = y[indices[-lt:]]
    
    return iris_X_train,iris_y_train,iris_X_test,iris_y_test
# Creamos variables y listas para realisar el experimento
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
# Calculamos los valores para training
for i in range(50,100,10):
    if(i==90):
        i= (float) ((n_muestras-1)*100)/n_muestras
        print i
                
    training.append(i) 
    
    for j in range(0,10):
        iris_X_train,iris_y_train,iris_X_test,iris_y_test = particionarDatos(n_muestras, i, iris_X,iris_y)
        neigh = KNeighborsClassifier(1, weights='uniform')
        y_ = neigh.fit(iris_X_train, iris_y_train).predict(iris_X_test);  
        a = precision_score(iris_y_test, y_, average=None)
    	# Añadunis los valores de la precision para cada clase
        clase1.append(a[0])
        clase2.append(a[1])
        clase3.append(a[2])
                
    for i in range(0,len(clase1)):
        sum1 = (float) (sum1) + (float) (clase1[i])
    for i in range(0,len(clase2)):
        sum2 = (float) (sum2) + (float) (clase2[i])
    for i in range(0,len(clase3)):
        sum3 = (float) (sum3) + (float) (clase3[i])
                
    sum1 = (float) (sum1) / len(clase1)
    sum2 = (float) (sum2) / len(clase2)
    sum3 = (float) (sum3) / len(clase3)
    # Añadimos valores de medias
    media1.append(sum1)
    media2.append(sum2)
    media3.append(sum3)
   
# Dibujamos las graficas     
pl.plot(training,media1, label="promedio clase 1")
pl.plot(training,media2, label="promedio clase 2")
pl.plot(training,media3, label="promedio clase 3")
pl.legend(loc="best")
indice = len(training)-1
pl.xlim(training[0],training[indice])
pl.ylim(-0.5,1.05)
pl.title("Aprendizaje con euclidea")    
pl.savefig('size_aprendizaje_l2')
pl.show() 
