# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:01:10 2013

@author: jose-ia
"""
print __doc__

import numpy as np
import pylab as pl
from sklearn import  datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

def leerDatos():
    iris = datasets.load_iris()
    iris_X,iris_y = iris.data,iris.target
    return iris_X,iris_y,len(iris_X)

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

X, y, muestras = leerDatos()
training_x, training_y, test_x, test_y = particionarDatos(muestras, 0.3, X, y)

#Creamos lista y variables para poder almacenar las medias del error de test y training
n_items=8#determinamos el numero de items en 8
media_test = []
media_final_test= []
media_total_test = 0
media_train = []
media_final_train = []
media_total_train = 0
    
#Para cada numero de min_samples_leaf(1-8 incluidos) repetimos 100 veces el experimento
for i in range(1,(n_items)+1,1):
    for j in range(0,100,1):        
        #particionamos los datos siempre que realicemos el experimento
        iris_X_train, iris_y_train, iris_X_test, iris_y_test = particionarDatos(muestras, 30, X, y)
        #creamos el arbol
        clf = DecisionTreeClassifier(min_samples_leaf=i)
        #entrenamos y predecimos valores para test y training
        y_ = clf.fit(iris_X_train, iris_y_train).predict(iris_X_test);   
        yy_ = clf.fit(iris_X_train, iris_y_train).predict(iris_X_train);   
        # Calculamos el error para training y test
        a = 1-precision_score(iris_y_test, y_, average=None)
        b = 1-precision_score(iris_y_train, yy_, average=None)
        #Calculamos la media de test y training para cada iteraccion
        c = (float) (a[0]+a[1]+a[2])/3
        media_test.append(c)
        d = (float) (b[0]+b[1]+b[2])/3
        media_train.append(d)
        #Despues de 100 iteracciones sumamos los valores de media[] en una nueva variable
    for k in range(0, len(media_test), 1):
        media_total_test = media_total_test + media_test[k]
        #hacemos lo mismo para los datos de training
    for k in range(0, len(media_train), 1):
        media_total_train = media_total_train + media_train[k]
        #calculamos el valor real de la media para cada 100 iteracciones
    media_total_test = float (media_total_test / len(media_test))
    media_total_train = float (media_total_train / len(media_train))
    # añadimos lo valores de la media reales en otra lista que sera la que representemos.
    media_final_test.append(media_total_test)
    media_final_train.append(media_total_train)
    # volovemos a inicialisar los valores de las variables
    media_total_test=0
    media_test = []
    media_total_train=0
    media_train = []
           
pl.plot(range(1,n_items+1,1),media_final_test, label="Media de error en test")
pl.plot(range(1,n_items+1,1),media_final_train, label="Media de error en entrenamiento")
pl.legend(loc="best")
pl.xlim(1,n_items+1)
pl.ylim(-0.05,0.2)
pl.title("ApartadoA")    
pl.savefig('ApartadoA')
pl.show() 
