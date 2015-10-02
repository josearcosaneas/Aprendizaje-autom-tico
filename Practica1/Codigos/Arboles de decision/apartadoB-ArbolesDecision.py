# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:30:15 2013

@author: jose-ia
"""
print __doc__

import numpy as np
from sklearn import  datasets, preprocessing


from sklearn import tree
def leerDatos():
    iris = datasets.load_iris()
    iris_X,iris_y = iris.data,iris.target
    return iris_X,iris_y

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
# Bucle que itera de 1 a 8 para min_samples_split
iris_X,iris_y=leerDatos()
n_muestras= (int) (len(iris_X))    
# Particionamos los datos   
iris_X_train,iris_y_train,iris_X_test,iris_y_test=particionarDatos(n_muestras,0.3,iris_X,iris_y)
 
 #Estandarizar
scaler=preprocessing.Scaler().fit(iris_X_train)
scaler.mean_
scaler.transform(iris_X_train,iris_X_test)
title="arbolmin_samples_leaf=1"
# Dibujamos el arbol con min_samples_leaf
arbol = tree.DecisionTreeClassifier(min_samples_leaf=8)
y_predict = arbol.fit(iris_X_train, iris_y_train)
import StringIO, pydot 
dot_data = StringIO.StringIO() 
tree.export_graphviz(y_predict, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf(title ) 
   
