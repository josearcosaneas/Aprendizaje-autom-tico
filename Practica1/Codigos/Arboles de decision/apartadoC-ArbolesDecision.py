# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:06:50 2013

@author: jose-ia
"""

print __doc__

import numpy as np 
#import pylab as pl 
from sklearn import datasets 
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import StringIO, pydot 
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
#Creo los dos arboles con los diferentes criterios
arbol1 = DecisionTreeClassifier(criterion='gini')
arbol2 = DecisionTreeClassifier(criterion='entropy')
# Entrenamos y predecimos los datos
y_predict = arbol1.fit(iris_X_train, iris_y_train).predict(iris_X_test)
y2_predict = arbol2.fit(iris_X_train, iris_y_train).predict(iris_X_test)  
#Calculamos la precision
a = precision_score(iris_y_test, y_predict, average='weighted')    
b = precision_score(iris_y_test, y2_predict, average='weighted')    
# imprimo la precision de los dos arboles.
print a 
print b
# Guardo los arboles en formato pdf
dot_data = StringIO.StringIO() 
tree.export_graphviz(arbol1, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("irisgini.pdf")  

dot_data = StringIO.StringIO() 
tree.export_graphviz(arbol2, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("irisentropia.pdf") 
