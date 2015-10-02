# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:14:26 2013

@author: jose-ia
"""

from sklearn.linear_model import  Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.grid_search import GridSearchCV 
from sklearn.cross_validation import cross_val_score
from os.path import dirname
from os.path import join
import numpy as np 
import pylab as pl 
import math
from sklearn import linear_model, preprocessing, svm
from sklearn.datasets import load_boston, load_iris
from decimal import *




#LECTURA DE FICHEROS Y CARGA DE ARCHIVO
base_dir = join(dirname(__file__), 'data')

traininfo = np.loadtxt(join(base_dir, 'mcycleTrain.txt'))
x_train=traininfo[:,1]
y_train=traininfo[:,0]

testinfo = np.loadtxt(join(base_dir, 'mcycleTest.txt'))
x_test=testinfo[:,1]
y_test=testinfo[:,0]
muestras=len(x_train)+len(y_train)
#FIN CARGA DE ARCUIVOS
def ComprobacionDatosTest():

    pl1 = pl.scatter(x_train, y_train, color='red')
    pl2 = pl.scatter(x_test, y_test, color='green')
    pl.legend((pl1, pl2), ("Entrenamiento", "Test"), loc='best')    
    pl.title('Representacion de los datos')
    pl.savefig('representacion de los datos')
    pl.show()
        
    return None
#ComprobacionDatosTest()



def CargarBaseDatos(bd):

    #CARGA DATOS
    if (bd=="boston"):
        x = load_boston()
    if (bd=="iris"):
        x = load_iris()
    
    data=x.data
    target=x.target
    nFil, nCol = data.shape
    return data, target, nFil


def particionarDatos(numero_muestras,training, X, y):
    #partimos los datos
    lt=np.round(training*len(X))
    # particion aleatoria de los datos para training y test
    np.random.seed(0)
    indices = np.random.permutation(numero_muestras)
    X_train = X[indices[:-lt]]
    y_train = y[indices[:-lt]]
    X_test  = X[indices[-lt:]]
    y_test  = y[indices[-lt:]]
    
    return X_train,y_train,X_test,y_test


def fun(x,d):
    dim=range(0,d+1)
    for dimension in dim:
        if(dimension<1):
            matriz=np.reshape(np.power(x, dimension), (len(x), 1))
        else:
            matriz=np.concatenate( (matriz, np.reshape(np.power(x, dimension), (len(x), 1))), 1 )
  
    return matriz


def RegresionApartadoA(modelo='Lineal'):
    dim=np.array([1, 3, 5, 7, 10, 18])
    c=['Red', 'Blue', 'Green', 'Magenta', 'Black', 'Orange']
    cdim=['Dim 1', 'Dim 3', 'Dim 5', 'Dim 7', 'Dim 10', 'Dim 18']
    cont=0
    vector=np.arange(0,2,0.004)

    for d in dim:
        x_entrenamiento=fun(x_train, d)
        valXtest=fun(vector,d)
        
        if(modelo=='Lineal'):
            reg=linear_model.LinearRegression()
        else:
            reg=linear_model.Ridge(0.5)
        
        reg.fit(x_entrenamiento, y_train)
        prediccion=reg.predict(valXtest)
        
        pl.subplot(3, 2, c.index(c[cont])+1)
        pl.plot(vector,prediccion, color=c[cont], label=cdim[cont])
        pl.title(str(cdim[cont])+" - "+str(c[cont])+" ("+str(modelo)+")")
        
        pl.scatter(x_train, y_train)
        
        pl.xlim(0, 2)
        pl.ylim(-150,100)
        pl.xlabel("f(x)");
        pl.ylabel("y")
        cont+=1
       
    pl.savefig("P2-RegresionApartadoA ("+str(modelo)+")")
    pl.show()

#RegresionApartadoA()
#RegresionApartadoA('Ridge')

def RegresionApartadoB():
    dim=list([1, 3, 5, 7, 10, 18])
    errorTrainLineal=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    errorTestLineal=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    errorTrainRidge=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    errorTestRidge=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for d in dim:
        x_entrenamiento=fun(x_train, d)
        valXtrain=fun(x_train,d)
        valXtest=fun(x_test,d)

        regL=linear_model.LinearRegression()
        
        regL.fit(x_entrenamiento, y_train)
        
        prediccionTrain=regL.predict(valXtrain)
        prediccionTest=regL.predict(valXtest)
        
        errorTrainLineal[dim.index(d)]=mean_squared_error(y_train, prediccionTrain)
        errorTestLineal[dim.index(d)]=mean_squared_error(y_test, prediccionTest)
        
        
        regR=linear_model.Ridge(0.8)
        
        regR.fit(x_entrenamiento, y_train)
        
        prediccionTrain=regR.predict(valXtrain)
        prediccionTest=regR.predict(valXtest)
        
        errorTrainRidge[dim.index(d)]=mean_squared_error(y_train, prediccionTrain)
        errorTestRidge[dim.index(d)]=mean_squared_error(y_test, prediccionTest)

    
    pl.subplot(1, 2, 1)
    pl.plot(dim, errorTrainLineal, color='r', label="Error Train")
    pl.plot(dim, errorTestLineal, color='b', label="Error Test")
    
    pl.title("Error Modelo Regresion Lineal")
    pl.legend(["Error Train", "Error Test"])
    
    pl.xlim(0,20)
    pl.ylim(0, 5000)
    
    
    pl.subplot(1, 2, 2)
    pl.plot(dim, errorTrainRidge, color='r', label="Error Train")
    pl.plot(dim, errorTestRidge, color='b', label="Error Test")
    
    pl.title("Error Modelo Ridge")
    pl.legend(["Error Train", "Error Test"])
    
    pl.xlim(0,20)
    pl.ylim(0, 5000)
    
    pl.savefig("P2-RegresionApartadoB")
    pl.show()


#RegresionApartadoB()

def RegresionApartadoCa():
    datos,etiquetas,tamanio=CargarBaseDatos("boston")
    X_train,y_train,X_test,y_test=particionarDatos(tamanio,0.7,datos, etiquetas)    
        
    alphas=np.arange(0.1, 1, 0.1)
       
    for a in alphas:
        reg=linear_model.Lasso(alpha=a)
                
        reg.fit(X_train, y_train)
        prediccion=reg.predict(X_test)
        print "Alpha="+str(a)+" = " +str(mean_squared_error(y_test, prediccion))
   

#RegresionApartadoCa()

'''REGRESION LINEAL
Ejecute estos apartados para resolver los problemas de regresión lineal
#ComprobacionDatosTest()
#RegresionApartadoA()
#RegresionApartadoA('Ridge')
#RegresionApartadoB()
#RegresionApartadoCa()
'''

def PerceptronApartadoA():
    x=load_iris()
    data=x.data
    label=x.target
    
    caract=list(["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"])
    label=list(["setosa", "versicolour", "virginica"])
    color=list(['r', 'g', 'b'])
    cont=0
    
    for c in caract:
        x=caract.index(c)
        for c in caract[x+1:]:
            y=caract.index(c)
            print str(x)+" "+str(y)
            pl.subplot(3, 3, cont+1)
            for l in label:
                indice=label.index(l)
                pl.scatter(data[indice*50:indice*50+50, x], data[indice*50:indice*50+50, y], c=color[indice])
                pl.title(str(caract[x]+"/"+str(caract[y])))
            
            cont=cont+1
    pl.legend(label, loc=2)
    pl.show()




#PerceptronApartadoA()

def Perceptron_Parejas(pareja1, pareja2):
     
    datos,etiquetas,tamanio=CargarBaseDatos("iris")
    X_train,y_train,X_test,y_test=particionarDatos(tamanio,0.7, datos, etiquetas) 
    X= X_train[:,[pareja1,pareja2]]
    Y= X_test[:,[pareja1,pareja2]]
    tamTrain=len(X_train)

    prc = Perceptron().fit(X,y_train)
    coef = prc.coef_
    intercept = prc.intercept_
    color = "rgb"

    for i in range(0,tamTrain,1):
        if(y_train[i]==0):
            pl.scatter(X_train[i,pareja1], X_train[i,pareja2], color=color[y_train[i]])
        elif(y_train[i]==1):
            pl.scatter(X_train[i,pareja1], X_train[i,pareja2], color=color[y_train[i]])
        elif(y_train[i]==2):
            pl.scatter(X_train[i,pareja1], X_train[i,pareja2], color=color[y_train[i]])
    pl.axis('tight')
    
    xmin, xmax = pl.xlim()
    ymin, ymax = pl.ylim()    
    
    for i in range(0,3,1):
        pl.plot([xmin, xmax], [((-(xmin * coef[i, 0]) - intercept[i]) / coef[i, 1]), ((-(xmax * coef[i, 0]) - intercept[i]) / coef[i, 1])],ls="--", color=color[i])
    pl.show()
    
    y_ = prc.predict(Y)
    accuracy = accuracy_score(y_test, y_)
    recall = recall_score(y_test, y_, average=None)
    precision = precision_score(y_test, y_, average=None)
    print "accuracy: "+str(accuracy)
    print "recall: "+str(recall)
    print "precision por clase: "+str(precision)
       
#Perceptron_Parejas(0,3)
'''
PERCEPTRON
para resolver los ejercicios de perceptron ejecute los siguientes apartados
#PerceptronApartadoA()
#Perceptron_Parejas(0,3) Altere el valor de las parejas.
'''
def SVM_HiperplanosLinealSVC(pareja1, pareja2):
    
    datos,etiquetas,tamanio=CargarBaseDatos("iris")
    X_train,y_train,X_test,y_test=particionarDatos(tamanio,0.7, datos, etiquetas) 
    X= X_train[:,[pareja1,pareja2]]
    tamTrain=len(X_train)
    
    svc = LinearSVC().fit(X, y_train)
    color = "rgb"
    
    
    for i in range(0,tamTrain,1):
        if(y_train[i]==0):
            pl.scatter(X_train[i,pareja1], X_train[i,pareja2], color=color[y_train[i]])
        elif(y_train[i]==1):
            pl.scatter(X_train[i,pareja1], X_train[i,pareja2], color=color[y_train[i]])
        elif(y_train[i]==2):
            pl.scatter(X_train[i,pareja1], X_train[i,pareja2], color=color[y_train[i]])
    pl.axis('tight')
    
    xmin, xmax = pl.xlim()
    ymin, ymax = pl.ylim()    

    for j in range(0,3,1):
        w = svc.coef_[j]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (svc.intercept_[j]) / w[1]     
    
        p = svc.decision_function(X)
        sv1 = []
        sv2 = []
        context = Context(prec=1, rounding=ROUND_DOWN)
        normal = math.sqrt((w[0]**2)+(w[1]**2))
        distance = context.create_decimal_from_float(1/normal)
        for i in range(len(p)):
            punto = (p[i,[j]])
            if (abs(context.create_decimal_from_float(punto[0]))-abs(distance)==0):
                pl.scatter(X[i,0], X[i,1],  s=80, facecolors='none')
                if(y_train[i]==j):
                    sv1.append(X[i])
                else:
                    sv2.append(X[i])
    
        if(len(sv1)>0):
            aux = sv1[0]
            yy2 = a * xx + (aux[1] - a * aux[0])
            pl.plot(xx, yy2, 'k--', color=color[j])

        if(len(sv2)>0):
            aux = sv2[0]
            yy3 = a * xx + (aux[1] - a * aux[0])  
            pl.plot(xx, yy3, 'k--', color=color[j])

        pl.plot(xx, yy, 'k-', color=color[j])
    
    pl.savefig("hiperplanos"+str(pareja1)+str(pareja2))
    pl.show()
    
    

    return None
    
#SVM_HiperplanosLinealSVC(1,3)
 
def SVMApartadoA():
    datos,etiquetas,tamanio=CargarBaseDatos("iris")
    X_train,y_train,X_test,y_test=particionarDatos(tamanio,0.7, datos, etiquetas)   
    #Escalar datos
    scaler=preprocessing.Scaler().fit(X_train)
    scaler.mean_
    scaler.std_
    scaler.transform(X_train)
    scaler.transform(X_test)

    for i in range(len(y_train)):
        if(y_train[i]==2):
            y_train[i]=1
     
    # fit the model
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train[:, 2:4], y_train)
        
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 10)
    yy = a * xx - (clf.intercept_[0]) / w[1]
     
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])
     
    # plot the line, the points, and the nearest vectors to the plane
    pl.plot(xx, yy, 'k-')
    pl.plot(xx, yy_down, 'k--')
    pl.plot(xx, yy_up, 'k--')
     
    pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=80, facecolors='none')
    pl.scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap=pl.cm.Paired)
   
    pl.axis('tight')
    pl.show()
#SVMApartadoA()    


def Rbf( X_train,y_train,X_test,y_test, rango_c, rango_gamma, pareja1, pareja2):
    
    X= X_train[:,[pareja1,pareja2]]
     
    
    parameters =   {'C': rango_c, 'gamma': rango_gamma},
    svr = SVC(kernel='rbf')
    clf = GridSearchCV(svr, parameters, cv=len(X)-1)
    clf.fit(X_train, y_train)
    
    a = clf.best_params_

    return a.get("C"),a.get("gamma")


   
   
def Poly( X_train,y_train,X_test,y_test,rango_c, rango_gamma, rango_poly, pareja1, pareja2):
    
    X= X_train[:,[pareja1,pareja2]]
    
    parameters =   {'C': rango_c, 'gamma': rango_gamma, 'degree': rango_poly},
    svr = SVC(kernel='poly')
    clf = GridSearchCV(svr, parameters, cv=len(X)-1)
    clf.fit(X_train, y_train)
    
    a = clf.best_params_

    return a.get("C"),a.get("gamma"),a.get("degree")
    
def P_C(rango_c, xtrain, ytrain, pareja1, pareja2):
    
    X= xtrain[:,[pareja1,pareja2]]
    
    
    scores = []
    scores_std = []
    accuracy_train = []
    
    print "Evaluando el rango de valores"

    for i in rango_c:    
        clf = SVC(C=i, kernel='linear',probability=True)
        score = cross_val_score(clf, X, ytrain, cv=10)
        y_ = clf.fit(X,ytrain).predict(X)
        accuracy_train.append(accuracy_score(ytrain, y_ ))
        scores.append(np.mean(score))
        scores_std.append(np.std(score))
        print "evaluando",i
    
     
    pl.plot(rango_c, scores, 'b')
    pl.plot(rango_c, accuracy_train, 'r')
    pl.ylabel('CV ')
    pl.xlabel('C')
    pl.xlim(min(rango_c), max(rango_c))
    pl.ylim(min(scores),max(scores))   
    pl.axhline(np.max(scores), linestyle='--', color='.5')
    pl.savefig("C"+str(pareja1)+str(pareja2))
    pl.show()
    
    
    pl.plot(rango_c, accuracy_train, 'r')
    pl.ylabel('accuaracy')
    pl.xlabel('C')
    pl.savefig("C"+str(pareja1)+str(pareja2))
    pl.show()


    index = scores.index(max(scores))

    return rango_c[index]
    

def Pintarsolucion(pareja1, pareja2):
    
    datos,etiquetas,tamanio=CargarBaseDatos("iris")
    X_train,y_train,X_test,y_test =particionarDatos(tamanio,0.7, datos, etiquetas) 
    C_s = np.arange(0.1, 30, 0.5)
    P_s = np.arange(1,8,1)
    G_s = np.arange(0.1, 10,1)
    C1 = P_C(C_s, X_train, y_train, pareja1,pareja2)

    C3,G3,D3=Poly(X_train,y_train,X_test,y_test,C_s, G_s, P_s, pareja1, pareja2)
    C2, G2 = Rbf(X_train,y_train,X_test,y_test,C_s, G_s, pareja1, pareja2)

    X = X_train[:,[pareja1,pareja2]]
    Y = y_train
    h = .02  # step size in the mesh
    
    svc = SVC(kernel='linear', C=C1).fit(X, Y)
    rbf_svc = SVC(kernel='rbf', C=C2, gamma=G2).fit(X, Y)
    poly_svc = SVC(kernel='poly', degree=D3, gamma=G3, C=C3).fit(X, Y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    # title for the plots
    
    titles = ['SVC with linear kernel','SVC with RBF kernel',
              'SVC with polynomial kernel',
            'LinearSVC (linear kernel)']

 
    for i, clf in enumerate((svc, rbf_svc, poly_svc)):

        pl.subplot(2, 2, i + 1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
        pl.axis('off')
        pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)    
        pl.title(titles[i])
    pl.savefig("final"+str(pareja1)+str(pareja2))
    pl.show()
    
    return None
    
    

#Pintarsolucion(2, 3)

'''
SVM 
Para este apartado hay tres funciones
#SVMApartadoA()    emplea LinearSVC
#SVM_HiperplanosLinealSVC(1,3) Dibuja los hiperplanos solucion.
#PintarSolucion(0,1) Muestra la solucion para diferentes kernerl.
'''