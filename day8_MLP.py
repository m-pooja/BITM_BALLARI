# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:28:58 2018

@author: pooja
"""

import pandas as pd
import sklearn.neural_network as nn
from sklearn.utils import shuffle
#from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
iris = load_iris()
#creating dataframe from iris bunch
data=pd.DataFrame(iris.data, columns=iris.feature_names)
data['Class']=iris.target
#shuffling data
nudata_suffled=shuffle(data)
x=nudata_suffled.iloc[:,:4]
y=nudata_suffled.iloc[:,4]
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y, random_state=0)
#with default parameters
clf=nn.MLPClassifier()
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
clf.score(Y_test,pred)
#with different hidden layer and size
clf=nn.MLPClassifier(hidden_layers=(100,4))
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
clf.score(Y_test,pred)
#Regression
#with default parameters
clfr=nn.MLPRegressor()
clfr.fit(X_train,Y_train)
pred=clfr.predict(X_test)
clfr.score(Y_test,pred)
#with different hidden layer and size
clfr=nn.MLPRegressor(hidden_layers=(100,4))
clfr.fit(X_train,Y_train)
pred=clfr.predict(X_test)
clfr.score(Y_test,pred)