# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 09:12:48 2018
DAY6, BITM, BALLARI
@author: pooja
"""
#--------------feature scalling
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape)
# separate the data from the target attributes
X = iris.data
y = iris.target
x_data=preprocessing.normalize(X)
normalized_X = pd.DataFrame(x_data, columns=iris.feature_names)
normalized_X['class']=iris.target
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(normalized_X,'class')
original=pd.DataFrame(X,columns=iris.feature_names)
original['class']=iris.target
parallel_coordinates(original,'class')
#standardize
s_x = preprocessing.scale(X)
#creating dataframe for std data
std_dataframe=pd.DataFrame(s_x, columns=iris.feature_names)
#adding target values to dataframe
std_dataframe['class']=iris.target
#ploting parallel coordinates on stdized data
parallel_coordinates(std_dataframe,'class')
#minmaxscaler-------------
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6],[0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.transform(data))
x_bin=preprocessing.binarize(X,threshold=4)
x_bin_df=pd.DataFrame(x_bin, columns=iris.feature_names)
x_bin_df['class']=iris.target
parallel_coordinates(x_bin_df,'class')
#----------------------classify---------

"""
what can i do on iris data?
SUPERVISED LEARNING:-
    when u have DATA+TARGET/LABEL
    if TARGET/LABEL=='categorical value'('car','tree')|discrete value(0,1,2)
        classification
    elif TARGET/LABEL==continuous values
        regression
unsupervised learning?
    when u only have data: no target values
"""
#classification on IRIS data
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd
#import data
from sklearn.datasets import load_iris
iris = load_iris()

#create (Y) dependent variable from data
Y=iris.target
#create (X) independent variale
X=iris.data
#shuffling data
nudata_suffled=sklearn.utils.shuffle(std_dataframe)
x=nudata_suffled.iloc[:,:4]
y=nudata_suffled.iloc[:,4]
#split X and Y into train and test set using:-
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y, random_state=0)
#call model: create its object
lr=LogisticRegression()
#fit and predict
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
#import accuracy_score from sklearn.metrics 
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,y_pred)*100)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))



"""Dumping the model"""   #(saving our models)
#joblib
from sklearn.externals import joblib
joblib.dump(lr,"name.model",compress=5)


model=joblib.load("name.model")

predict=model.predict(x_t)
print(accuracy_score(y_t,predict))
model.close


"""pickle"""
import pickle
mymodel=open("RF.model",'wb')
pickle.dump()

mymodel=open('RF.model', 'rb')
pickle.load()











































