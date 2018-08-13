# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:41:05 2018
DAY7 BITM, BALLARI
@author: pooja
"""
import sklearn
a
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
iris = load_iris()
#creating dataframe from iris bunch
data=pd.DataFrame(iris.data, columns=iris.feature_names)
data['Class']=iris.target
#shuffling data
nudata_suffled=sklearn.utils.shuffle(data)
x=nudata_suffled.iloc[:,:4]
y=nudata_suffled.iloc[:,4]
#spliting data for training and testing
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y, random_state=0)
#creating model
knn=KNeighborsClassifier()
knn.fit(X_train, Y_train)
predict=knn.predict(X_test)
print(accuracy_score(Y_test,predict)*100)
c_mat=confusion_matrix(Y_test,predict)
class_report=classification_report(Y_test,predict)
#writing results in a file
with open('iris_results.txt', 'a') as f:
    f.write('Results of KNN')
    f.write('\n')
    f.write(class_report)

#LABEL ENCODING-----------
"""
A LabelEncoder converts a categorical data into a number ranging from 0 to n-1,
 where n is the number of classes in the variable.
For example, in case of Outlook, there are 3 clasess – 
Overcast, Rain, Sunny. 
These are represented as 0,1,2 in alphabetical order.
"""
play_tennis=pd.read_csv('PlayTennis.csv')
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])
#Defining the features and the target variables.
features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"
#To validate the performance of our model, 
#we create a train, test split. 
#We build the model using the train dataset and 
#we will validate the model on the test dataset.
features_train, features_test, target_train, target_test = train_test_split(play_tennis[features],play_tennis[target],test_size = 0.33,random_state = 54)
#Let’s create the model now.
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(features_train, target_train)
#Now we are ready to make predictions on the test features.
#We will also measure the performance of the model using accuracy score.
#Accuracy score measure the number of right predictions.
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
#The accuracy is in this case about 0.80000000000000004
#Now suppose we want to predict for the conditions,
#Outlook	Temperature	Humidity	Wind
#Rain	Mild	High	Weak
print (model.predict([[1,2,0,1]]))
#which gives a prediction 1 (Yes)
#--------------------
titanic=pd.read_csv('TST.csv')
Bins = [0, 14, 20, 64, 80]
BinLabels = ['under14','adolescents', 'adult','senior']
titanic['Age']=pd.cut(titanic['Age'], Bins, labels=BinLabels)
titanic['Age']=titanic['Age'].fillna('adult')
from sklearn.preprocessing import LabelEncoder
e=LabelEncoder()
titanic['Age']=e.fit_transform(titanic['Age'])
#---------------------------------
#LINEAR REGRESSION
import numpy as np
import statsmodels.api as sm
import pandas as pd
df=pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv',index_col=0)
df.head()
y = df.Employed
X = df[['GNP','Population','Armed.Forces']]
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

est.summary()
est.fittedvalues                                                          
est.resid
np.mean(est.resid)
import pylab 
import scipy.stats as stats
#pylab.probplot(est.resid)
stats.probplot(est.resid)
pylab.show()

import scipy
scipy.stats.probplot(est.resid)
plt.scatter(df.GNP,y)
stats.probplot(est.resid)
pylab.show()
#multiple regression
stud_reg = pd.read_excel('Students.xlsx', sheetname='Data')
print("The list of row indicies")
print(stud_reg.index)
print("The column headings")
print(stud_reg.columns)
X = stud_reg[ ['PLACE_RATE', 'NO_GRAD_STUD'] ]
y = stud_reg.APPLICANTS
X = sm.add_constant(X)
est = sm.OLS(y, X)
est = est.fit()
est.summary()
from sklearn.linear_model import LinearRegression
x=df[['GNP','Population','Armed.Forces']]
y=df.Employed
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y)
linreg=LinearRegression()
linreg.fit(X_train, Y_train)
linreg.score(X_train, Y_train)
print(linreg.coef_)
print(linreg.intercept_)
X_test['predict_Y']=linreg.predict(X_test)
X_test['actual_Y']=Y_test

"""
import pandas as pd
nudata=pd.read_csv('USA_Housing.csv')
"""
from sklearn.datasets import fetch_california_housing
chdata=fetch_california_housing()
chdata.DESCR
x_uh=chdata.data
y_uh=chdata.target
from sklearn.cross_validation import train_test_split
Xuh_train,Xuh_test,Yuh_train,Yuh_test=train_test_split(x_uh,y_uh)
linreguh=LinearRegression()
linreguh.fit(Xuh_train, Yuh_train)
linreguh.score(Xuh_train, Yuh_train)
print(linreguh.coef_)
print(linreguh.intercept_)
dataframe_xuh=pd.DataFrame(x_uh)
X_uh_nu=pd.DataFrame(dataframe_xuh.iloc[:,2])
X_uh_nu['4']=dataframe_xuh.iloc[:,4]
Xuh_train,Xuh_test,Yuh_train,Yuh_test=train_test_split(X_uh_nu,y_uh)
linreg_uh=LinearRegression()
linreg_uh.fit(Xuh_train, Yuh_train)
linreg_uh.score(Xuh_train, Yuh_train)
print(linreg_uh.coef_)
print(linreg_uh.intercept_)
predict=linreg_uh.predict(Xuh_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Yuh_test,predict))

#----------------------
from sklearn.cluster import KMeans


















