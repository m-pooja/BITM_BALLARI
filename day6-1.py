# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 13:16:02 2018
DAY6, BITM, BALLARI
@author: pooja
"""
"""
In this project we will be working 
#with a fake advertising data set, 
#indicating whether or not a particular 
#internet user clicked 
#on an Advertisement on a company website. 
#We will try to create a model that 
#will predict whether or not they will 
#click on an ad based 
#off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': 
    consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': 
    Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': 
    Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line':
    Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': 
    Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad':
    0 or 1 indicated clicking on Ad
"""
#--------------------------
"""
#import data
#Check the head 
#use info nd describe
# ## Exploratory Data Analysis:-
#       Create a histogram of the Age
#       plot Area Income versus Age
#       plot Daily Time spent on site vs. Age.
#       'Daily Time Spent on Site' vs. 'Daily Internet Usage'
#       'Clicked on Ad' column feature vs whole data
#Split the data into training set and 
      testing set using train_test_split**
#Train and fit a logistic regression model on 
      the training set.**
#Now predict values for the testing data.**
#Create a classification report for the model.**
"""
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('advertising.csv')
data.columns
data.head()
data.describe()
data.Age.hist()
plt.scatter(data['Area Income'],data.Age)
plt.scatter(data['Daily Time Spent on Site'], data.Age)
col=list(data.columns)
from sklearn.cross_validation import train_test_split
X=data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']]
Y=data['Clicked on Ad']
x_tr,x_t,y_tr,y_t=train_test_split(X,Y)
from sklearn.linear_model import LogisticRegression
m=LogisticRegression()
m.fit(x_tr, y_tr)
predict=m.predict(x_t)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_t,predict))
print(accuracy_score(y_t,predict)*100)
#on titanics data classify as per 'Survived' column
#titanics data:-make classes of 'Age' as children, 
#adolescents, adults, senior citizens and classify as per these classes

#------Support Vector Machine for Classification------ 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
X=data[['Daily Time Spent on Site','Clicked on Ad','Area Income','Daily Internet Usage']]
Y=data['Age']
x_tr,x_t,y_tr,y_t=train_test_split(X,Y)

s=SVC(kernel='poly')
s.fit(x_tr,y_tr)
prediction_s=s.predict(x_t)
print(classification_report(y_t,prediction_s))
print(accuracy_score(y_t,prediction_s)*100)
print(confusion_matrix(y_t, prediction_s))



import pandas as pd
nudata=pd.read_csv('amex-listings.csv')
nudata.info()
nudata.columns



 
 



 
 
 
 
 
 
 




























