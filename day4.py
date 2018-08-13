# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:45:06 2018
DAY4, BITM, BALLARI
@author: pooja
"""
import numpy as np
np.abs(-56)
a=np.arange(6)
print(type(a))
b=np.arange(10,16)
#c=np.arange(100,201,4)
c=np.arange(20,26)
d=np.arange(30,36)
e=np.arange(40,46)
f=np.arange(50,56)
arr=np.array([a,b,d,c,e,f],dtype='str')
arr[0:2,2:]
"""
[[0, 2, 4],
[30, 32, 34],
[40, 42, 44]]
"""
arr[::2,::2]
arr1=np.arange(0,80,10)
l=[1,2,3,5,6,7] 
a=np.array(l)  
l1=[[1,2,3],[11,12,13],[21,22,23]] 
a1=np.array(l1)  
a2=np.arange(0,11,2) 
a3=np.zeros(3) 
a4=np.ones((3,2))  
a5=np.linspace(0,20,50)
a6=a5.reshape(10,5)
a6.shape
aa=np.full((3,4),12) 
from numpy import random
random.randint(4,56)
arand=random.rand(3,5)
arn=random.randn(4,4)
a6=np.eye(4) 
a7=np.random.rand(5) #uniform dist a8=np.random.randn(2) 
a_7=np.random.rand(5,4)
arr1=np.random.randint(5,100,4)
arr1.max() 
arr1.min() 
a1.max()
arr1.sum()
a1.sum()
a1.sum(axis=0)
"""
array([33, 36, 39])
"""
#index location of max n min  
arr1.argmin() 
arr1.argmax()  
arand.argmax()



arr1[2]
#element at index 2? 
a=np.arange(0,50)
a_slice=a[:6] 
a_slice[:]=80
#so numpy doesnot 
#created a new array for 
#a_slice.  
#for copy 
slice_a_copy=a[:6].copy() 
slice_a_copy[:]=67   

a1.std() 
a1.var()
arr1.mean()   
arr1.median()
a1=[1,2,3,4] 
a2=[10,11,12,13] 

#a2-a1  
a1=np.array(a1) 
a2=np.array(a2) 
a2-a1 
a1-a2 
a1*a2 
a1=a1*2 
a1/a1 
asq=np.sqrt(a1) 
np.exp(a1)
a1=np.inf
a1=np.nan
np.isnan(a1)
a2[0]=90
a2[3]=89
a2.sort()
np.insert(a2,1,67)
a6.cumsum(axis=0)
np.delete(a2,[3])
np.append(a2,[90,67,78])

a2[1].size()
aaa=np.add(a2,a6)
np.subtract(a2,a6)
np.multipy()

np.split(a,6)

a2>20
af=a6.ravel()
np.concatenate((aa,aaa), axis=1)

#--------------------------------------------------------------------------
#PANDAS
#---------------------------------------------
import pandas as pd
import numpy as np

labels=['a','b','c','d']  
data=[10,20,30,40]  
arr=np.array(data)  
d={'a':10,'b':20,'c':30,'d':40} 
s=pd.Series(data)
s1=pd.Series(data, labels)
s2=pd.Series([10,20,30,40] , ['a','b','c','d'])
sd=pd.Series(d)
type(sd)
sd.dtype
s1['d']

l1=[[1,2,3],[22,1,12],[21,22,23]] 
a1=np.array(l1) 
ss=pd.DataFrame(a1)
type(ss)
ss1=pd.DataFrame(a1,['a','b','c'], columns=['x','y','z'])
ss1['x']['b']
ss1['y']['c']
sx=ss1.x
sx=ss1['x']

s=pd.Series([10,34,6,34], index=['a','c','d','e'])
s3 = pd.Series([7, 4,-2, 3],index=['a','b', 'c', 'd']) 
s + s3
s.add(s3, fill_value=0)
s.sub(s3, fill_value=2)
s.div(s3, fill_value=4)
s[~(s > 16)]                     
#Series s where value is not >1 >>> 
s[(s < -1) | (s > 2)]           #s where value is <-1 or >2 >>>
df=pd.DataFrame(a1)
type(df)
df=pd.DataFrame(a1,['a','b','c'], columns=['x','y','z'])
df.iloc[[0],[2]]
df.iat[0,2]
df.at['a','y']
df.loc['a','y']
df.ix[2]
df.ix[2,'z']
df.shape            
df.index          
df.columns          
df.info()
df.sort_index()
df.sort_values(by='y') 
df.rank()
df.drop(['a', 'c'])       #Drop values from rows (axis=0)
df.drop('Country', axis=1)
f = lambda x: x*2 
df.apply(f)         
df.applymap(f)
df.sum(axis=1)               
df.cumsum()                                     
df.min()
df.max()      
df.idxmin()
df.idxmax() 
df.describe()          
df.mean()
df.median()
data=pd.read_csv('TST.csv')
data.info()
data.describe()
data.index
data.columns
age=data['Age']
data.Age
aasix=age[age>60]
len(aasix)
#-------------------------------------------------------------
import pandas as pd
D1=pd.read_csv('TST.csv') 
D1 
D1.info() 
D1.describe()## to see the max,min,mean,std,count and quartiles 
## to axcess a particular column 
D1['Age'] 

D1.Age[0:5]
D1.head(20)
D1.tail()
nudata=D1[['Age', 'Survived']]
## to access particular column n row 
D1[['Survived','Age']][10:50]
## to acess particular columns n rows
D1['Pclass'].unique() 
D1['Age'].value_counts() 
##gives the counts in a particular column 
D1['Age'].value_counts(normalize=True)*10000
pd.crosstab(D1.Sex,D1.Survived)



### combining th condition  and combin 2 tables with a particualr condition 
pd.crosstab(D1.Sex,D1.Survived,normalize="index")
## normalize by index 
pd.crosstab(D1.Sex,D1.Survived,normalize="columns")
## normalise by column 
pd.crosstab(D1.Sex,D1.Survived,normalize="all")
##normalise everything 
age5=D1[D1['Age']<=5] 
age5.shape
D1[D1.Age<=5][['Survived','Pclass']]



age5[0:3] 
len(age5<5)
## to count the sum f the particular column with a condition 
D1[D1.Age<=5]['Survived'].value_counts() 
D1[D1.Name.str.contains("Allen")] 
D1.Embarked.unique()## to calculate the unique values 
D1['Age'].value_counts(dropna=False)## to see the nas as unique values 
D1['Embarked'].value_counts(dropna=False) 
D1[(D1.Age<=5)& (D1.Survived==1)][['Name','PassengerId']] 
D1[(D1.Age<=5)& (D1.Survived==1)][['Name','PassengerId']][0:10] 
D1.groupby('')['Age'].mean() 
D1.groupby('Survived')['Age'].mean()## to do some function in a catogorical column 
D1.groupby(['Pclass','Survived'])['Age'].mean() 
D1.groupby(['Pclass','Survived'])['Age'].sum() 
D1.groupby(['Pclass','Survived'])['Age'].mean().reset_index()## to do the mean and do rename the index 
#-------------------------------------
pd.read_excel('file.xlsx')
pd.to_excel('dir/myDataFrame.xlsx', sheet_name='Sheet1')   
#Read multiple sheets from the same file 
xlsx = pd.ExcelFile('file.xls') 
df = pd.read_excel(xlsx, 'Sheet1')
pd.read_csv('file.csv' ,header=None, nrows=5)
df.to_csv('myDataFrame.csv'
df[df[['Population']>1200000000]]


























































