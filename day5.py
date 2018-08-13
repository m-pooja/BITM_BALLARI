# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:30:32 2018
DAY 5, BITM, BALLARI
@author: pooja
"""
#------drinks data----------------------------------
#import required libraries
import pandas as pd
import numpy as np
#import the dataset of drinks
data=pd.read_csv('drinks.csv')
data.shape
data.head()
col=list(data.columns)
data.describe()
data.index
#which continent drinks more beer on average
data.groupby(['continent'])['beer_servings'].mean().idxmax()
#for each continent print the statistics for wine consumption
data.groupby('continent').wine_servings.describe()
#Print the mean alcoohol consumption per continent for every column
data.groupby('continent').mean()
#Print the median alcoohol consumption per continent for every column
data.groupby('continent').median()
#Print the mean, min and max values for 
#spirit consumption(output a dataframe).
#Print the mean, min and max values for spirit consumption(output a dataframe).
agg_data=data.groupby('continent').spirit_servings.agg(['mean', 'min', 'max'])
data.spirit_servings.hist(bins=6)
#-------------------------------------------
"""
Use pandas to read the file (titanics).
"""
import pandas as pd
D1=pd.read_csv('TST.csv')
"""
Use groupby method to calculate
the proportion of passengers that
survived by gender.
"""
D1.groupby('Sex').Survived.sum()
"""
Use groupby method to calculate
the proportion of passengers that died by gender.
"""
len(D1[D1['Survived']==0][D1['Sex']=='male'])
len(D1[D1['Survived']==0][D1['Sex']=='female'])
"""
Calculate the same proportion(survived) but by 
'Pclass and gender'.
"""
D1.groupby(['Sex', 'Pclass']).Survived.mean()
"""
create age categories:-
  childern(under 14yrs)
  adolescents(14-20yrs)
  adults(21-64 yrs)
  senior(65+ yrs)
and calculate survival proportion
by age category, Pclass and gender.
"""
#age=age.dropna()
Bins = [0, 14, 20, 64, 80]
BinLabels = ['under14','adolescents', 'adult','senior']
pd.cut(D1['Age'],Bins,labels=BinLabels)
#checking if there 
#are any nan values
np.isnan(D1['Age'])
#filling nana values
age=D1['Age'].fillna(int(D1['Age'].mean()), inplace=True)
pd.cut(D1['Age'],Bins,labels=BinLabels)
np.isnan(age)

#----------------------------------
#Import the necessary libraries
import numpy as np
import pandas as pd
#Import the dataset and assign to a variable
data1=pd.read_csv('student-mat.csv',  sep=';')
#Display top 5 rows of data
data1.head()
data1.shape
col=list(data1.columns)
#For the purpose of this exercise slice the dataframe from 'school' until the 'guardian' column
stud_alcoh = data1.iloc[: , :12]
stud_alcoh=data1.loc[:,'school':'guardian']
data1.column
data1.info()
stud_alcoh.head()
#Create a lambda function that captalize strings.
c = lambda x: x.upper()
#Capitalize both Mjob and Fjob
stud_alcoh['Mjob'].apply(c)
stud_alcoh['Fjob'].apply(c)
#Print the last elements of the data set.
stud_alcoh.tail
#Did you notice the original dataframe is still lowercase? Why is that?
# Fix it and captalize Mjob and Fjob.
stud_alcoh['Mjob'] = stud_alcoh['Mjob'].apply(c)
stud_alcoh['Fjob'] = stud_alcoh['Fjob'].apply(c)
stud_alcoh.tail()


stud_alcoh['newcol']=np.arange(0,395)
del(stud_alcoh['newcol'])
"""
Create a function called majority that
 return 
a boolean value to a new 
column called legal_drinker
(Consider majority as older than 
 17 years old)
"""
def majority(x):
    if x > 17:
        return True
    else:
        return False
stud_alcoh['legal_drinker'] = stud_alcoh['age'].apply(majority)
#try to do this with .cut method!
stud_alcoh['legal_drinker']=pd.cut(stud_alcoh['age'], [0, 17,22], labels=[False, True])
stud_alcoh.head()
"""
Multiply every number of the dataset by 10.
"""
def times10(x):
    if type(x) is int:
        return 10 * x
    return x

stud_alcoh.applymap(times10).head(10)
stud_alcoh.set_index(stud_alcoh['famsize'])
#joining, merging dataframes------------------
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

# ## Concatenation
# 
# Concatenation basically glues together DataFrames. Keep in mind that dimensions should match along the axis you are concatenating on. You can use **pd.concat** and pass in a list of DataFrames to concatenate together:


pd.concat([df1,df2,df3])


pd.concat([df1,df2,df3],axis=1)

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})    

# ## Merging
# 
# The **merge** function allows you to merge DataFrames together using a similar logic as merging SQL Tables together. For example:
pd.merge(left,right,how='outer',on='key')

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})
pd.merge(left, right, on=['key1', 'key2'])
pd.merge(left, right, how='outer', on=['key1', 'key2'])
pd.merge(left, right, how='right', on=['key1', 'key2'])
pd.merge(left, right, how='left', on=['key1', 'key2'])
# ## Joining
# Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame.
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])
left.join(right)
left.join(right, how='outer')
#---------------------------------------------------
#data visualization
#---------------------------------------------------
"""
The basic steps to creating 
plots with matplotlib are:              
1 Prepare data     
2 Create plot     
3 Plot     
4 Customize plot    
5 Save plot    
6 Show plot
"""
import matplotlib.pyplot as plt
import numpy as np
#genrate 1D data
x = np.linspace(0, 10, 100)
y = np.cos(x) 
z = np.sin(x)
#genrate 2D data
data = 2 * np.random.random((10, 10)) 
data2 = 3 * np.random.random((10, 10)) 
#from matplotlib.cbook import get_sample_data 
#img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))
#im = ax.imshow(img,cmap='gist_earth',interpolation='nearest',vmin=-2,vmax=2)
gdp_cap=[]
for i in range(50):
    n=np.random.randint(155, 389)
    n=n/100
    gdp_cap.append(n)
life_exp = []
for i in range(50):
    n=np.random.randint(100, 150)
    n=n/100
    life_exp.append(n)
pop = []
for i in range(50):
    n=np.random.randint(190, 230)
    n=n/100
    pop.append(n)       
import matplotlib.pyplot as plt
plt.plot(x,z)
"""
histogram-----------------------------------
"""
plt.hist(y)
plt.hist(pop, bins= 5, orientation='horizontal')
plt.cla()               #Clear an axis >>> 
plt.clf()               #Clear the entire figure >>> 
plt.close()  

plt.xlabel('my data')
plt.ylabel('your data')
plt.xticks([0, 1, 2,2.3])
plt.yticks([0,2,4,8])           #Close a window
plt.hist(pop, color='yellow')
#plt.show()
plt.savefig('hist.png')
plt.savefig('hist1.png', transparent=True)   
plt.xscale('log')
plt.yscale('log')
plt.hist(pop, orientation='horizontal')# orientation changes the look of plot try it with 'vertical'

"""
Scatter ------------------------------------------------------
"""
"""
import numpy as np
x= np.random.rand(5)
y=np.random.rand(5)
s1= [10,20,80,40,50]
c=[3,3,5,6,7]
c1=np.random.rand(5)
"""
plt.scatter(x,y)
plt.scatter(x,y, s=s1, c= c,cmap='rainbow')
plt.scatter(gdp_cap, life_exp, s =pop)
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])
plt.legend(["curve1"])
#---------------------------
#pie chart
#---------------------
l = 'Python', 'C++', 'Ruby', 'Java'
sizes = [310, 30, 145, 110]
col = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
exp = (0.1, 0.1, 0.1, 0.1)  # explode 1st slice
plt.pie(sizes, labels=l, colors=col, explode=exp, startangle=50)
plt.pie(sizes, explode=exp, labels=l, colors=col,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.tight_layout()
plt.axis('equal')
plt.show()
#------------------
#bar plot
x = np.arange(4)
money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]
a,b,c,d=plt.bar(x, money)
a.set_facecolor('r')
b.set_facecolor('g')
c.set_facecolor('b')
d.set_facecolor('black')
plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
plt.show()
plt.barh(x, money)
import pandas as pd
df2 = pd.DataFrame(np.random.rand(10, 4), 
                   columns=['a', 'b', 'c', 'd'])
df2.plot.bar()
df2.plot.bar(stacked=True)
"""
box plot:
    The box plot (a.k.a. box and whisker diagram) is a 
    standardized way of displaying the distribution of 
    data based on the five number summary: minimum, 
    first quartile, median, third quartile, and maximum.
    In the simplest box plot the central rectangle spans
    the first quartile to the third quartile 
    (the interquartile range or IQR). 
    A segment inside
    the rectangle shows the median and 
    "whiskers" 
    above and below the box show the locations of the 
    minimum and maximum.
If the data happens to be normally distributed,
IQR = 1.35 σ
where σ is the population standard deviation.
"""
df = pd.DataFrame(np.random.rand(10, 5),
                  columns=['A', 'B', 'C', 'D', 'E'])
df.loc[3,'B']=1
color = dict(boxes='DarkGreen', 
             whiskers='DarkOrange', 
             medians='DarkBlue', caps='Gray')
df.plot.box(color=color, sym='r+')
plt.boxplot(df['A'])
s=np.random.rand(50)*100
c=np.ones(25)*50
fh=np.random.randn(10)*100+100
fl=np.random.randn(10)*-100
dd=np.concatenate((s,c,fh,fl),0)
plt.boxplot(dd)
#--------------------------------
from pandas.tools.plotting import parallel_coordinates
from sklearn.datasets import load_iris
data=load_iris()
dataframe=pd.DataFrame(data.data,columns=data.feature_names)
dataframe['class']=data.target
dataframe.shape
plt.figure()
parallel_coordinates(dataframe , 'class')
dataframe.groupby('class').size()
print(dd.groupby('class').size())
plt.savefig('pp.png')
df.plot.area()
dataframe.plot.area()

#---------------------
#date-time module
d1 = "10/24/2017"
d2 = "11/24/2016"
max(d1,d2)
d1 - d2
import datetime
d1 = datetime.date(2016,11,24)
d2 = datetime.date(2017,10,24)
max(d1,d2)
print(d2 - d1)
century_start = datetime.date(2000,1,1)
today = datetime.date.today()
print(century_start,today)
print("We are",today-century_start,"days into this century")

century_start = datetime.datetime(2000,1,1,0,0,0)
time_now = datetime.datetime.now()


time_since_century_start = time_now - century_start
print("days since century start",
      time_since_century_start.days)
print("seconds since century start",
      time_since_century_start.total_seconds())
print("minutes since century start",
      time_since_century_start.total_seconds()/60)
print("hours since century start",
      time_since_century_start.total_seconds()/60/60)
dtn = datetime.datetime.now()
tn = dtn.time()
print(tn)
today=datetime.date.today()
fdl=today+datetime.timedelta(days=5, minutes=89, seconds=9)
print(fdl)
#--------------feature scalling
from sklearn.datasets import load_iris
data = load_iris()
import pandas as pd
from sklearn import preprocessing
iris = load_iris()
print(iris.data.shape)

# separate the data from the target attributes
X = iris.data
y = iris.target

normalized_X = pd.DataFrame(preprocessing.normalize(X), columns=iris.feature_names)
normalized_X['class']=iris.target
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(normalized_X,'class')
standardized_X = preprocessing.scale(X)
































































































