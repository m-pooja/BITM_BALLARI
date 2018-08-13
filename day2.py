# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:24:01 2018
DAY2, BITM, BALLARI
@author: pooja
"""
#LIST-----------------
l=[1,5,4,7,9,12]

l2=['u','s','b','t',5]
l[0]=34
l2.append(89)
l2.pop()
l2.pop(3)
l2.remove(5)
l2.reverse()
l=[1,2,[3,4],5,[3,6]]
l[2][1]

l.insert(3,'t')
len(l)
l=[1,7,5,9,3,0]
l.sort()
l=['a','t','g','u']
l=[1,2,3]
l1=[4,5,6]
l2=[7,8,9]
listnu=[l,l1,l2]
for i in listnu:
    print(i)
listnu[2][2]
#------------
#LIST COMPREHENSION
#--------------------
l=[5,8,6,7,9]
ln=[]
for i in l:
    ln.append(i)
lsq=[]
for i in l:
    lsq.append(i**2)    
lsq=[i**2 for i in l]  
le=[]  
for i in l:
    if i%2==0:
        le.append(i)
lens=[i for i in l if i%2==0]        
ln2=[i for i in l]
l*2
L = [ 19, True, 'Charts', 3.1459 ]
L[0] = True
L[1] = 24
L[3] = 'Books'
for i in L:
    print(type(i))
l1 = ['a', 'b', 'c']
l2 = ['d', 'e']
l1.extend(l2) 
l2.extend(l1)
print( l1)
l=[2,5,['g','t'],'r',56.90,True]
for i in l:
    print(type(i))
lt=[type(i) for i in l]
l = ['a','b','c',7,6,'r']
del l[1:4] 
print (l)
#temp in c to f
c=[25, 25.6,34,456,78,89]
ft=[9/5*i+32 for i in c]
sq2=[i**2 for i in [i**2 for i in range(10,21)]]
T = (1, 6, False, True, 'wow',35.3)
type(T)
t=(11,)
type(t)
t = (11, 22, 33)
t1 = t
t += (44,)
t[2]=45
print(t)
print(t1)
T= (11, 22, 33)
T=T+(11,)
T= (100,11, 22, 33)
print(sorted(T))
T=('e','j','i')
y=sorted(T)
a= "I belong to India"
at = tuple(a)
T= (100,11, 22, 11,33)
T.index(11)
at.index('n')
T.count(11)
at.count('n')
T= (100,11,(6,7),22,11,33)
tt=[type(i) for i in T]
T[2][1]
T[2:]
T.append(9)
--------------------
#Dictionaries
#--------------------
d={1:'one',2:'Two',3:'Three'}
d[3]
d[4]='Four'
d[1]
d.values()
d.keys()
d.items()
for i, j in d.items():
    print(i)
    print(j)
l='hello'
for i in l:
    print (i, end='|')
d1={1:123,2:[12,22,33], 
    3:['item1','item2','item3']}
d1[2][2]
print(d1[3][2])
d2={1:{'A':'Apple', 'B': 'Ball','C':{1:'Cam',2:'Car'}},2:123}
type(d2[1])
d2[1]['B'].upper()
d2[1]['C'][1]
d2[2]=d2[2]+5
#creating new key/value pairs
d[5]	= 'V'
print(d[5])
"""
get( INDEX, DEFAULT) 
"""
d.get(6,'None')
d2.get(2,'None')	
d.get(5, 'None') 	
d.get(500, 'NONE')	
d1.get('test', 'None')   
age = {'Alice' : 25, 'Carol': 'twenty-two'}
age.items()		
age.keys()		
age.values()	
age = {'Alice': 26 , 'Carol' : 22}
age.update({'Bob' : 29})	
age.update({'Carol' : 26})
age.update({'aman':35})
age.update({'aman':25})
age['aman']
A=age.copy()
A.update({'parul':78})
name=list(A.keys())
name.sort()
#print(cmp(A,age))
print(str(A))
d={1:'one',2:'two', 3:'three'}
d1={5:'five',3:'three',4:'four'}
for k in d.keys():
    if k in d1.keys():
        print('{} is there in d1'.format(k))
    else:
        print('{} is not in d1'.format(k))
#-----------------
#SETS
#-----------------
X  = set() 
X.add(1)
X.add(2)
X.add(3)
X.add(1)
print(X)
l=[1,8,1,4,8,9,5,1,2,5,6,8,476,86,16,4]
len(l)
s=set(l)
len(s)
print(s)
tweet='Proposals are invited for pre/post-conference tutorials/workshops. Tutorials/Workshops can be of half-day or full-day duration. The proposal should be presented in the form of a 200-word abstract, one page topical outline of the content, description of the proposers and their qualifications relating to the tutorial content.'
len(tweet)
lt=tweet.split()
len(lt)
ss=set(lt)
len(ss)
ss.pop()
ss.pop()
print(type(True))
bool(2)
a=90
b=89
print(a<b)
for r in range(7):
    for c in range(5):
        if (c==0 or c==4)or((r==0 or r==3 or r==6) and (c>0 and c<4)):
            print('@', end='')
        else:
            print(end=' ')
    print()
o=ord('a')-ord('a')
t=ord('c')-ord('a')
h=ord('e')-ord('a')
print(str(h)+str(t)+str(o))
0b111
print(str(0b100)+str(0b10)+str(0b000))
#------
#functions
#---------------
def myfun():
    print('I am a function')
myfun()
def myfun1(num=0):
    sq=num**2
    print('The number is {}'.format(num))
    print('The square of number is {}'.format(sq))
    return sq
myfun1()
s=myfun1(4)
def square(x):
    return x**2
square(4)
s=lambda x:x*2
s(9)
names=['Pooja','Aman','Manisha','Sophia','Shravan','Varun','Jerry']
rev=lambda s:s[::-1]
revn=[]
for i in names:
    revn.append(rev(i))
ren=list(map(rev,names))
c=[25, 25.6,34,456,78,89]
f=list(map(lambda x: 9/5*x+32, c))
a=[1,2,3]
b=[4,5,6]
c=[7,8,9]
l=[a,b,c]
s=list(map(lambda x,y:x+y, a,b))
s=list(map(lambda x,y,z:x+y-z, a,b,c))
l=list(zip(a,b))
d=dict(zip(a,b))
d={1:'a',2:'b'}
d1={3:'t',4:'u'}
dl=list(zip(d,d1.values()))
#filter--------------
def even(n):
    return n%2==0
l=[55,44,32,68,90]
for i in l:
    print(even(i))
lf=list(filter(even,l))
"""[44, 32, 68, 90]"""
l=[55,44,32,68,90]
l=['a','b','c']
for i, j in enumerate(l):
    print(i)
    print(j)
for i,j in enumerate(l):
    if i>1:
        break
    else:
        print(j)
m=['jun','jul','aug','sep','oct','nov','dec']
list(enumerate(m, start=6))
l=[False,False,0,0]
all(l)
any(l)
l=[6,5,4,67,8,9990,7]
sum(l)
import math
math.ceil(23.45)
import math as m
m.sqrt(99)
m.ceil(9090.32)
from math import sqrt
sqrt(89)
for i in l:
    print (sqrt(i))


"""
WAP to create a function which 
takes a sentence as an input 
if any letter in it is 'A' or 'a' 
convert it to 'X'. 
Return the updated sentence.
"""
def func2(sen):
    for c in sen:
        if c=='A' or c=='a':
            print('X',end='')
        else:
            print(c, end='')
func2('Return the updated sentence')
"""
Displays which Letters are in the 
First String but not in the Second
"""
s1=set(['a','b','t'])
s2=set(['a','c','b'])
s2-s1
s1=input("Enter first string:")
s2=input("Enter second string:")
a=list(set(s1)-set(s2))
"""
Remove the nth Index 
Character from a Non-Empty String

input==='hello world'
n=6
output==='hello orld'
"""
def remove(string, n):  
      first = string[:n]   
      last = string[n+1:]  
      return first + last     
 remove('this is python class',5)     





























































































    
    
    
    
    
    
    
    




















