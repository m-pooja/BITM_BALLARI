# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:08:53 2018
BITM, BALLARI
@author: pooja
"""
x=3.5
x='s'
a=2+3
s=2-3
m=2*3
s=2**4
d=2/3
y=0.1+0.9


x=int(input('Enter the number'))
y=int(input('Enter second number'))
print(x+y)
print(x%y)
s='This is a beautiful day'
s[5:-3]
s[5:-4]
s[6:23]
s[-13:-4]
s[:7]
s[7:]
s[:]
s[:23:3]
s[::-1]
s1='Hello'
s2="I'm Pooja"
s3=s1+'! '+s2
print(s3)

print(s3[2])

#s3[2]='o'#cannot be done
#-----------in operator-----
'e' in s1
s1*2
#------------Loop--

for i in s1:
    print(i)
print(s3)

"""
WAP to display 'is there' 
and 'no-more'
if a character 
is there in a string or not.
user will provide a string
user will provide the 
character
"""
s=input("Enter the sttring")
c=input("Enter the letter")
if c.lower() in s.lower():
    print('is there')
else:
    print('no more')

"""
WAP to display an input string  like this:-
'computer'
r				
er			 
ter				
uter				
puter				
mputer			
omputer			
computer			
"""  
string = "Computer"
for i in range(len(string)+1):
    print(string[0:i])

string = "Computer"
for i in range(len(string)+1):
    print(string[i:])
 
    
    
    
 
    
    
    
    
    
"""
Ask the user to input a 
word in English. 
Make sure the user 
entered a valid word. 
Convert the word from English like this:-
Input : SUKUN
Output : UKUNSPK
[(2nd alphabet to end) + 1st alphabet + ‘PK’]. Display the translated result.
"""
a = input (" ")
if a.isalpha():
    #a=a.upper()
    a1=a[1:] + a[0]+'PK'
    print (a1.upper())
else:
    print('invalid word')
    
l=s3.split()

ss=''.join(l)
"""
input a string, go through it, and if the length of a word is even, print 'Even' else print 'odd'.
"""
s= input ("Enter the string")
l=s.split()
for i in range(len(l)):
    if len(i)%2==0:
        print("The word '{}' is Even".format(i))
    else:
        print('odd')
        
"""
WAP that prints the integers from 1to100. 
 but for multiples of 3 print 'FUZZ' and 
 multiples of 5 print 'BUZZ' for numbers 
 which are 
multiple of both 3 and 5 print'FIZZBUZZ'.
"""        
x=2
print('the value of x is ', str(x))

s=9
#.format
print('Insert one string : {}'.format(s))

students = 'Forty'
print('There are %s students in the class.' % students  )
students = 42
print('There are %d students in the class.' % students)
l=[1,5,4,7,9,12]

l2=['u','s','b','t',5]
print(min(l))
print(min(l2))
print(l2[:3])
l[0]=34
l2.append(89)
l2.pop()
l2.pop(3)
l2.remove(5)
l2.reverse()





s=input("Enter the sentence")
#s='this is not fair to sart working today'
for i in s.split():
    if i.startswith('s'):
        print('True')
    else:
        print('False')









































