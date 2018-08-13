# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:48:32 2018
DAY3, BITM, BALLARI
@author: pooja
"""
#File handling-----------
f=open ('day3_bitm.txt', 'a')
f.write('This is supposed to be a new day in history.\nWe are from different places.')
f.close()

f=open ('day3_bitm.txt', 'r')
f.readline()
f.close()
f.seek(0)
f.readlines()


f=open ('day3_bitm.txt', 'a')
f.writelines(['\nabcd\n','efgh\n','hijk'])
f.close()

f=open('a1.txt', 'r')
f.read()
f.close()
with open('a1.txt', 'r') as f:
    for l in f.readlines():
        print(l)


#-----------------------------------CLASSES--------------------

class Student1():     
    Sclass='CA'  

s=Student1()    
s.Sclass 
#--------------------------------------------------   
class Student():     
    #university="Punjabi University"     
    def __init__(self, name,class_s,rollnumber,news):#userdefined attributes         
        self.name=name         
        self.class_s=class_s         
        self.rollnumber=rollnumber         #wat it to be a boolean attribute         
        self.news=news 
stud1=Student('Pooja', 'CS', 144920,False)

stud2=Student()
#---------------------------------------------        
class Student():     
    university="BITM"     
    def __init__(self, name,class_s,rollnumber,news):#userdefined attributes         
        self.name=name         
        self.class_s=class_s         
        self.rollnumber=rollnumber         #wat it to be a boolean attribute         
        self.news=news
        
    def message(self):         
        if self.news==False:             
            print("{} !Welcome back at {}!".format(self.name,self.university))         
        else: 
            print("{}! welcome to the class of {} at {}.".
                  format(self.name, self.class_s, Student.university))  
    
stud3=Student('Nivedita', 'ECE', 12345, False)       
stud3.message()
print(stud3.rollnumber)
stud4=Student('Keertana', 'CSE', 7890, True)
print(stud3.name+' ' +stud4.name)

#--------------------------------------
class Animal():
    #main class     
    def __init__(self):  
        #self.name=name
        self.name=input('Whats ur name: ')
        print('My name is {}'.format(self.name)) 
        #cat=Animal()         
    def eat(self):        
        print('I need food')              
    def walk(self):         
        print("I want to go out on walk!")  

a=Animal()
a.eat()
a.walk()
b=Animal('Brown')
b.eat()
b.walk()
class cat(Animal):     
    def __init__(self): 
        #self.name=name
        Animal.__init__(self)
        print("I'm a little kitty")
    def eat(self):         
        print("I eat tuna!")#overwite 
    def sleep(self):         
        print("I sleep alot!")  
a=Animal('Kitty')
catty=cat()    
catty.eat()
catty.walk()
#------------------------------------------------
len(a)
print(a)

#special methos dunders 
class Book:     
    def __init__(self, title, author, pages):         
        print("A book is created")         
        self.title = title         
        self.author = author         
        self.pages = pages  
    def __str__(self):         
        return "Title: %s, author: %s, pages: %s" %(self.title, self.author, self.pages)  
    def __len__(self):         
        return self.pages  
    def __del__(self):         
        print("A book is destroyed") 
b=Book('Python','Pooja',180) 
print(b) 
print(len(b)) 






    
    
    
    
    
    
    
    
    
    
    
    


























