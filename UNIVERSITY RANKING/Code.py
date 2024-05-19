import csv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import curve_fit

#we airm at doing logistic regression in the model
#let's analyse the probability of getting into the university and try to set a threshold for classifying each data object into 
'ADMIT' and 'REJECT'
#assuming a nominal acceptance rate of 30% which around 0.8 on probability
def step(i):
  return i//0.8

def prediction_function(X,g,t,u,s,l,c,r):                                                       #assuming a polynomial power relation
  GRE,TOEFEL,UNIRANK,SOP,LOR,CGPA,RESEARCH=X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7]
  value=(GRE**g)*(TOEFEL**t)*(UNIRANK**u)*(SOP**s)*(LOR**l)*(CGPA**c)+(r*RESEARCH)              #Research just adds value to the resume (boolean value)
  return value

f=open('Admission_Predict_Ver1.1.csv','r')
reader=csv.DictReader(f)
data=np.genfromtxt('Admission_Predict_Ver1.1.csv',delimiter=',',skip_header=1)

#normalising parameters
data[:,1]=data[:,1]/340
data[:,2]=data[:,2]/120
data[:,3]=data[:,3]/5
data[:,4]=data[:,4]/5
data[:,5]=data[:,5]/5
data[:,6]=data[:,6]/10


#splitting dataset into train and test
train=data[0:400,:]
test=data[400:,:]
(g,t,u,s,l,c,r),_= curve_fit(prediction_function, train,train[:,8],maxfev=10000)               #curve fit prediction
print('POWER RELATIONS:\nGRE:',g,'\nTOEFEL:',t,'\nUNIVERSITY RANK:',u,'\nSOP:',s,'\nLOR:',l,'\nCGPA:',c,'\nRESEARCH:',r)


#analysing predicitons
value_prediction=prediction_function(test,g,t,u,s,l,c,r)
print('standard deviation=',np.std(value_prediction-test[:,8])) #standard deviation calculated from error function
accuracy=0
for i in range(len(value_prediction)):
  if step(value_prediction[i])==step(test[i][8]):               #step function is used to classify a person into 'ADMIT', 'REJECT' scheme
    accuracy+=1
print('accuracy=',accuracy)

GRE,TOEFEL,UNIRANK,SOP,LOR,CGPA,RESEARCH=0.8,0.8,0.4,0.6,0.6,0.9,1    #fixing parameters for varying one particular parameter alone
#Let's analyse which among GRE,TOEFEL AND CGPA IS MORE IMPORTANT, keeping other factors consatant
def analysis_1():
    X=np.linspace(0,1,100)                                            #samplind data points for X axis
    Y=(X**g)*(TOEFEL**t)*(UNIRANK**u)*(SOP**s)*(LOR**l)*(CGPA**c)     #GRE PREDICTION
    plt.plot(X,Y,color='cyan')
    Y=(GRE**g)*(X**t)*(UNIRANK**u)*(SOP**s)*(LOR**l)*(CGPA**c)        #TOEFEL PREDICTION
    plt.plot(X,Y,color='r')
    Y=(GRE**g)*(TOEFEL**t)*(UNIRANK**u)*(SOP**s)*(LOR**l)*(X**c)      #CGPA PREDICTION
    plt.plot(X,Y,color='g')
    plt.axvline(0.7)
    plt.legend(['gre','toefel','cgpa'],loc='lower right')
    plt.savefig('analysis_1.png')
    plt.clf()
analysis_1()

#how scores vary as university rank changes
def analysis_2():
    X=np.linspace(0,1,100)
    UNIRANK=0.6                                                      #HIGHER SCORE HERE MEANS LOWER UNIVERSITY
    Y=(X**g)*(TOEFEL**t)*(UNIRANK**u)*(SOP**s)*(LOR**l)*(CGPA**c)
    plt.plot(X,Y,color='cyan')
    UNIRANK=0.2                                                       #LOWER SCORE HERE MEANS BETTER UNIVERSITY
    Y=(X**g)*(TOEFEL**t)*(UNIRANK**u)*(SOP**s)*(LOR**l)*(CGPA**c)
    plt.plot(X,Y,color='r')
    plt.legend(['lower ranking','top ranking'],loc='lower right')
    plt.savefig('analysis_2.png')
    plt.clf()
analysis_2()

#SOP VS LOR
def analysis_3():
  X=np.linspace(0,1,5,0.2)
  Y=(GRE**g)*(TOEFEL**t)*(UNIRANK**u)*(SOP**s)*(X**l)*(CGPA**c)     #VARYING LOR
  plt.plot(X,Y,color='b')
  Y=(GRE**g)*(TOEFEL**t)*(UNIRANK**u)*(X**s)*(LOR**l)*(CGPA**c)     #VARYING SOP
  plt.plot(X,Y,color='r')
  plt.legend(['lor','sop'],loc='lower right')
  plt.savefig('analysis_3.png')
  plt.clf()
analysis_3()

#HOW IMPORTANT IS SOP FOR A GIVEN UNIVERSITY RANK
def analysis_4():
  X=np.linspace(0,1,5,0.2)
  UNIRANK=0.2
  Y=(GRE**g)*(TOEFEL**t)*(UNIRANK**u)*(X**s)*(LOR**l)*(CGPA**c)
  plt.plot(X,Y,color='b')
  UNIRANK=0.4
  Y=(GRE**g)*(TOEFEL**t)*(UNIRANK**u)*(X**s)*(LOR**l)*(CGPA**c)
  plt.plot(X,Y,color='r')
  plt.axvline(0.8)
  plt.legend(['top ranking','lower ranking'],loc='lower right')
  plt.savefig('analysis_4.png')
  plt.clf()
analysis_4()


