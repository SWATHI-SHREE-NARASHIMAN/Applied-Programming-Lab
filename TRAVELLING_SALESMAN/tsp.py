import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import timeit

def distance_tot(cities, cityorder):                        #this function calculates the total distance travelled by the salesman for a given order
  cityorder=cityorder[:len(cities)]
  cityordernew=(cityorder+[cityorder[0]])[-len(cities):]                                                                                  #using the distance formula : distance= {(x2-x1)^2+(y2-y1)^2}^(0.5)
  return np.sum(np.sum((cities[cityorder]-cities[cityordernew] )**2,axis=1)**0.5)

Temp=5                                                      #Parameter for simulated annealing
decay_rate=0.8                                              #parameter for simulated annealing
fig, ax = plt.subplots()
plt.xlim([0,1])
plt.ylim([0,1])                                            #plotting variables for animation
lnall,  = ax.plot([], [], 'ro-')
xall, yall = [], []
init_state=[]                                               #this contains the initial prediction - the random guess order
state=[]


def find_next(i,next,state,cities):                        #This is the function that determines if the new guess is accepted or not
     #print(state)
     global Temp
     temporary=copy.deepcopy(state)
     index=state.index(next)
     temporary[index]=state[i+1]                           #temporary has the modified state to generate the new guess
     temporary[i+1]=state[index]
     if(distance_tot(cities,state)>distance_tot(cities,temporary)):
         return temporary                                  #if new guess is efficient it is accepted
     else:
      cost=distance_tot(cities,temporary)-distance_tot(cities,state)  #cost is the cost of accepting new guess over the old guess
      toss = np.random.random_sample()
      if(toss<np.exp(-cost/Temp)):                          #with a probability of e^(-cost/temp) the new guess is accepted - SIMULATED ANNEALING
           return temporary
     return state                                           #if new guess is not accepted the old guess is retained

def plot_fn(frame,state,cities):                            #plotting function for animation
  global ax,fig
  xall.append(cities[state[0]][0])
  yall.append(cities[state[0]][1])
  lnall.set_data(xall,yall)
  state.pop(0)

def tsp(cities):
  N=len(cities)
  global Temp,decay_rate,bestcost,ax,fig,init_state
  state=[i for i in range(N)]
  np.random.shuffle(state)                                  #this gives the random initial guess solution to the tsp problem
  state.append(state[0])
  init_state=copy.deepcopy(state)                           #starting guess os stored permanently
  cities=np.array(cities)
  epoch=1
  iterations=10
  initial_distance=distance_tot(cities,state)               #initial distance is stored for comparitive optimization
  print('started with distance:',initial_distance,'\norder=',state[:N] )
  for epoch in range(int(N)):                                    #this is a function of complexity O(N^3) which is still better than O(N!)
    start_distance=distance_tot(cities,state)
    for j in range(int(N)):
      for i in range(int(N)):                                   
        if(i!=N-1):
          next=np.random.choice(state[i+1:N])               #we are making a randomised exchange in positions for a given index i element and some element of index>i
        else:
          next=np.random.choice(state[1:N])                 #if it is the last city we take any other element and exchange
        if(next!=state[i+1]):
         state=find_next(i,next,state,cities)
      state[0]=state[-1]                                    #we ensure that the salesman returns to the starting city at the end
    new_distance=distance_tot(cities,state)
    improvement=(start_distance-new_distance)/start_distance*100
    print('epoch=',epoch,'percentage improvement=',improvement)  #measuring improvement in after every  N^2 runs
    Temp=Temp*decay_rate

  final_distance=distance_tot(cities,state)
  print('\nfinal_distance=',final_distance,'\nfinal order=',state[:N])                                                     #computing the final distance and improvement with respect to initial guess
  print('\npercentage_improvement=',(initial_distance-final_distance)/initial_distance*100)
  ani= FuncAnimation(fig,plot_fn,fargs=(copy.deepcopy(state),np.array(cities)), frames=range(N), interval=10, repeat=False)   #animating the final result 
  ani.save('travelling--salesman--solved2.gif',writer='pillow')
  return state                                               #returns the final order 

#This is a sample dataset created for testing and displaying the results
N=100
cities=[]
import csv
f=open('tsp40.txt','r')
reader=csv.reader(f,delimiter=' ')
count=0
for i in reader:
  if(count!=0):
    cities.append((float(i[0]),float(i[1])))
  count=1

#Displaying results
def time_fn():
  global state
  state=tsp(cities)     #for testing it on any particular list of cities, you can include the list in this line
print('total time taken=',timeit.timeit(time_fn,number=1))
cities=np.array(cities)
figure,axis= plt.subplots(nrows=1,ncols=2)
xplot = cities[:,0][init_state]
yplot = cities[:,1][init_state]
axis[0].plot(xplot, yplot, 'bo-')
axis[0].set_title('initial prediciton2')
xplot = cities[:,0][state]
yplot = cities[:,1][state]
axis[1].plot(xplot, yplot, 'ro-')
axis[1].set_title('final  prediction2')
plt.savefig('final order1.png')
