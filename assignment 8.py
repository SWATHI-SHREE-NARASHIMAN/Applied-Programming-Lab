#importing modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Distance from src to a mic after reflecting through pt
def dist(src, pt, mic):
   d1 = ((pt[0]-src[0])**2+(pt[1]-src[1])**2)**(0.5)
   d2 = ((pt[0]-mic[0])**2+(pt[1]-mic[1])**2)**(0.5)
   return d1 + d2

def generate_mics(Nmics):
  mics=[]
  if(Nmics%2!=0):
      Nmics=Nmics-1
      mics=[(0,0)]
  for i in range(Nmics//2):
      coord=pitch*(i+1)
      mics.insert(0,(0,-coord))
      mics.append((0,coord))
  return mics

def sinc_generator(t,sincP):
    return np.sinc(sincP*t)

def amplitude(xv,yv,amp_array):
   d1=((xv-src[0])**2+(yv-src[1])**2)**0.5
   amp=np.zeros((len(xv),len(xv[0])))
   for k in range(Nmics):
     d2=((xv)**2+(yv-mics[k][1])**2)**0.5
     delay=(d1+d2)/C
     for m in range(len(delay)):
      for n in range(len(delay[0])):
        if(delay[m][n]<=Nsamp*dist_per_samp):
          index=int(delay[m][n]/dist_per_samp)
          amp[m,n]+=amp_array[k,index]
   return amp


Nmics = 64
Nsamp = 200
C = 0.5
src=(0,0)            #source coordinates
pitch=0.1
dist_per_samp=0.1
sincP=1
obstacle=(3,-1)

mics=generate_mics(Nmics)
t= np.linspace(0,Nsamp*dist_per_samp, Nsamp)
xv,yv=np.meshgrid(t,np.array(mics)[:,1])
amp_array=[]
for i in range(Nmics):
  distance=dist(src,obstacle,mics[i])
  t_delay=distance/C
  amp_array.append(sinc_generator(t-t_delay,sincP))
  plt.plot(t,sinc_generator(t-t_delay,sincP)+mics[i][1])
plt.title('Plot for amplitude')
plt.savefig('Initial amplitude plot')
plt.show()
print()
plt.imshow(amp_array)
plt.title('Heatmap for amplitude')
plt.savefig('Initial heatmap')
plt.show()
print()
plt.imshow(amplitude(xv,yv,np.array(amp_array)),cmap='viridis',aspect='equal')
plt.title('Heatmap for obstacles')
plt.savefig('Initial Obstacle')
plt.show()

C= 1.0
src=(0,0)            #source coordinates
pitch=0.1
dist_per_samp=0.1
sincP=5

f1= open('rx2.txt','r')
readf1=np.genfromtxt(f1)
f2=open('rx3.txt','r')
readf2=np.genfromtxt(f2)
Nmics1,Nsamp1=readf1.shape
Nmics2,Nsamp2=readf2.shape

mics1=generate_mics(Nmics1)
mics2=generate_mics(Nmics2)
t1=np.linspace(0,dist_per_samp*Nsamp1,Nsamp1)
t2=np.linspace(0,dist_per_samp*Nsamp2,Nsamp2)

for i in range(Nmics1):
  plt.plot(readf1[i]+mics1[i][1])
plt.title('plot for amplitude of dataset 2')
plt.savefig('Dataset2 plot')
plt.show()
print()

xv1,yv1=np.meshgrid(t1,np.array(mics1)[:,1])
plt.imshow(readf1,cmap='viridis',aspect='equal')
plt.title('Heatmap for amplitude - dataset2')
plt.savefig('Dataset2 heatmap')
plt.show()
print()
plt.imshow(amplitude(xv1,yv1,readf1),cmap='viridis',aspect='equal')
plt.title('Heatmap showing obstacles- dataset2')
plt.colorbar()
plt.savefig('Dataset2 obstacle')
plt.show()
print()

for i in range(Nmics2):
  plt.plot(readf2[i]+mics2[i][1])
plt.title('plot for amplitude of dataset 3')
plt.savefig('Dataset3 plot')
plt.show()
print()
xv2,yv2=np.meshgrid(t2,np.array(mics2)[:,1])
plt.imshow(readf2,cmap='viridis',aspect='equal')
plt.title('Heatmap for amplitude-dataset3')
plt.savefig('Dataset3 heatmap')
plt.show()
print()
plt.imshow(amplitude(xv2,yv2,readf2),cmap='viridis',aspect='equal')
plt.title('Heatmap showing obstacles - dataset3')
plt.colorbar()
plt.savefig('Dataset3 obstacles')
plt.show()
