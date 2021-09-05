#code written by Jan Zimbelmann
#it is required to have the following files:
#1. a file for the ising simulation for magnetization at different temperatures
#with the name './confiugrations/McL<L>.csv'
#2. a file for the decimated ising simulation for magnetization at different temperatures
#with the name './confiugrations/RgL<L>.csv'
#the terms in the angle brackets refere to numeric variables, with:
#'L' is the system length, here 16
######

#importing librariers
######
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

#initializing variables
######
L=16 #system length
steps = 5 #super resolution steps
depth = 21 #order of the polynomial regression

temperature_range_min = 1.5 #starting temperature of the saved temperature transformation list
temperature_range_max = 3.0 #final temperature of the saved temperature transformation list
temperature_range_size= 16  #+2 ...
#two different ways of calculating the temperatures is added later on


#load magnetization of the original/decimated
######

folder = 'numerical/' #folder of the ising simulations for this numerical solution

mcName = 'McL'+str(L)+'.csv' #monte carlo observables
mcM = np.loadtxt(folder + mcName, delimiter =',')[:,1]

rgName = 'RgL'+str(L)+'.csv' #decimated monte carlo obvseables
rgM = np.loadtxt(folder + rgName, delimiter =',')[:,1]

T = np.loadtxt(folder + mcName, delimiter =',')[:,0] #temperature
data_points = len(mcM) #size of different temperature data points

#regression of the data points for the decimated and original magnetization
######
mcY = mcM
mcX = T
mcP = P.fit(mcX,mcY,depth)

rgY = rgM
rgX = T
rgP = P.fit(rgX,rgY,depth)

#tighter array for plotting the solution of the regression
tightT = np.linspace(T[0],T[-1],101) 

#calculation of the critical temperature
######
test = []
iterations=100000
for i in range(iterations):
    tc = 2.2 + (i/iterations)
    test.append(abs(rgP(tc)-mcP(tc)))
intersect = np.array(test).argmin()
#2.26174
TC = 2.2 + (intersect/iterations)
realTC = 2.26918531421 # the critical temperature at infinite system size for 2D

#printing some information on the crititcal temperature
print("The critical temperature is calculated at", TC)
print("The critical temperature at inifinite system size is known to be", realTC)

#regression 
######
#create a list of intially equally distributed temperature points
step_size = (temperature_range_max-temperature_range_min)/(temperature_range_size-1)
saveList = np.arange(temperature_range_min, temperature_range_max+0.0001, step_size)
#adding the calculated and infinite sized critical temperatures to the list
index = np.where(saveList>TC)[0][0]
saveList = np.insert(saveList, index,TC)
index = np.where(saveList>realTC)[0][0]
saveList = np.insert(saveList, index,realTC)
#size of the later to be saved temperature transformations
savePoints = len(saveList)

#create the arrays for the regressions of the temperatures and magnetizations
aT = np.zeros((steps+1,savePoints))
aM = np.zeros((steps+1,savePoints))
aT[0] = saveList
aM[0] = mcP(saveList)

#performing the regression for a certain amount of super resolutions steps
for step in range(0,steps):
    for i in range(savePoints):
        newT = (rgP - aM[step][i]).roots()
        newT = [a for a in newT if a.imag == 0]
        newT = np.real(newT[0])
        aT[step+1][i] = newT
        aM[step+1][i] = mcP(newT)

#saving the numerical solution of the temperatures for a <steps> amount of super resolution steps
np.savetxt('transformT.csv',aT,delimiter = ",",fmt='%g')

#creating the plots for varifying the regressions
print('Please rerun the simultion with a different depth value if something does not look appriopriate.')
N = (L/2)**2
plt.scatter(T, mcM/N, label = "MC Simulation")
plt.scatter(T, rgM/N, label = "Decimated MC Simulation")
plt.plot(tightT, mcP(tightT)/N, label = "Monte Carlo (MC) Regression")
plt.plot(tightT, rgP(tightT)/N, label = "Decimated MC Regression")
plt.vlines(temperature_range_min,0,1.1,colors='red',linestyles = 'dashed',label = 'Temperature Range')
plt.vlines(temperature_range_max,0,1.1,colors='red',linestyles = 'dashed')
plt.grid(color='0.88', linestyle = '--', linewidth = 2)
plt.xlabel("T",fontsize = 18.5)
plt.ylabel("|m|", fontsize = 18.5)
plt.legend(fontsize=15)
plt.title("L="+str(int(L/2)),fontsize=19)
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylim(0,1.1)
plt.tight_layout()
plt.legend()
#plt.savefig("transformationT",dpi=200)
plt.show()
