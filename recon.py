#code written by Jan Zimbelmann
#it is required to have the following files:
#1. a file for the numeric results of the temperature transformation
#with the name './transformT.csv'
#2. a set of original and decimated spin configurations
#original name: './configurations/z<step>Mc<it>L<L>.csv'
#the terms in the angle brackets refere to numeric variables
#'step' referes to the super resolution step
#'it' referes to the iteration index pointing to the transformT.csv temperature
#'L' is the system length, here 16
#spin configurations are stored as 0 and 1, not -1 and 1
#3. a set of stored weights and biases
#checkpoint path: './checkpoints/z<step>T<it>cp<num>.ckpt'
#'num' is the repitition index for running multiple CNNs
#additionally a file is required pointing at the best
#'num' checkpoint data file
#this file is originally saved as './recon.py'
######

#importing libraries
######
import os
import os.path
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from tensorflow.python.framework import ops
import random
from tensorflow.keras.callbacks import EarlyStopping
tf.keras.backend.set_floatx('float64')
import time
import argparse

#initializing variables
######
#parsing the iteration index pointing in the transformT.csv
parser = argparse.ArgumentParser(description='Specifying which data point to super resolve.')
parser.add_argument('iteration', type=int, nargs='?', default = 0)
it = (parser.parse_args()).iteration

repeat_output = 100000 #the amount of statistics used for calculating the observable
L=16 #system length
N=L*L #number of spins
dist = 3 #distance for two point spin correlation function
T = np.array(np.loadtxt("transformT.csv", delimiter =',')) #temperature points
data_points = len(T[0]) #amount of all temperature points

steps=2 #how many super resolution steps are to be calculated
best_of = 9 #how often the training had been repeated in the learn.py file
#padding configurations

#the following variables are not actively used but are kept for recycling
#the CNN models from the previous learn.py code
lr = 1e-3 #learning rate
loss_factor=2e-8

#CNN setup conditions
padding_sizes = [5,5,5]
padding_size = len(padding_sizes)
padding_sum = sum(padding_sizes)-padding_size 
print(padding_sum)

#folders
folder_cp = 'checkpoints/' #weights and biases saving folder
folder_mc = 'configurations/' #configurations folder
folder_nn = 'reconstructions/' #folder for the CNN reconstruction data
os.makedirs(folder_nn[:-1], exist_ok=True)
folder_hi = 'nn_histogram/' #solutions for higher steps
os.makedirs(folder_hi[:-1], exist_ok=True)

#path for the saved weights and biases
checkpoint_path = [[ [None for z  in range(best_of)] for y in range(data_points) ] for x in range(steps+1)]
for step in range(steps+1):
    for i in range(data_points):
        for j in range(best_of):
            checkpoint_path[step][i][j] = folder_cp + "z" + str(step) + "T"+ str(i) +"cp" + str(j) + ".ckpt"

#printing some information on the variables
print("the amount of different spin configurations super resolved is:", repeat_output)
print("best_of variable is set to:", best_of)

#defining functions, no additional functions to 'learn.py' is used
######
#implementing the regularization term
def regularization_term(y_true, y_pred):
    #spin input is set from 0..1 to -1..+1 
    y_true_new = (y_true*2)-1
    y_pred_new = (y_pred*2)-1
    #indexing for the nearest neighbor interaction of the energy
    idx = (np.array(list(range(L)))+1)%L
    #calculating the differences in the regularization term
    reg  = y_true_new * tf.gather(y_true_new, idx, axis=1)
    reg += y_true_new * tf.gather(y_true_new, idx, axis=2)
    reg -= y_pred_new * tf.gather(y_pred_new, idx, axis=1)
    reg -= y_pred_new * tf.gather(y_pred_new, idx, axis=2)
    reg = tf.reduce_sum(reg,axis=2)
    reg = tf.reduce_sum(reg,axis=1)
    #calculating the square in the regularization term
    reg = tf.square(reg)
    #summing over the entire mini-batch
    reg = tf.reduce_sum(reg)
    #returning the regularization term with a proportionality factor
    return p_f * reg

#adding the regularization term to the cross-entropy loss
def my_custom_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + regularization_term(y_true, y_pred)

#initiate the neural network model
def create_model(inp):
    #initialize a sequential model
    model = models.Sequential()
    #setting the elu activation function for all layers other than the last
    total_padding = padding_sum
    for a in range(padding_size-1):
        model.add(layers.Conv2D(1, padding_sizes[a], activation= lambda x : tf.nn.elu(x), input_shape=[inp+total_padding,inp+total_padding,1]))
        total_padding -= padding_sizes[a]-1
    #setting the sigmoid activation function for the final layer
    model.add(layers.Conv2D(1, padding_sizes[-1], activation='sigmoid', input_shape=[inp+total_padding,inp+total_padding,1]))
    #compile and return
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr),loss=my_custom_loss)
    return model

#pbc padding construction
#a pbc cut condition required for the next function
def cut(number,length):
    if(number<length):
        return number
    else:
        return length

def pbc_padding(array, p_size):
    #conditios in place for if the pbc padding is larger than
    #the spin configurations and applying them appropriately 
    output = np.array(array)
    #in the first dimension
    new_size=p_size
    [cfgs, lng, lng] = output.shape
    column = np.empty([cfgs,lng,0])
    for i in range(int((new_size/lng)-1e-50)+1):
        column = np.append(column,output[:,:,:cut(new_size,lng)],axis=2)
        new_size -= lng
    output = np.append(output, column, axis=2)
    #in the second dimension
    new_size= p_size
    row = np.empty([cfgs,0,lng+new_size])
    for i in range(int((new_size/lng)-1e-50)+1):
        row = np.append(row,output[:,:cut(new_size,lng),:],axis=1)
        new_size -= lng
    output = np.append(output, row, axis=1)
    return output

#calculating observables
def getM(array): #absolute magnetization
    m = 0
    for i in array:
        for j in i:
            m += (j*2)-1
    return abs(m)

def getE(array): #energy
    e = 0
    size = len(array)
    for i in range(size):
        for j in range(size):
            e += ((array[i][j]*2)-1) * ((array[(i+1)%size][j]*2)-1)
            e += ((array[i][j]*2)-1) * ((array[i][(j+1)%size]*2)-1)
    return -e

def getG(array, dist): #two point spin correlation function
    g = 0
    amount = 0
    size = len(array)
    for i in range(size):
        for j in range(size):
            g += (array[i][j]*2-1)*(array[(i+dist)%size][j]*2-1)
            g += (array[i][j]*2-1)*(array[i][(j+dist)%size]*2-1)
    return g/(2*size*size)

#loading a csv file and saving the file at a specific condition of this file
#if condition is not fulfilled, create a new data point with this conditions
def savingData(array, condition, name):
    #check if file exists
    #if yes, load file as array
    #otherwise create new array stored later as file
    if(os.path.isfile(name)):
        old = np.loadtxt(name, delimiter =',')
        if(len(old.shape)==1):
            old = old.reshape(1,len(array))
        try: # replace the data point with condition if it already exists
            index = np.where(old[:,0]==condition)[0][0]
            print(index)
            old = np.delete(old,index,0)
        except:
            None
        new = np.append(old, [array],axis=0)
        new = new[new[:,0].argsort()]
    else:
        print('write')
        new = [array]
    #replace the old file with the new, resulting in only 1 line change
    np.savetxt(name, new, delimiter = ',')
    time.sleep(random.random()*3)
    np.savetxt(name, new, delimiter = ',')

#load observables and original/decimated spin configurations + data preperation
######
#original spin configurations
mcName = 'z'+str(0)+'Mc'+str(it)+'L'+str(L)+'.csv'
mcS = np.array(np.loadtxt(folder_mc + mcName, delimiter =','))
size = len(mcS)
mcS = mcS.reshape(size, L, L)

#decimated spin configurations
rgName = 'z'+str(0)+'Rg'+str(it)+'L'+str(L)+'.csv'
rgS = np.array(np.loadtxt(folder_mc + rgName, delimiter =','))
rgS = rgS.reshape(size,L,L)

#reconstructed/super-resolved spin configurations
nnS = [None for i in range(steps+1)]

#preparing arrays to save the observables for every super resolution
nnM = np.zeros(steps+1) #absolute magnetization array
nnE = np.zeros(steps+1) #energy array
nnG = np.zeros(steps+1) #two point spin correlation function array
histM = [[] for i in range(steps+1)] #magnetization histograms
histE = [[] for i in range(steps+1)] #energy histograms

#preparing arrays for the CNN super resolution procedure
best_choices = np.zeros(steps+1) #best choice of weights and biases
forward_model = [None for i in range(steps+1)] #loading the CNN models
#the following output array is created in this way for visualization reasons
output = [0 for i in range(steps+1)] #array of outputs 

#further step dependent initializations
for step in range(0,steps+1):
    newL = L * (2 ** step) #system size after super-resolutions
    nnS[step] = np.zeros([1,newL,newL]) #store a empty spin configuration
    name_bc = folder_cp+'z'+str(step)+'best_choice.csv' #name of the best choice file
    best_choice_load = np.loadtxt(name_bc, delimiter =',') #loading the best choice
    best_choices[step] = int(best_choice_load[np.where(best_choice_load[:,0]==int(it))[0][0]][1])
    forward_model[step] = create_model(newL) #create a CNN model
    load_path = checkpoint_path[step][it][int(best_choices[step])] #path to weights and biases
    forward_model[step].load_weights(load_path).expect_partial() #load the weights and biases

    #initialize the possible observable values for the histograms
    for a in range(int((newL*newL)/2)+1):
        histM[step].append([a*2,0])
    for a in range(int(newL*newL)+1):
        histE[step].append([(a*4)-(newL*newL*2),0])

#start the reconstruction/super-resolution procedure
start = time.time() #start a timer

#iteration over every step, including the reconstruction to the original configuration
for dp_ in range(repeat_output):
    dp = dp_%size #repeat data point for all original spin configurations
    for step in range(0,steps+1):
        newL = L * (2 ** step) #system size after super-resolutions

        #load the input spin configuration for the reconstruction/super resolution
        if(step==0): #for the reconstruction
            inputS = np.array([rgS[dp]])
        elif(step==1): #for the first super-resolution step
            inputS = np.array([mcS[dp]])
            inputS = inputS.repeat(2,axis=1).repeat(2,axis=2)
        else: #for any super-resolution steps > 1
            inputS = np.array([nnS[step-1][0]]) #nnS is always size 1
            inputS = inputS.repeat(2,axis=1).repeat(2,axis=2)

        #applying the pbc padding
        inputS = pbc_padding(inputS,padding_sum)

        #diversify the spin configurations
        if(random.random()>0.5):
            inputS = -(inputS-1)

        inputSize = 1 #every spin configuration is super resolved 1 by 1
        output[step] = forward_model[step](inputS.reshape(inputSize,newL+padding_sum,newL+padding_sum,1)).numpy().reshape(inputSize,newL,newL) #output of the CNN

        #setting the output probability condition to a binary value 
        for a, spin in enumerate(output[step]):
            for b in range(newL):
                for c in range(newL):
                    r = random.uniform(0,1) #random number
                    condition = spin[b][c] #condition for spin to point in either direction
                    if(r>condition):
                        nnS[step][a][b][c] = 0
                    else:
                        nnS[step][a][b][c] = 1
        
        #calculate observables
        for a in nnS[step]:
            E = getE(a)
            M = getM(a)
            G = getG(a,dist)
            nnE[step] += E
            nnM[step] += M
            nnG[step] += G
            #calculate histograms
            intM = M/2 #index of magnetization
            intE = int(E+(newL*newL*2))/4 #index of energy
            histM[step][int(intM)][1] += 1 #set counter +1 for magetization
            histE[step][int(intE)][1] += 1 #set counter +1 for energy

#expectation of the observables
nnE = nnE / repeat_output
nnM = nnM / repeat_output
nnG = nnG / repeat_output

#storing observables in .csv files
######
for step in range(steps+1):
    #saving histograms
    np.savetxt(folder_hi+"z"+str(a)+"nnHistE"+str(it)+"L"+str(L)+".csv",histE[a],fmt='%g',delimiter=',')
    np.savetxt(folder_hi+"z"+str(a)+"nnHistM"+str(it)+"L"+str(L)+".csv",histM[a],fmt='%g',delimiter=',')
    #saving observables
    nn_sol = [it,nnM[a],nnE[a],nnG[a]] #constructing the line of the histogram
    savingData(nn_sol,int(it),folder_nn+"z"+str(a)+"NnL"+str(L)+".csv")

    #printing the previously saved data
    print("M:", nnM[a])
    print("E:", nnE[a])
    print("G:", nnG[a])

#printing the total run time
end = time.time() #end the timer
print(end - start)
