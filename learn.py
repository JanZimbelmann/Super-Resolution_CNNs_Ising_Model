#code written by Jan Zimbelmann
#it is required to have the following files:
#1. a file for the numeric results of the temperature transformation
#with the name './transformT.csv'
#2. a set of original and decimated spin configurations
#   and the observables of the original spin configurations
#original name: './configurations/z<step>Mc<it>L<L>.csv'
#decimated name: './configurations/z<step>Rg<it>L<L>.csv'
#observables name: './configurations/z<step>McL<L>.csv'
#the terms in the angle brackets refere to numeric variables
#'step' referes to the super resolution step
#'it' referes to the iteration index pointing to the transformT.csv temperature
#'L' is the system length, here 16
#spin configurations are stored as 0 and 1, not -1 and 1
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
verbose = 0 #chosing a verbose or silent neural network output

#parsing the super resultion step and the iteration index pointing in the transformT.csv 
parser = argparse.ArgumentParser(description='Specifying on which datapoints to learn.')
parser.add_argument('step', type=int, nargs='?', default = 0)
parser.add_argument('iteration', type=int, nargs='?', default = 0)
step = (parser.parse_args()).step
it = (parser.parse_args()).iteration

L=16 #system length
N=L*L #number of spins
dist = 3 #distance for the two point spin correlation function
T = np.array(np.loadtxt("transformT.csv", delimiter =',')) #temperature points
data_points = len(T[0]) #amount of all temperature points

#CNN setup conditions
EPOCHS = 3000 #maximum epoch size, will later be stopped with early stopping
BATCH_SIZE = 1000 #mini batch size
best_of = 9 #how often the training is to be repeated
lr = 1e-3 #learning rate
p_f = 2e-8 #proportionality factor

padding_sizes = [5,5,5] #initialize kernel amount (array size) and kernel length (array entries)
padding_size = len(padding_sizes) #storing kernel amount as a variable
padding_sum = sum(padding_sizes)-padding_size #calculating the padding length

es_patience = 15 #early stopping patience
es_thresh = 0 #early stopping threshold

#folders
folder_cp = 'checkpoints/' #weights and biases saving folder
folder_mc = 'configurations/' #configurations folder
folder_nn = 'reconstructions/' #folder for the CNN reconstruction data

#paths for the saved weights and biases
checkpoint_path = [[ None for y in range(best_of) ] for x in range(data_points)]

for i in range(data_points):
    for j in range(best_of):
        checkpoint_path[i][j] = folder_cp + "z" + str(step) + "T"+ str(i) +"cp" + str(j) + ".ckpt"

#printing some information on the variables
print("super-resolution counter.",step)
print("iteration counter:", it)
print("total amount of data points:", len(T[0]))
print("current T:", T[0][it])
print("padding length:",padding_sum)

#defining functions
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

#adding the pbc padding on a set of decimated spin configurations
#according to the CNN setup
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

#diversify data set by multiplying spin configurations randomly with -1
def variation(array, nums):
    new_array = []
    for i, s in enumerate(array):
        if(nums[i]==0):
            new_array.append(1-s)
        else:
            new_array.append(s)
    return np.array(new_array)

#load observables and original/decimated spin configurations + data preperation
######
#original spin configurations
mcS, train_mcS, test_mcS = [],[],[] #load monte carlo configurations
mcName = 'z'+str(step)+'Mc'+str(it)+'L'+str(L)+'.csv'
mcSpin = np.array(np.loadtxt(folder_mc + mcName, delimiter =','))

#store some data poin size variables
train_size = int(len(mcSpin)/2)
test_size = int(len(mcSpin)-train_size)
size = int(len(mcSpin))

#proper 2D reshaping
mcSpin = mcSpin.reshape(size,L,L)
mcS = mcSpin
#split into training and testing/validation dataset
train_mcS = mcS[:train_size]
test_mcS = mcS[train_size:]

#decimated spin configurations
rgS, train_rgS, test_rgS = [],[],[] #load decimated MC configurations
rgName = 'z'+str(step)+'Rg'+str(it)+'L'+str(L)+'.csv'
rgSpin = np.array(np.loadtxt(folder_mc + rgName, delimiter =','))
#propper 2D shaping
rgSpin = rgSpin.reshape(size,L,L)
rgS = rgSpin
#appling the pbc padding
rgS = pbc_padding(rgS,padding_sum)
#split into training and testing/validation dataset
train_rgS = rgS[:train_size]
test_rgS = rgS[test_size:]

#create array for configurations for testing the reconstruction
#is only to be validated on the test part of the data set
test_mdS = np.empty([best_of,test_size,L,L])
    
#loading calculated observables of the original spin configurations 
name_mc = 'z'+str(step)+'McL'+str(L)+'.csv'
MC = np.loadtxt(folder_mc + name_mc, delimiter =',')
mcK = MC[:,0]
mcM = MC[:,1]
mcE = MC[:,2]
mcG = MC[:,3]

#create and train the CNN model
######
#best_of amounts of models are to be trained
model_train = [ 0 for y in range(best_of) ]
mdE = np.empty([best_of])
mdM = np.empty([best_of])
#difference of the observables towards the original observables
model_dif = np.empty([best_of])

#training
tf.get_logger().setLevel('INFO')

#input data, the decimated spin configuration with padding
initial = np.array(train_rgS)
test = np.array(test_rgS)
#true data, the original spin configurations
target = np.array(train_mcS)

#diversifying the spin configurations
nums = np.random.choice([0, 1], size=size, p=[.5, .5])
initial = variation(initial,nums)
target = variation(target,nums)
nums = np.random.choice([0, 1], size=size, p=[.5, .5])
test = variation(test,nums)

#iterate the training procedure for a best_of amount
for j in range(best_of):
    now = time.time() #start timer
    model_train[j] = create_model(L)
    cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path[it][j],save_weights_only=True,verbose=0) #condition for saving the weights & biases
    es = EarlyStopping(monitor='loss', mode='min',verbose = 1, patience = es_patience, min_delta = es_thresh) #condition for early stopping
    #start training
    model_train[j].fit(initial.reshape(train_size,L+padding_sum,L+padding_sum,1),target.reshape(train_size,L,L,1),batch_size = BATCH_SIZE, epochs=EPOCHS, verbose=verbose, callbacks=[cb,es])
    #stop timer & print
    later = time.time()
    print(later - now)
    #testing the trained network on the validation data
    output = model_train[j](test.reshape(test_size,L+padding_sum,L+padding_sum,1)).numpy().reshape(test_size,L,L)
    #setting the output probability condition to a binary value
    for a, spin in enumerate(output):
        for b in range(L):
            for c in range(L):
                r = random.uniform(0,1) #random number
                condition = spin[b][c] #condition for spin to point in either direction
                if(r>condition):
                    test_mdS[j][a][b][c] = 0
                else:
                    test_mdS[j][a][b][c] = 1
    #calculate observable of the testing/validation data
    #later used to decide which model is the best according to best_of
    partM = 0
    partE = 0
    for a in test_mdS[j]:
        partM += getM(a)
        partE += getE(a)
    partM /= test_size
    partE /= test_size
    print(step, it, j, partM, partE)
    print(step, it, j, mcM[it], mcE[it])
    mdM[j]= partM
    mdE[j] = partE
    model_dif[j] = abs(mcE[it] - mdE[j]) + abs(mcM[it] - mdM[j])

#identifying the best choice of all the stored weights and biases
best_choice = np.argmin(model_dif)
print('best choice is %s \n'%(best_choice))
#creating or appending the best choice
name = folder_cp+'z'+str(step)+'best_choice.csv'
savingData([int(str(it)),best_choice], int(str(it)), name )
