#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 19:31:55 2022

@author: Swe-Rana
"""


from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from sklearn.metrics import accuracy_score

(train_x_orig,train_y),(test_x_orig,test_y)= fashion_mnist.load_data()
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
train_y = train_y.reshape(1, len(train_y))
test_y = test_y.reshape(1, len(test_y))
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
no_of_class=10

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
layers_dims = [len(train_x),256,128,no_of_class]
onehot_encoded = list()

for i in range(train_y.shape[1]):
    c=train_y[:,i][0]
    letter = [0 for _ in range(no_of_class)]
    letter[c] = 1
    onehot_encoded.append(letter)

N=np.array(onehot_encoded)
Y=N.reshape(10,60000)
for i in range(0,60000):
      Y[:,i] = N[i]
layers_dims = [len(train_x),256,128,no_of_class]


def initialize_parameters(layers_dims):    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/(layers_dims[l]+layers_dims[l-1]))            
        parameters['b' + str(l)] =  np.zeros((layers_dims[l], 1))
    return parameters

def prev_updates(layers_dims):
        previous_updates = {}
        L = len(layers_dims)            # number of layers in the network
        for l in range(1, L):
            previous_updates["W"+str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
            previous_updates["b"+str(l)] = np.zeros((layers_dims[l], 1))
                    
        return previous_updates
previous_updates = prev_updates(layers_dims)


def feed_forward(A, W, b):

    Z =np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache 





def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def Relu_derivative(Z):
    return 1*(Z>0) 

def tanh(Z):
    return np.tanh(Z)

def tanh_backward(Z):
    t = np.tanh(Z)
    dt = 1 - (t**2)
    return dt

def sigmoid_backward(dA, cache):
   
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def relu_backward(dA, cache):

    
    Z = cache
    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
    

 
def softmax(n):
 	e = exp(n)
 	return e / e.sum()
 	
 	
 	
def activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache  = feed_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    if activation == "tanh":
        Z, linear_cache  = feed_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    
    if activation == "relu":
        
        Z, linear_cache = feed_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="sigmoid")
        caches.append(cache)
        
    AL, cache = activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
            
    return AL, caches
    
#backpropagtion    
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1./m*np.dot(dZ, A_prev.T)
    db = 1./m*np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

    
    
def activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    if activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

        
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    
    return dA_prev, dW, db    

def L_model_backward(Y,AL, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, activation="sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 2)],  current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
def update_parameters(parameters, grads, learning_rate,lamda):
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)] 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters
    
#stochastic gradient    
batch_size = 100
iterations_bat = int(train_x.shape[1]/batch_size)   
def stochastic_gradient(X, Y, layers_dims, learning_rate,num_epochs,lamda):
          parameters = initialize_parameters(layers_dims)
          for j in range(0,num_epochs):
            for i in range(0,iterations_bat):
                start = i*batch_size
                end = start+batch_size
                AL, caches = L_model_forward(X[:,start:end], parameters)
                grads = L_model_backward(Y[:,start:end],AL, caches)
                params = update_parameters(parameters, grads, learning_rate,lamda)
            z_pred_1, caches = L_model_forward(train_x, parameters)
            z_pred = np.argmax(z_pred_1,axis = 0)
            zyy = train_y.flatten()
            z_acc = accuracy_score(zyy,z_pred)
            print("accuracy",z_acc)             
            #print("iteration" + str(i) + "done")
            #print(start,end)
          return params
#momentum gradient descent optimizer
def momentum(X,Y,layers_dims,learning_rate,beta,num_epochs):
    parameters = initialize_parameters(layers_dims)
    previous_updates =prev_updates(layers_dims)
    L = len(parameters) // 2 # number of layers in the neural network
    for j in range(0,num_epochs):
        for i in range(0,iterations_bat):
            start = i*batch_size
            end = start+batch_size
            AL, caches = L_model_forward(X[:,start:end], parameters)
            grads = L_model_backward(Y[:,start:end],AL, caches)
                                   
            for l in range(1, L + 1):
                previous_updates["W"+str(l)] = (beta*previous_updates["W"+str(l)]) + ((1-beta)*grads["dW" + str(l)])
                parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate*previous_updates["W"+str(l)])
                
                previous_updates["b"+str(l)] = (beta*previous_updates["b"+str(l)]) + ((1-beta)*grads["db" + str(l)])
                parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate*previous_updates["b"+str(l)])
            
            
        z_pred_1, caches = L_model_forward(train_x, parameters)
        z_pred = np.argmax(z_pred_1,axis = 0)
        zyy = train_y.flatten()
        z_acc = accuracy_score(zyy,z_pred)
        print("accuracy",z_acc)             
#        print("iteration" + str(j) + "done")  
    return parameters, previous_updates
# rmsprop optimizer
def rmsprop(X,Y,layers_dims,learning_rate,beta,num_epochs):
    parameters = initialize_parameters(layers_dims)
    previous_updates =prev_updates(layers_dims)
    for j in range(0,num_epochs):
        for i in range(0,iterations_bat):
           
           start = i*batch_size
           end = start+batch_size
           AL, caches = L_model_forward(X[:,start:end], parameters)

           grads = L_model_backward(AL, Y[:,start:end], caches)

           delta = 1e-6 
            
           L = len(parameters) // 2 
        
           for l in range(1, L + 1):
                vdw = beta*previous_updates["W" + str(l)] + (1-beta)*np.multiply(grads["dW" + str(l)],grads["dW" + str(l)])
                vdb = beta*previous_updates["b" + str(l)] + (1-beta)*np.multiply(grads["db" + str(l)],grads["db" + str(l)])
        
                parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)] / (np.sqrt(vdw)+delta)
                parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)] / (np.sqrt(vdb)+delta)
        
                previous_updates["W" + str(l)] = vdw
                previous_updates["b" + str(l)] = vdb
             
        z_pred_1, caches = L_model_forward(train_x, parameters)
        z_pred = np.argmax(z_pred_1,axis = 0)
        zyy = train_y.flatten()
        z_acc = accuracy_score(zyy,z_pred)
        print("accuracy",z_acc)              
    return parameters, previous_updates

def adam(X,Y,layers_dims,v,m,t,learning_rate,beta,num_epochs):
    parameters = initialize_parameters(layers_dims)
    for j in range(0,num_epochs):
            for i in range(0,iterations_bat):
                start = i*batch_size
                end = start+batch_size
                AL, caches = L_model_forward(X[:,start:end], parameters)
                grads = L_model_backward(Y[:,start:end],AL, caches)
                
                L = len(parameters) // 2 # number of layers in the neural network
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8
            
                for l in range(1, L+1):
                    mdw = beta1*m["W"+str(l)] + (1-beta1)*grads["dW"+str(l)]
                    vdw = beta2*v["W"+str(l)] + (1-beta2)*np.square(grads["dW"+str(l)])
                    mw_hat = mdw/(1.0 - beta1**t)
                    vw_hat = vdw/(1.0 - beta2**t)
            
                    parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate * mw_hat)/np.sqrt(vw_hat + epsilon)
            
                    mdb = beta1*m["b"+str(l)] + (1-beta1)*grads["db"+str(l)]
                    vdb = beta2*v["b"+str(l)] + (1-beta2)*np.square(grads["db"+str(l)])
                    mb_hat = mdb/(1.0 - beta1**t)
                    vb_hat = vdb/(1.0 - beta2**t)
            
                    parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate * mb_hat)/np.sqrt(vb_hat + epsilon)
            
                    v["dW"+str(l)] = vdw
                    m["dW"+str(l)] = mdw
                    v["db"+str(l)] = vdb
                    m["db"+str(l)] = mdb
            
                t = t + 1 # timestep                
            z_pred_1, caches = L_model_forward(train_x, parameters)
            z_pred = np.argmax(z_pred_1,axis = 0)
            zyy = train_y.flatten()
            z_acc = accuracy_score(zyy,z_pred)
            print("accuracy",z_acc) 
                #print("iteration" + str(j) + "done")
    return parameters,v,m,t

def Nadam(X,Y,layers_dims,m,v,t,learning_rate,beta,num_epochs):
    parameters = initialize_parameters(layers_dims)
    L = len(parameters )//2
    for j in range(0,num_epochs):
        for l in range(1, L+1):
            parameters ["W"+str(l)] = parameters ["W"+str(l)] - beta*previous_updates["W"+str(l)]
            parameters ["b"+str(l)] = parameters ["b"+str(l)] - beta*previous_updates["b"+str(l)]
        for i in range(0,iterations_bat):
            start = i*batch_size
            end = start+batch_size
            AL, caches = L_model_forward(X[:,start:end], parameters)
            grads = L_model_backward( Y[:,start:end],AL,caches)
            
            L = len(parameters) // 2 # number of layers in the neural network
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
        
            for l in range(1, L+1):
                mdw = beta1*m["W"+str(l)] + (1-beta1)*grads["dW"+str(l)]
                vdw = beta2*v["W"+str(l)] + (1-beta2)*np.square(grads["dW"+str(l)])
                mw_hat = mdw/(1.0 - beta1**t)
                vw_hat = vdw/(1.0 - beta2**t)
        
                parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate * mw_hat)/np.sqrt(vw_hat + epsilon)
        
                mdb = beta1*m["b"+str(l)] + (1-beta1)*grads["db"+str(l)]
                vdb = beta2*v["b"+str(l)] + (1-beta2)*np.square(grads["db"+str(l)])
                mb_hat = mdb/(1.0 - beta1**t)
                vb_hat = vdb/(1.0 - beta2**t)
        
                parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate * mb_hat)/np.sqrt(vb_hat + epsilon)
        
                v["dW"+str(l)] = vdw
                m["dW"+str(l)] = mdw
                v["db"+str(l)] = vdb
                m["db"+str(l)] = mdb
        
            t = t + 1 # timestep            

        z_pred_1, caches = L_model_forward(train_x, parameters)
        z_pred = np.argmax(z_pred_1,axis = 0)
        zyy = train_y.flatten()
        z_acc = accuracy_score(zyy,z_pred)
        print("accuracy",z_acc)  

    return parameters

def nesterov(X,Y,learning_rate,beta,previous_updates,num_epochs):
        
    parameters=initialize_parameters(layers_dims)
    L = len(parameters)//2
    for j in range(0,num_epochs):
        for l in range(1, L+1):
            parameters["W"+str(l)] = parameters["W"+str(l)] - beta*previous_updates["W"+str(l)]
            parameters["b"+str(l)] = parameters["b"+str(l)] - beta*previous_updates["b"+str(l)]
        for i in range(0,iterations_bat):
            start = i*batch_size
            end = start+batch_size    
            AL, caches = L_model_forward(X[:,start:end], parameters)
            grads = L_model_backward( Y[:,start:end],AL,caches)
            
            L = len(parameters) // 2 # number of layers in the neural network
           
            for l in range(1, L + 1):
                previous_updates["W"+str(l)] = beta*previous_updates["W"+str(l)] + (1-beta)*grads["dW" + str(l)]
                parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*previous_updates["W"+str(l)]
                
                previous_updates["b"+str(l)] = beta*previous_updates["b"+str(l)] + (1-beta)*grads["db" + str(l)]
                parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*previous_updates["b"+str(l)]
             
        z_pred_1, caches = L_model_forward(train_x, parameters)
        z_pred = np.argmax(z_pred_1,axis = 0)
        zyy = train_y.flatten()
        z_acc = accuracy_score(zyy,z_pred)
        print("accuracy_nestrov",z_acc) 
    return params            

t = 1
learning_rate = 0.01
beta = 0.9
num_epochs=10
lamda = 0.0005

gd_optimizer = "stochastic_gradient"
if(gd_optimizer == "stochastic_gradient"):
    parameters =stochastic_gradient(train_x, Y, layers_dims, learning_rate,num_epochs,lamda)
if(gd_optimizer == "momentum"):
    parameters, previous_updates=momentum(train_x,Y,layers_dims,learning_rate,beta,num_epochs)
if(gd_optimizer == "rmsprop"):
    parameters, previous_updates=rmsprop(train_x,Y,layers_dims,learning_rate,beta,num_epochs)
if(gd_optimizer == "Adam"):
    t=1
    parameters,v,m,t=adam(train_x,Y,layers_dims,previous_updates,previous_updates,t,learning_rate,beta,num_epochs)
if(gd_optimizer == "Nadam"): 
    parameters=Nadam(train_x,Y,layers_dims,previous_updates,previous_updates,t,learning_rate,beta,num_epochs)
if(gd_optimizer == "nesterov"):
    parameters=nesterov(train_x,Y,learning_rate,beta,previous_updates,num_epochs)
