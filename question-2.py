#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:06:21 2022

@author: tenet
"""


from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from numpy import exp

(train_x_orig,train_y),(test_x_orig,test_y)= fashion_mnist.load_data()
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
train_y = train_y.reshape(1, len(train_y))
test_y = test_y.reshape(1, len(test_y))
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
no_of_class=len(np.unique(train_y))

train_x = train_x_flatten/255
test_x = test_x_flatten/255
layers_dims = [len(train_x),256,128,no_of_class]
onehot_encoded = list()

for i in range(train_y.shape[1]):
    c=train_y[:,i][0]
    letter = [0 for _ in range(no_of_class)]
    letter[c] = 1
    onehot_encoded.append(letter)

Y=np.array(onehot_encoded)
num_features = train_x.shape[0]
layers_dims = [num_features,256,128,no_of_class]

# Function: initialize_parameters_ 
# used for initialising the parameters of feed forward network
# Returns parameters
def initialize_parameters_(layer_dims):
   
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] =  np.zeros((layer_dims[l], 1))
       
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

       
    return parameters

# Function: feed_forward
# performs the feed forward    
#returns post activation(Z), pre activation (cache) values 
def feed_forward(A, W, b):
   
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

#function: sigmoid
#performs the sigmoid activation function

def sigmoid(Z):
   
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

#function: sigmoid
#performs the ReLu activation function
def relu(Z):    
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache
def softmax(Z):
    Z = cache
 	A = exp(Z)
 	A= A / A.sum()
    return A,cache
 	

#function: activation_forward
#calls the feed forward and sigmoid or Relu activation functions
#called from 
def activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":

        Z, linear_cache  = feed_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
   
    elif activation == "relu":
        Z, linear_cache = feed_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

#Function : L_model_forward
# Will perform forward propagation 

def L_model_forward(X, parameters):
   
    caches = []
    A = X
    L = len(parameters) // 2 
   
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "sigmoid")
        caches.append(cache)
          
    AL, cache = activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    AP=[]
    for i in range(AL.shape[1]):
         AA,cache = softmax(AL[:,i])
         AP.append(AA) 

    return AL, caches,AA

#Function : L_layer_model
#Will intialize parameters by calling initialize_parameters_
#Calls L_model_forward to perform feed forward

def L_layer_model(X, Y, layers_dims):
   
    parameters = initialize_parameters_(layers_dims)
    AL, caches,AA = L_model_forward(X, parameters)
    return parameters,caches,AL,AA

parameters,caches,AL,output_probability = L_layer_model(train_x, Y, layers_dims)

