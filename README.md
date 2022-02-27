# Assignment_1
CS6910
**Program Files description:**
question_1.py : The code downloads the fashion_mnist data sets and plots the one image from each class

question_2.py : This program file implements the feed forward propagation and outputs the probabilities of the images for each class.
The program is flexible to change the no.of hidden layers and size of each hiiden leyer. 
In order to change the hidden layers and size of hidden layer, set layers_dims variable.

question_3.py : Implement the backpropagation algorithm with support for the following optimisation functions 
                  - sgd
                  - momentum based gradient descent
                  - nesterov accelerated gradient descent
                  - rmsprop
                  - adam
                  - nadam
                  
By setting the batch_size variable we can change the batch size. 

Question4.ipynb program has algorithm to do wandb sweeps. 

**Usage and Test Instructions**
 **Question_1.py** :  Change the wandb credentials to test the code. By running the code you can generate a plots consists a image from each class
 
 **Question-2.py** This programs generates the  prediction probability of an image for all the 10 clasess, To change the hidden layer dimensions or number of hidden layers set the "Hidden_layer_dimensions" variable. for ex: Hidden_layer_dimensions = [256,128,64] (means 3 hidden layers with sizes 256,128,64 size respectivly)  
 
To perform feed forward propagation use function "L_layer_model(X, Y, layers_dims)" which takes X as input Y as labels and layer dimensions. And this function returns three variables. In which output_probability contains the probability of all the test images for each class.

ex:parameters,caches,AL,**output_probability** = L_layer_model(test_x, test_y, layers_dims) 

And the program prints the probability of the first image passed for all the classes (we are printing it just for the testing convinience)

 **Question-3.py** Implements theback propagation along with all the optimizers. by changing the "gd_optimizer" variable we can select which optimizer to run.
 To add the new optimizer you need to follow below steps. 
 
To perform feed forward propagation you need to call the function L_model_forward : L_model_forward(X, parameters) where X is input features and parameters is weights and biases
 
To perform back propagation you need to call the function  L_model_backward : L_model_backward(Y,AL, caches) where Y is labels of input data and AL is output from feed forward , caches is a variable which contains post activations, weights , biases and pre activations.
 
Then you have to call your optimizer function.

We have developed All the optimizers in similar manner and tested. This program outputs the accuracy for each epoch. And at the end outputs the accuracy of the model on test set.

 **Question4.ipynb** This program contains wandb sweeps and resposible for generating the sweeps.      

            
  




