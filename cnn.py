import numpy as np
import random
import matplotlib.pyplot as plt 
import time
import sys
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from numpy import unravel_index

class Neuron:
    def __init__(self, activation_func, n_inputs, eta, w, b):
        self.activation_func = activation_func
        self.n_inputs = n_inputs
        self.eta = eta
        self.w = w
        self.b = b
        
    def activate(self, z):
        #logistic
        if self.activation_func == 'logistic':
            return 1/(1 + np.exp(-z))
        #linear
        else:
            return z
        
    def calculate(self, a):
        # returns activation of neuron
        return self.activate(self.z(a))
    
    def z(self, a):
        return np.dot(self.w.flatten(), a.flatten().T)+ self.b.T
    
    def update_weights(self, delta_nabla_w, eta):
        # update weights given backpropagated error
        self.w = self.w + eta*delta_nabla_w
            
    def update_biases(self, delta_nabla_b, eta):
        # update biases given backpropagated error
        self.b = self.b + eta*delta_nabla_b

class FullyConnectedLayer:
    
    def __init__(self, n_neurons, activation_function, n_inputs, eta):
        
        self.n_neurons = n_neurons
        self.activation_function = activation_function
        self.n_inputs = n_inputs
        self.eta = eta
        index = [i for i in range(0, n_inputs)]
        #self.weights = np.array([[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0, 1.0]])
        self.weights = np.array([[(i%2) for i in range(0,n_inputs)]])
        self.bias = np.array([0.0])
        # initialize neurons of layer
        self.neurons = [Neuron(activation_function, n_inputs, eta, self.weights[n], self.bias) for n in range(0, n_neurons)]
        
        
    def calculate(self, a):
        # returns activations of each neuron (1d vector size n_neurons)
        outputs = np.zeros((self.n_neurons, 1))
        for i, neuron in enumerate(self.neurons):
            outputs[i] = (neuron.calculate(a))
            
        return outputs.flatten()
    
    def backprop_error(self, deltas, activations):
        #compute error of previous layer 
        return (np.array(deltas)).T, np.multiply(activations.reshape(len(activations.flatten()), 1).T, (np.array(deltas))).T
    
    def sum_weighted_deltas(self, delta, sp):
        #compute error of previous layer 
        deltas = np.zeros((len(self.neurons), 1)) 
        for n_i in range(0, len(self.neurons)):
            weights = self.neurons[n_i].w
            deltas = np.add(deltas, np.array([delta.flatten()[n_i]*weights[w] for w in range(0, len(weights))]).flatten())
        deltas = deltas[0].reshape(1,len(deltas[0]))
        # return 1d vector size n_neurons current layer
        return (np.array(deltas)).T #, np.multiply(activations.reshape(len(activations.flatten()), 1).T, (np.array(deltas) * sp)).T
        

    def update_weights(self, delta_nabla_w, eta):
        # update weights of layer following backpropagation
        for n_i in range(0, len(self.neurons)): 
            self.neurons[n_i].update_weights(delta_nabla_w[n_i].flatten(), eta)
            #print('NEW WEIGHT self.neurons[n_i].w', self.neurons[n_i].w)
            
    def update_biases(self, delta_nabla_b, eta):
        # update biases of layer following backpropagation
        for n_i in range(0, len(self.neurons)): 
            self.neurons[n_i].update_biases(delta_nabla_b[n_i], eta)
            #print('bias after', self.neurons[n_i].b)
            
    def print_weights_biases(self):
        for n_i in range(0, len(self.neurons)):
            print(self.neurons[n_i].w, self.neurons[n_i].b)

    def z(self, a):
        # return z outputs of layer
        zs = np.zeros((self.n_neurons, 1))
        for n_i in range(0, len(self.neurons)): 
            zs[n_i] = self.neurons[n_i].z(a)
        
        return zs.flatten()

class ConvolutionalLayer:
    # restrict to 2d convolutions
    def __init__(self, n_kernels, kernel_size, activation_func, input_dim, eta):
        
        # assume stride 1; padding valid
        
        if kernel_size == (3,3) and n_kernels == 1:
            self.weights = np.array([[[1.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,1.0]]])
        elif kernel_size == (3,3) and n_kernels == 2:
            self.weights = np.array([[[1.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,1.0]], 
                                     
                                     [[1.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,1.0]]])
        
        self.bias = np.array([0.0 for n in range(0,n_kernels)])
        
        # initialize neurons
        self.neurons = []
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.activation_function = activation_func
        self.input_dim = input_dim
        self.eta = eta
        
        kernel_neurons = []
        for k in range(0, n_kernels):
            neurons_mat = []
            for i in range(0, input_dim[0]-kernel_size[0] + 1):
                neurons_row = []
                for j in range(0, input_dim[1]-kernel_size[1] + 1):
                    neurons_row.append(Neuron(activation_func, kernel_size[0]*kernel_size[1], eta, np.array([self.weights[k]]), np.array(self.bias[k])))
                neurons_mat.append(neurons_row)
            kernel_neurons.append(neurons_mat)      
        self.neurons = kernel_neurons
        
    def calculate(self, a):
        # calculates output of all neurons in layer given input
        out_w = (a.shape[1]-self.kernel_size[0]) + 1
        out_h = (a.shape[2]-self.kernel_size[1]) + 1
        
        output = np.zeros((self.n_kernels, out_w, out_h))
        
        # convolve
        for k in range(0, self.n_kernels):
            for i in range(0, a.shape[1]-self.kernel_size[0]+1):
                for j in range(0, a.shape[2]-self.kernel_size[1]+1):
                    output[k, i, j] = self.neurons[k][i][j].calculate(a[0,i:i+self.kernel_size[0],j:j+self.kernel_size[1]])          
        return output    

    # additional backprop methods
    def backprop_error(self, delta,  activations):
        
        out_w = (self.input_dim[0]-self.kernel_size[0]) + 1
        out_h = (self.input_dim[1]-self.kernel_size[1]) + 1
        delta = delta.reshape((self.n_kernels, out_w, out_h))
        
        # iterate through weights
        delta_kernels = []
        for k in range(0, self.n_kernels):
            delta_weights = []
            for c in range(0, len(activations)):
                neurons_mat = []
                for a in range(0, self.kernel_size[0]):
                    neurons_row = []
                    for b in range(0, self.kernel_size[1]):
                        err_sum = 0.0
                        for i in range(0, self.input_dim[0]-self.kernel_size[0] + 1):
                            for j in range(0, self.input_dim[1]-self.kernel_size[1] + 1): 
                                err_sum += delta[k,i,j] *  activations[c][i+a][j+b]
                        neurons_row.append(err_sum)
                    neurons_mat.append(neurons_row)
                delta_weights.append(neurons_mat) 
            delta_kernels.append(delta_weights)
              
        # return 1d vector size n_neurons current layer
        return np.array([[np.sum(d)] for d in delta]), np.array(delta_kernels)
    
    def sum_weighted_deltas(self, delta, sp):
        
        delta_kernels_out = []
        for c in range(0, self.n_kernels):
            # input dim
            delta_out = []
            for k in range(0, delta.shape[0]):
                neurons_mat = []
                for i in range(0, self.input_dim[0]):
                    neurons_row = []
                    for j in range(0, self.input_dim[1]):
                        err_sum = 0.0
                        weights_flipped = self.neurons[k][0][0].w#np.flip(np.flip(self.neurons[k][0][0].w, 0), 1) #self.matrixflip(self.matrixflip(np.matrix(self.neurons[k][0][0].w),'h'),'v')
                        for a in range(0, self.kernel_size[0]):
                            for b in range(0, self.kernel_size[1]):
                                if i-a < 0 or j-b < 0 or i-a > delta.shape[1]-1 or j-b > delta.shape[2]-1:
                                    err_sum += 0.0*weights_flipped[c,a,b]                              
                                else:
                                    err_sum += delta[k,i-a,j-b]*weights_flipped[c,a,b]
                        neurons_row.append(err_sum)
                    neurons_mat.append(neurons_row)
                delta_out.append(neurons_mat)
            delta_kernels_out.append(delta_out)

        return np.array(delta_kernels_out)
    
    def z(self, a):
        # return z outputs of layer
        out_w = (a.shape[1]-self.kernel_size[0]) + 1
        out_h = (a.shape[2]-self.kernel_size[1]) + 1
        
        output = np.zeros((self.n_kernels, out_w, out_h))
        
        # convolve
        for k in range(0, self.n_kernels):
            for i in range(0, a.shape[1]-self.kernel_size[0]+1):
                for j in range(0, a.shape[2]-self.kernel_size[1]+1):
                    conv_sum = 0.0
                    output[k, i, j] = self.neurons[k][i][j].z(a[0,i:i+self.kernel_size[0],j:j+self.kernel_size[1]])         
        return output
    
    def update_weights(self, delta_nabla_w, eta):
        # update weights of layer following backpropagation
        for k in range(0, self.n_kernels):
            for i in range(0, self.input_dim[0]-self.kernel_size[0]+1):
                for j in range(0, self.input_dim[0]-self.kernel_size[1]+1):
                    self.neurons[k][i][j].update_weights(delta_nabla_w[k], eta)
                    
    def update_biases(self, delta_nabla_b, eta):
        # update weights of layer following backpropagation
        for k in range(0, self.n_kernels):
            for i in range(0, self.input_dim[0]-self.kernel_size[0]+1):
                for j in range(0, self.input_dim[0]-self.kernel_size[1]+1):
                    self.neurons[k][i][j].update_biases(delta_nabla_b[k], eta)
                    
    def print_weights_biases(self):
        for k in range(0, self.n_kernels):
            print(self.neurons[k][0][0].w, self.neurons[k][0][0].b)

class MaxPoolingLayer:
    def __init__(self, kernel_size, input_dim):
        # stride = filter size 
        # no padding
        self.kernel_size = kernel_size
        self.kernel = np.matrix([[1,0],[0,1]])
        self.input_dim = input_dim
        self.mask = np.zeros((input_dim[0], input_dim[1], input_dim[2]))
    
    def calculate(self, a):
        out_w = int((a.shape[1]-self.kernel_size)/self.kernel_size + 1)
        out_h = int((a.shape[2]-self.kernel_size)/self.kernel_size + 1)
        
        output = np.zeros((self.input_dim[0], out_w, out_h))
        
        mask_list = []
        for k in range(0, a.shape[0]):
            mask_k = []
            for i in range(0, a.shape[1]-self.kernel_size+1, self.kernel_size):
                for j in range(0, a.shape[2]-self.kernel_size+1, self.kernel_size):
                    sub_a = a[k,i:i+self.kernel_size,j:j+self.kernel_size]
                    mask_k.append(np.array(unravel_index(sub_a.argmax(), sub_a.shape))+np.array([i, j]))
                    output[k,int(i-(i)/self.kernel_size),int(j-j/self.kernel_size)] = np.max(a[k,i:i+self.kernel_size,j:j+self.kernel_size])
            mask_list.append(mask_k)
        
        for k in range(0, self.input_dim[0]):
            for i in range(0, len(mask_list[k])):
                    self.mask[k, mask_list[k][i][0], mask_list[k][i][1]] = 1
        
        return output
    
    def backprop_error(self, delta):
        
        delta = delta.reshape((self.input_dim[0], int((self.input_dim[1]-self.kernel_size)/self.kernel_size+1), int((self.input_dim[2]-self.kernel_size)/self.kernel_size+1)))
        output_d = np.zeros(self.mask.shape)
        for k in range(0, self.input_dim[0]):
            mask_k = []
            for i in range(0, self.input_dim[1]-self.kernel_size+1, self.kernel_size):
                for j in range(0, self.input_dim[2]-self.kernel_size+1, self.kernel_size):
                    output_d[k,i:i+self.kernel_size,j:j+self.kernel_size] = self.mask[k,i:i+self.kernel_size,j:j+self.kernel_size]*delta[k,int(i-(i)/self.kernel_size),int(j-j/self.kernel_size)]
                    
        
        return output_d 
    
class FlattenLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.activation_function = None
        
    def calculate(self, a):
        # given input calculates output of layer (no neurons)
        return a.flatten()
    
    def z(self, a):
        return self.calculate(a)

    def backprop_error(self, delta, activations):
        #compute error of previous layer 
        return (np.array(delta)), np.multiply(activations, (np.array(delta)))
    
    def update_weights(self, delta_nabla_w, eta):
        return
    
class NeuralNetwork:
    def __init__(self, n_inputs, loss_func, eta):
        self.n_inputs = n_inputs
        self.loss_func = loss_func
        self.eta = eta
        self.weights = [] 
        self.has_maxpool = False
        self.bias = [] 
        self.layers = []
        
    def addLayer(self, newLayer):
        # params: all details for initializing layer & weights
        # input size : current final layer
        self.layers.append(newLayer)
        
        if type(self.layers[-1]) != FlattenLayer and type(self.layers[-1]) != MaxPoolingLayer:
            self.weights.append(self.layers[-1].weights)
            self.bias.append(self.layers[-1].bias)
            
        if type(self.layers[-1]) == MaxPoolingLayer:
            self.has_maxpool = True
    
    def calculate(self, input_x):
        # feed forward; returns final output of network
        a = input_x
        for k, layer in enumerate(self.layers):
            a = layer.calculate(a)
            #print('layer', k, 'output', a)
        return a
    
        
    def calculate_loss(self, input_x, y_true):
        # calculate loss given input and true value
        y_pred = self.calculate(input_x)
        
        # squared error loss
        if self.loss_func == "squared_error":
            return np.square(np.subtract(y_true, y_pred))
        
        # binary cross entropy loss
        else:
            return -(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
        
    def train(self, inputs_x, ys_true, epochs = 100):
        self.w_shape = [np.zeros(l.shape) for l in self.weights]
        self.b_shape = [np.zeros(l.shape) for l in self.bias]
        
        # train via gradient descent learning
        loss = []
        for k in range(0, epochs):
            
            # select input sequentially
            r = k%len(inputs_x)
                   
            # perform backprogation to compute gradient of cost function
            delta_nabla_w, delta_nabla_b = self.backprop(inputs_x[r], ys_true[r])
            
            # update FC Layer weights
            self.layers[-1].update_weights(delta_nabla_w[-1], self.eta)
            self.layers[-1].update_biases(delta_nabla_b[-1], self.eta)
            
            # update weights with gradient descent layer by layer
            # start at last conv layer
            for k in range(len(self.layers)-3-self.has_maxpool, -1, -1):
            
                self.layers[k].update_weights(delta_nabla_w[k], self.eta)
                self.layers[k].update_biases(delta_nabla_b[k], self.eta)
                
            print('Epoch', k, 'Loss', self.calculate_loss(inputs_x[r], ys_true[r])[0])
            
            # print weights
            print('Updated weights and biases:')
            for k in range(0, len(self.layers)-3-self.has_maxpool+1):
                self.layers[k].print_weights_biases()
            self.layers[-1].print_weights_biases()
            
            # calculate loss
            loss.append(self.calculate_loss(inputs_x[r], ys_true[r])[0])
            
	# return array of losses
        return loss
        
        
        #return input_x
    
    def backprop(self, input_x, y_true):
        
        #print('begining backprop')
        nabla_b = self.b_shape#np.copy(self.b_shape)
        nabla_w = self.w_shape#np.copy(self.w_shape)
        
        # store input as activation for input layer
        a = input_x
        activations = [input_x] # list to store all the activations, layer by layer
        zs = [[0]] # list to store all the z vectors, layer by layer
        
        for k, layer in enumerate(self.layers):
            if type(self.layers[k]) == FlattenLayer:
                continue
                
            if type(self.layers[k]) == MaxPoolingLayer:
                z = np.array([0.0])
            else:
                z = layer.z(a)
            zs.append(z)
            a = layer.calculate(a)
            activations.append(a)   
            
        # delta layer l
        delta = - self.cost_derivative(activations[-1], y_true) * self.sigmoid_prime(zs[-1], len(self.layers)-1)
        
        # gradients of cost function for last layer
        nabla_b[-1] = np.array([delta]).T
        nabla_w[-1] = np.multiply((activations[-2]).flatten(), delta)#np.dot(activations[-2].reshape(len(activations[-2]), 1), delta.reshape(1, len(delta))).T
        
        # gradients of cost function for layers L-1 ... 2 (hidden layer at index 0)
        skip = 0
        
        if self.has_maxpool:
            sp = self.sigmoid_prime(zs[-2], len(self.layers)-2)
            # should be FC
            delta = self.layers[-1].sum_weighted_deltas(delta, sp)
            delta = self.layers[len(self.layers)-3].backprop_error(delta)
        
        # starts at first convolutional layer
        for k in range(len(self.layers)-3-self.has_maxpool, -1, -1):
            
            if k == len(self.layers)-3:
                next_layer_ind = len(self.layers)-1
            else: 
                next_layer_ind = k+1
                
            sp = self.sigmoid_prime(zs[k+1], k)
            
            # calculate delta of layer using previous layer's weights, current layer's sigmoid prime
            # returns 1d vector size n_neurons in layer
            
            if not self.has_maxpool:
                delta = self.layers[next_layer_ind].sum_weighted_deltas(delta, sp)
            
            delta = delta.reshape(sp.shape) * sp 
            
            nb, nw = self.layers[k].backprop_error(delta, activations[k])
            # store gradients 
            nabla_b[k] = nb
            # returns matrix size n_activations (n_neurons in next layer) by n_neurons (assumes fully connected)
            nabla_w[k] = nw

        return (nabla_w, nabla_b)
    
    def cost_derivative(self, output_activations, y):
        # squared error derivative
        if self.loss_func == "squared_error":
            return 2*(output_activations-y)
        else:
            # cost derivative for binary cross entropy
            return -y/output_activations + (1-y)/(1-output_activations)
            
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def sigmoid_prime(self, z, l):
        # derivative of logistic activation function
        if self.layers[l].activation_function == 'logistic':
            return self.sigmoid(z)*(1-self.sigmoid(z)) 
        # derivative of linear activation function
        else:
            return 1

def main():

    command = sys.argv[1]
    print(sys.argv)
    if command == 'example1':
        print("Network with 5x5 input, one 3x3 convolution layer w/ single kernel, flatten layer, single neuron for output.")
        # example network
        
        print('Keras Solution')
        model = models.Sequential()
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', input_shape=(5, 5, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, input_shape=(9,), activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.5)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        x_train = np.array([[[[0.0,1.0,2.0,3.0,4.0],[0.0,1.0,2.0,3.0,4.0],[0.0,1.0,2.0,3.0,4.0],
                              [0.0,1.0,2.0,3.0,4.0],[0.0,1.0,2.0,3.0,4.0]]]])
        y_train = np.array([[0.0]])
        model.layers[0].set_weights([np.array([[[[ 1.0]],[[0.0]],[[1.0]]],
                [[[0.0]], [[1.0]],[[0.0]]],
                [[[1.0]], [[ 0.0]],[[1.0]]]]), 
                np.array([0.0])])
        model.layers[2].set_weights([np.array([[0.0],[1.0], [0.0], [1.0],[ 0.0],[1.0],[0.0],[1.0],[0.0]]), 
                                       np.array([0.0])])
        
        # FIT MODEL
        model.fit(x_train.reshape(1,5,5,1), y_train, epochs=1, validation_data=(x_train.reshape(1,5,5,1), y_train))
        print('Updated weights and biases', model.get_weights())
        print('Predicted output', model.predict(x_train.reshape(1,5,5,1)), '\n')
        
        # our network
        print("Our Solution")
        exampleNetwork1 = NeuralNetwork((5,5), 'squared_error', 0.5)
        exampleNetwork1.addLayer(ConvolutionalLayer(1, (3,3), 'logistic', (5,5), 0.5))
        exampleNetwork1.addLayer(FlattenLayer((1,3,3)))
        exampleNetwork1.addLayer(FullyConnectedLayer(1, 'logistic', 9, 0.5))
        print('Loss after 1 epoch:', exampleNetwork1.train(x_train, y_train, epochs=1))
        print('Predicted output', exampleNetwork1.calculate(x_train[0]))
        
    elif command == 'example2':
        print("Network with 5x5 input, one 3x3 convolution layer w/ single kernel, another 3x3 convolution layer w/ single kernel, flatten layer, single neuron for output.")
        
        print('Keras Solution')
        model = models.Sequential()
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', input_shape=(5, 5, 1)))
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', input_shape=(3, 3, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, input_shape=(1,), activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.5)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        x_train = np.array([[[[0.0,1.0,2.0,3.0,4.0],[0.0,1.0,2.0,3.0,4.0],[0.0,1.0,2.0,3.0,4.0],
                              [0.0,1.0,2.0,3.0,4.0],[0.0,1.0,2.0,3.0,4.0]]]])
        y_train = np.array([[0.0]])
        
        model.layers[0].set_weights([np.array([[[[ 1.0]],[[0.0]],[[1.0]]],
                [[[0.0]], [[1.0]],[[0.0]]],
                [[[1.0]], [[ 0.0]],[[1.0]]]]), 
                np.array([0.0])])
        model.layers[1].set_weights([np.array([[[[ 1.0]],[[0.0]],[[1.0]]],
                [[[0.0]], [[1.0]],[[0.0]]],
                [[[1.0]], [[ 0.0]],[[1.0]]]]), 
                np.array([0.0])])
        model.layers[3].set_weights([np.array([[0.0]]), 
                                       np.array([0.0])])
        
        # FIT MODEL
        model.fit(x_train.reshape(1,5,5,1), y_train, epochs=1, validation_data=(x_train.reshape(1,5,5,1), y_train))
        print('Updated weights and biases', model.get_weights())
        print('Predicted output', model.predict(x_train.reshape(1,5,5,1)),'\n')
        
        
        print('Our Solution:')
        exampleNetwork1 = NeuralNetwork((5,5), 'squared_error', 0.5)
        exampleNetwork1.addLayer(ConvolutionalLayer(1, (3,3), 'logistic', (5,5), 0.5))
        exampleNetwork1.addLayer(ConvolutionalLayer(1, (3,3), 'logistic', (3,3), 0.5))
        exampleNetwork1.addLayer(FlattenLayer((1,1,1)))
        exampleNetwork1.addLayer(FullyConnectedLayer(1, 'logistic', 1, 0.5))
        
        print('Loss after 1 epoch:', exampleNetwork1.train(x_train, y_train, epochs=1))
        print('Predicted output', exampleNetwork1.calculate(x_train[0]))

    elif command == 'example3':
        print("Network with 6x6 input, one 3x3 convolution layer with two kernels, a 2x2 max pooling layer, a flatten layer, and a single layer for the output.")

        print('Keras Solution')
        model = models.Sequential()
        model.add(layers.Conv2D(2, (3, 3), activation='sigmoid', input_shape=(6, 6, 1)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, input_shape=(8,), activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.5)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        x_train = np.array([[[[0.0,1.0,2.0,1.0,2.0,0.0],[0.0,1.0,2.0,1.0,2.0,0.0],[0.0,1.0,2.0,1.0,2.0,0.0],
                              [0.0,1.0,2.0,1.0,2.0,0.0],[0.0,1.0,2.0,1.0,2.0,0.0],[0.0,1.0,2.0,1.0,2.0,0.0]]]])
        y_train = np.array([[1.0]])
        
        model.layers[0].set_weights([np.array([[[[1.0,  1.0]],

        [[1.0, 1.0]],

        [[ 1.0,  1.0]]],


       [[[1.0, 1.0]],

        [[2.0, 2.0]],

        [[1.0, 1.0]]],


       [[[1.0, 1.0]],

        [[1.0, 1.0]],

        [[1.0, 1.0]]]]), np.array([0.0, 0.0])])
        
        model.layers[3].set_weights([np.array([[0.0], [1.0], [0.0],[1.0],[0.0],[1.0],[0.0], [1.0]]), 
                                       np.array([0.0])])
        
        model.fit(x_train.reshape(1,6,6,1), y_train, epochs=1, validation_data=(x_train.reshape(1,6,6,1), y_train))
        print('Updated weights and biases', model.get_weights())
        print('Predicted output', model.predict(x_train.reshape(1,6,6,1)),'\n')      
        
        print('Our Solution:')
        
        
        exampleNetwork3 = NeuralNetwork((6,6), 'squared_error', 0.5)
        exampleNetwork3.addLayer(ConvolutionalLayer(2, (3,3), 'logistic', (6,6), 0.5))
        exampleNetwork3.addLayer(MaxPoolingLayer(2, (2,4,4)))
        exampleNetwork3.addLayer(FlattenLayer((2,2,2)))
        exampleNetwork3.addLayer(FullyConnectedLayer(1, 'logistic', 8, 0.5))
        print('Loss after 1 epoch:', exampleNetwork3.train(x_train, y_train, epochs=1))
        print('Predicted output', exampleNetwork3.calculate(x_train[0]))

    
    elif command == 'class_example1':

        x_train = np.array([[[                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]])
        y_train = np.array([[0.0]])
        model = models.Sequential()
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', input_shape=(7, 7, 1)))
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', input_shape=(5, 5, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, input_shape=(9,), activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.5)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        
        
        model.layers[0].set_weights([np.array([[[[ 1.0]],[[0.0]],[[1.0]]],
                [[[0.0]], [[1.0]],[[0.0]]],
                [[[1.0]], [[0.0]],[[1.0]]]]), 
                np.array([0.0])])
        model.layers[1].set_weights([np.array([[[[ 1.0]],[[0.0]],[[1.0]]],
                [[[0.0]], [[1.0]],[[0.0]]],
                [[[1.0]], [[0.0]],[[1.0]]]]), 
                np.array([0.0])])
        model.layers[3].set_weights([np.array([[0.0],[1.0], [0.0], [1.0],[ 0.0],[1.0],[0.0],[1.0],[0.0]]), 
                                       np.array([0.0])])
        #print('weights', model.get_weights())
        #print('model.predict(x_train.reshape(1,5,5,1))', model.predict(x_train.reshape(1,8,8,1)))
        
        model.fit(x_train.reshape(1,7,7,1), y_train, epochs=1, validation_data=(x_train.reshape(1,7,7,1), y_train))
        print('Updated weights and biases', model.get_weights())
        print('Predicted output', model.predict(x_train.reshape(1,7,7,1)),'\n')
        
        exampleNetwork3 = NeuralNetwork((8,8), 'squared_error', 0.5)
        exampleNetwork3.addLayer(ConvolutionalLayer(1, (3,3), 'logistic', (7,7), 0.5))
        exampleNetwork3.addLayer(ConvolutionalLayer(1, (3,3), 'logistic', (5,5), 0.5))
        exampleNetwork3.addLayer(FlattenLayer((1,3,3)))
        exampleNetwork3.addLayer(FullyConnectedLayer(1, 'logistic', 9, 0.5))
        print('Loss after 1 epoch:', exampleNetwork3.train(x_train, y_train, epochs=1))
        print('Predicted output', exampleNetwork3.calculate(x_train[0]))
        
    
    elif command == 'class_example2':

        x_train = np.array([[[                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0],
                                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]])
        y_train = np.array([[0.0]])
        model = models.Sequential()
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', input_shape=(8, 8, 1)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, input_shape=(8,), activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.5)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        
        
        model.layers[0].set_weights([np.array([[[[ 1.0]],[[0.0]],[[1.0]]],
                [[[0.0]], [[1.0]],[[0.0]]],
                [[[1.0]], [[0.0]],[[1.0]]]]), 
                np.array([0.0])])
        model.layers[3].set_weights([np.array([[0.0],[1.0], [0.0], [1.0],[ 0.0],[1.0],[0.0],[1.0],[0.0]]), 
                                       np.array([0.0])])
        #print('weights', model.get_weights())
        #print('model.predict(x_train.reshape(1,5,5,1))', model.predict(x_train.reshape(1,8,8,1)))
        
        model.fit(x_train.reshape(1,8,8,1), y_train, epochs=1, validation_data=(x_train.reshape(1,8,8,1), y_train))
        print('Updated weights and biases', model.get_weights())
        print('Predicted output', model.predict(x_train.reshape(1,8,8,1)),'\n')
        
        exampleNetwork3 = NeuralNetwork((8,8), 'squared_error', 0.5)
        exampleNetwork3.addLayer(ConvolutionalLayer(1, (3,3), 'logistic', (8,8), 0.5))
        exampleNetwork3.addLayer(MaxPoolingLayer(2, (1,6,6)))
        exampleNetwork3.addLayer(FlattenLayer((1,3,3)))
        exampleNetwork3.addLayer(FullyConnectedLayer(1, 'logistic', 9, 0.5))
        print('Loss after 1 epoch:', exampleNetwork3.train(x_train, y_train, epochs=1))
        print('Predicted output', exampleNetwork3.calculate(x_train[0]))
        
    else:
        print('Invalid Command')

main()
