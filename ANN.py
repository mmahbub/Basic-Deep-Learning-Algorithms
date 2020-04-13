import numpy as np
import random
import matplotlib.pyplot as plt 
import time
import sys

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
        
        return np.dot(self.w, a.T)+ self.b.T
    
    def update_weights(self, delta_nabla_w, eta):
        # update weights given backpropagated error
        self.w = self.w + eta*delta_nabla_w
            
    def update_biases(self, delta_nabla_b, eta):
        # update biases given backpropagated error
        self.b = self.b + eta*delta_nabla_b
        
    
class FullyConnectedLayer:
    
    def __init__(self, n_neurons, activation_function, n_inputs, eta, w, b):
        #self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation_function = activation_function
        self.n_inputs = n_inputs
        self.eta = eta
        self.b = b
        index = [i for i in range(0, n_inputs)]
        # initialize neurons of layer
        self.neurons = [Neuron(activation_function, n_inputs, eta, w[n], b[n][0]) for n in range(0, n_neurons)]
        
        
    def calculate(self, a):
        # returns activations of each neuron (1d vector size n_neurons)
        outputs = np.zeros((self.n_neurons, 1))
        for i, neuron in enumerate(self.neurons):
            outputs[i] = (neuron.calculate(a))
            
        return outputs.flatten()
    
    def backprop_error(self, delta, sp):
        #compute error of previous layer 
        deltas = np.zeros((len(self.neurons), 1)) 
        for n_i in range(0, len(self.neurons)):
            weights = self.neurons[n_i].w
            deltas = np.add(deltas, np.array([delta.flatten()[n_i]*weights[w] for w in range(0, len(weights))]).flatten())
        deltas = deltas[0].reshape(1,len(deltas[0]))
        
        # return 1d vector size n_neurons current layer
        return np.array(deltas) * sp
    
    def update_weights(self, delta_nabla_w, eta):
        # update weights of layer following backpropagation
        for n_i in range(0, len(self.neurons)): 
            self.neurons[n_i].update_weights(delta_nabla_w[n_i], eta)
            
    def update_biases(self, delta_nabla_b, eta):
        # update biases of layer following backpropagation
        for n_i in range(0, len(self.neurons)): 
            self.neurons[n_i].update_biases(delta_nabla_b[n_i], eta)

    def z(self, a):
        # return z outputs of layer
        zs = np.zeros((self.n_neurons, 1))
        for n_i in range(0, len(self.neurons)): 
            zs[n_i] = self.neurons[n_i].z(a)
        
        return zs.flatten()
            
class NeuralNetwork:
    def __init__(self, n_layers, n_neurons, activation_functions, n_inputs, loss_func, eta, w, b):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation_functions = activation_functions
        self.n_inputs = n_inputs
        self.loss_func = loss_func
        self.eta = eta
        self.w_shape = np.array([np.zeros(l.shape) for l in w])
        self.b_shape = np.array([np.zeros(l.shape) for l in b])
        # initialize layers of network
        self.layers = [FullyConnectedLayer(self.n_neurons[l], self.activation_functions[l], self.n_neurons[l-1], self.eta, w[l-1], b[l-1]) for l in range(1, len(self.n_neurons))]
        
        
    def calculate(self, input_x):
        # feed forward; returns final output of network
        a = input_x
        for k, layer in enumerate(self.layers):
            a = layer.calculate(a)
            #print('layer output', a)
        return a
    
        
    def calculate_loss(self, input_x, y_true):
        # calculate loss given input and true value
        y_pred = self.calculate(input_x)
        
        # squared error loss
        if self.loss_func == "squared_error":
            return np.square(np.subtract(y_true, y_pred))
        # binary cross entropy loss
        else:
            return -sum(y_true[i]*log(y_pred[i]) for i in range(len(y_true)))
        
    def train(self, inputs_x, ys_true, epochs = 100):
        # train via gradient descent learning
        loss = []
        for k in range(0, epochs):
            
            # select input sequentially
            r = k%len(inputs_x)
                   
            # perform backprogation to compute gradient of cost function
            delta_nabla_w, delta_nabla_b = self.backprop(inputs_x[r], ys_true[r])
            
            
            # update weights with gradient descent layer by layer
            for l in range(len(self.layers)-1, -1, -1):
                self.layers[l].update_weights(delta_nabla_w[l], self.eta)
                self.layers[l].update_biases(delta_nabla_b[l], self.eta)
                
            print('Epoch', k, 'Loss', self.calculate_loss(inputs_x[r], ys_true[r])[0])
            
            # calculate loss
            loss.append(self.calculate_loss(inputs_x[r], ys_true[r])[0])
            
	# return array of losses
        return loss
        
        
        #return input_x
    
    def backprop(self, input_x, y_true):
        
        #print('begining backprop')
        nabla_b = np.copy(self.b_shape)
        nabla_w = np.copy(self.w_shape)
        
        # store input as activation for input layer
        a = input_x
        activations = [input_x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
        for k, layer in enumerate(self.layers):
            z = layer.z(a)
            zs.append(z)
            a = layer.calculate(a)
            activations.append(a)   
            
            
        # delta layer l
        delta = - self.cost_derivative(activations[-1], y_true) * self.sigmoid_prime(zs[-1], len(self.layers)-1)
        
        # gradients of cost function for last layer
        nabla_b[-1] = np.array([delta]).T
        nabla_w[-1] = np.dot(activations[-2].reshape(len(activations[-2]), 1), delta.reshape(1, len(delta))).T
        
        # gradients of cost function for layers L-1 ... 2 (hidden layer at index 0)
        for k in range(len(self.layers)-2, -1, -1):
            # calculate sigmoid prime given outputs of layer (1d vector size n_neurons in layer)
            sp = self.sigmoid_prime(zs[k], k)
            # calculate delta of layer using previous layer's weights, current layer's sigmoid prime
            # returns 1d vector size n_neurons in layer
            delta = self.layers[k+1].backprop_error(delta, sp)
            # store gradients 
            nabla_b[k] = delta.T
            # returns matrix size n_activations (n_neurons in next layer) by n_neurons
            nabla_w[k] = np.dot(activations[k].reshape(len(activations[k]), 1), delta).T

        return (nabla_w, nabla_b)
    
    def cost_derivative(self, output_activations, y):
        # squared error derivative
        if self.loss_func == "squared_error":
            return (output_activations-y)
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
	if command == 'example':
		print("Training Example Network ")
		# example network
		input_layer_size = 2
		hidden_layer_sizes = [2]
		output_layer_size = 2
		layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
		weights = np.array([np.array([[0.15, 0.2],[0.25, 0.3]]), np.array([[0.4, 0.45], [0.5, 0.55]])])
		biases=np.array([np.array([[0.35], [0.35]]), np.array([[0.6], [0.6]])])#np.array([np.ones((3,1)), np.ones((2,1))])

		exampleNetwork = NeuralNetwork(3, layer_sizes, [None, 'logistic', 'logistic'], 2, 'squared_error', 0.5, weights, biases)
		exampleNetwork.train(np.array([[0.05,0.1]]), np.array([[0.01, 0.99]]), epochs=1)
		print('Output After One Epoch', exampleNetwork.calculate(np.array([0.05,0.1])))

	elif command == 'and':
		# generate AND dataset
		x = np.array([np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])])#np.array([np.array([random.randint(0, 1), random.randint(0, 1) ]) for i in range(max_samples)])
		y = [x[i,0]*x[i,1] for i in range(0, len(x))]

		input_layer_size = 2
		hidden_layer_sizes = []
		output_layer_size = 1
		layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
		weights = np.array([np.array([[random.random(), random.random()],[random.random(), random.random()]])])
		biases=np.array([np.array([[random.random()], [random.random()]])])

		andNetwork = NeuralNetwork(2, layer_sizes, [None, 'logistic'], 2, 'squared_error', 0.99, weights, biases)
		loss = andNetwork.train(x, y, epochs=5000)

		for x_ in x:
			print("Input", x_)
			print('Output After 5000 Epochs', andNetwork.calculate(x_)[0], '\n')

	elif command == 'xor':
		# xor
		x = np.array([np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])])
		y = [x[i,0]^x[i,1] for i in range(0, len(x))]

		# XOR operator perceptron

		input_layer_size = 2
		hidden_layer_sizes = []
		output_layer_size = 1
		layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
		weights = np.array([np.array([[random.random(), random.random()],[random.random(), random.random()]])])
		biases=np.array([np.array([[random.random()], [random.random()]])])

		xorNetwork = NeuralNetwork(2, layer_sizes, [None, 'logistic'], 2, 'squared_error', 0.99, weights, biases)
		loss = xorNetwork.train(x, y, epochs=5000)

		# XOR operator network with hidden layer
		input_layer_size = 2
		hidden_layer_sizes = [2]
		output_layer_size = 1
		layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
		weights = np.array([np.array([[random.random(), random.random()],[random.random(), random.random()]]), np.array([[random.random(), random.random()]])])
		biases=np.array([np.array([[random.random()], [random.random()]]), np.array([[random.random()]])])
		
		xorHiddenNetwork = NeuralNetwork(3, layer_sizes, [None, 'logistic', 'logistic'], 2, 'squared_error', 0.99, weights, biases)
		loss = xorHiddenNetwork.train(x, y, epochs=5000)

		for x_ in x:
			print("Input", x_)
			print('XOR Perceptron Output After 5000 Epochs', xorNetwork.calculate(x_)[0])
			print('XOR Network with Hidden Layer Output After 5000 Epochs', xorHiddenNetwork.calculate(x_)[0], '\n')

	else:
		print('Invalid Command')

main()
