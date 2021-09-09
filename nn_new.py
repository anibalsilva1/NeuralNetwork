import numpy as np
from activations import *
from cost_functions import *
from metrics import *

class NeuralNetwork(object):
    def __init__(self):
        self.input_shape = None
        self.layers = []
        self.weights = None
        self.bias = None
        self.cost = None    
        

    def add_layer(self, layer):
        if layer.activation_fun == "input_layer":
            self.input_shape = layer.n_neurons
        self.layers.append(layer)

    def compile_model(self, cost, metric):

        neurons_per_layer = []

        for layer in self.layers:
            if layer == "input_layer": 
                neurons_per_layer.append(self.input_shape) 
            else: 
                neurons_per_layer.append(layer.n_neurons)

        self.weights = [np.random.randn(j,i)*np.sqrt(2/j) for j,i in zip(neurons_per_layer[1:], neurons_per_layer[:-1])]
        self.bias = [np.random.randn(j)*np.sqrt(2/j) for j in neurons_per_layer[1:]]

        self.cost = cost
        self.cost_deriv = self.cost + '_deriv'

        self.metric = metric
    
    def _cost(self, a, y):
        return eval(self.cost + '(a,y)')


    def _cost_deriv(self, a, y):
        return eval(self.cost_deriv + '(a,y)')
    
    def _metric(self, y_preds, y_true):
        return eval(self.metric + "(y_preds, y_true)")


    def train(self, X_train, y_train, epochs, eta):
        cost_avg = 0
        n_train = len(X_train)
        
        for epoch in range(epochs):

            i = 0
            acc = 0
            train_predictions = []
            losses = []
            while i < n_train:

                activations = []
                zs = []
                losses = []
                delta_errors = []

                input_ = X_train[i]

                activations.append(input_)
                for layer in self.layers[1:]:
                    l = self.layers.index(layer)
                    z, activation = layer.feed_forward(activations[l-1], self.weights[l-1], self.bias[l-1])
                    zs.append(z)
                    activations.append(activation)

                
                train_predictions.append(activations[-1])
                loss = self._cost(activations[-1], y_train[i])
                loss_deriv = self._cost_deriv(activations[-1], y_train[i])
                losses.append(loss)

                output_error = loss_deriv # already takes into account the derivative of the activation func
                delta_errors.append(output_error)
                
                depth = len(self.layers[:-1])

                for layer in reversed(self.layers[:-1]):
                    l = self.layers.index(layer)
                    self.bias[l] -= eta*delta_errors[depth-l-1]
                    
                    if layer.activation_fun == "input_layer":
                        self.weights[l] -= eta*np.outer(input_, delta_errors[depth-l-1]).T
                    else:
                        self.weights[l] -= eta*np.outer(activations[l], delta_errors[depth-l-1]).T    
                        delta_error = delta_errors[depth-l-1].dot(self.weights[l])*layer.activate_deriv(zs[l-1])
                        delta_errors.append(delta_error)   
                
                i += 1
            
            y_pred = np.array(train_predictions)
            cost_avg = np.mean(losses)
            acc = self._metric(y_pred, y_train)
            print("Epoch: {0} ----- Cost: {1} ----- Train Accuracy: {2}".format(epoch+1, round(cost_avg,2), round(acc,3)))
                

    def predict(self, X_test, y_test):

        n_test = len(X_test)
        preds = []
        acc = 0

        for i in range(n_test):

            activations = []
            zs = []
            input_ = X_test[i]
            activations.append(input_)
            

            for layer in self.layers[1:]:
                l = self.layers.index(layer)
                z, activation = layer.feed_forward(activations[l-1], self.weights[l-1], self.bias[l-1])
                zs.append(z)
                activations.append(activation)

            preds.append(activations[-1])
            #if y_test != None:
            y_pred = np.array(preds)
        acc = self._metric(y_pred, y_test)
        print("Testing set accuracy: ", acc)
            #return preds, acc
        #return preds





class DenseLayer(object):
    def __init__(self, n_neurons, activation_fun = True):

    
        self.n_neurons = n_neurons

        if activation_fun is True:
            self.activation_fun = "input_layer"
        else:
            self.activation_fun = activation_fun


        self.activation_fun_deriv = self.activation_fun + "_deriv"

    def feed_forward(self, x, weights, bias):
        
        z = weights.dot(x) + bias
        activation = eval(self.activation_fun + '(z)')
        return z, activation

    def update_parameters(self, activation, delta_error, weights, bias, eta):
        weights = weights - eta*np.outer(activation, delta_error).T
        bias = bias - eta*delta_error

    def activate_deriv(self, z):
        return eval(self.activation_fun_deriv + '(z)')
