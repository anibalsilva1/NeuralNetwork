import numpy as np
import tensorflow as tf

from nn_new import NeuralNetwork, DenseLayer
from activations import *
from cost_functions import *
from metrics import *



def onehot_encode(y):
    b = np.zeros((y.size, y.max()+1))
    b[np.arange(y.size), y] = 1
    return b


def main():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    n_obs = x_train.shape[0]
    n_obs_test = x_test.shape[0]
    X_train = x_train.reshape(n_obs, 28*28)/np.max(x_train)
    X_test = x_test.reshape(n_obs_test, 28*28)/np.max(x_test)
    y_train = onehot_encode(y_train)
    y_test = onehot_encode(y_test)
    X_testing = X_train[:1000,:]
    y_testing = y_train[:1000,:]
    X_testing_test = X_test[:50,:]
    y_testing_test = y_test[:50,:]
    nn = NeuralNetwork()
    nn.add_layer(DenseLayer(n_neurons = X_testing.shape[1]))
    nn.add_layer(DenseLayer(n_neurons=500, activation_fun="ReLU"))
    nn.add_layer(DenseLayer(n_neurons=100, activation_fun="ReLU"))
    nn.add_layer(DenseLayer(n_neurons=y_testing.shape[1],activation_fun="softmax"))
    nn.compile_model(cost = "cross_entropy", metric = "multi_categorical_accuracy")    
    nn.train(X_train = X_testing, y_train = y_testing, epochs = 2, eta = 10**(-3))
    #nn.predict(X_testing_test, y_testing_test)

if __name__ == "__main__":
    main()
