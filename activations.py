import numpy as np

def ReLU(z):
    return np.maximum(z, 0)


def sigmoid(z):
    return (1+np.exp(-z))**(-1)


def TanH(z):
    return (np.exp(z) - np.exp(-z))/2


def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

##### Derivatives ######


def ReLU_deriv(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))


def TanH_deriv(z):
    return 1 - TanH(z)**2


def softmax_deriv(z):
    exp = np.exp(z)
    return exp*(1/np.sum(exp) - exp/np.sum(exp)**2)
