import numpy as np

# Cost functions

def cross_entropy(a, y):
    return -np.sum(np.nan_to_num(y*np.log2(a) + (1-y)*np.log2(1-a)))


def cross_entropy_deriv(a, y):
    return a - y


def MSE(a, y):
    return np.sum((a-y)**2)


def MSE_deriv(a, y):
    return a - y
