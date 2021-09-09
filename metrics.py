import numpy as np


def multi_categorical_accuracy(y_pred, y_true):
    res = np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    return np.mean(res)

def multi_target_accuracy(y_pred, y_true):
    res = np.mean(np.all(y_true.astype(int) == y_pred.round().astype(int), axis = 1))
    return res