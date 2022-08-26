import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def relu_prime(Z):
    '''
    Z - weighted input matrix

    Returns gradient of Z where all
    negative values are set to 0 and
    all positive values set to 1
    '''
    Z[Z < 0] = 0
    Z[Z > 0] = 1
    return Z


def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)


def accuracy(y, y_hat):
    return np.sum(y == y_hat) / y.size


def one_hot_encode(y, n_values=None):
    if n_values is None:
        n_values = np.max(y) + 1
    return np.eye(n_values)[y]


def pred_to_label(y):
    return np.argmax(y, 0)


def print_acc(epoch, acc, n_digits=4):
    print(f"{'Epoch:':<10} {epoch:<30}")
    print(f"{'Accuracy:':<10} {acc:<30.{n_digits}f}")
    print()
