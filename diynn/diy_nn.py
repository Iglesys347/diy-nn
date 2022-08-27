import numpy as np

from diynn.utils import accuracy, one_hot_encode, pred_to_label, print_acc, relu, relu_prime, softmax


class DIYNN():
    def __init__(self, inp_layer_size, hidden_layer_size, output_layer_size, bias=0.1, seed=None) -> None:
        self.n = inp_layer_size
        self.p = hidden_layer_size
        self.q = output_layer_size
        # updated with the dataset (in `backward_prop` or `train` methods)
        self.m = None
        
        # random number generator used for initialisation methods and to shuffle in train_minibatch
        self.rng = np.random.default_rng(seed)

        self.W_h, self.W_o = self._init_weights()
        self.b_h, self.b_o = self._init_bias(bias=bias)

        # H, Z_h and Z_o are initialized to None and updated in forward_prop
        self.H, self.Z_h, self.Z_o = None, None, None

    def _init_weights(self):
        # He initialisation of the weights ; there is probably a better init method
        W_h = self.rng.standard_normal((self.p, self.n)) * np.sqrt(2.0/self.n)
        W_o = self.rng.standard_normal((self.q, self.p)) * np.sqrt(2.0/self.p)
        return W_h, W_o

    def _init_bias(self, bias=0.1):
        b_h = np.full((self.p, 1), bias)
        b_o = np.full((self.q, 1), bias)
        return b_h, b_o

    def _forward_prop(self, X):
        # Hidden layer
        self.Z_h = np.dot(self.W_h, X) + self.b_h
        self.H = relu(self.Z_h)

        # Output layer
        self.Z_o = np.dot(self.W_o, self.H) + self.b_o

        y_hat = softmax(self.Z_o)
        return y_hat

    def _backward_prop(self, X, y, y_hat, learning_rate):
        self.m = X.shape[1]
        # y_hat has a shape of (m, q) while y has a shape of (m, 1)
        # to make y and y_hat match we need to one hot encode y
        y = one_hot_encode(y, n_values=self.q)

        # Layer Error
        E_o = (y_hat - y.T)
        E_h = np.dot(self.W_o.T, E_o) * relu_prime(self.Z_h)

        # Cost derivative for weights
        dW_o = 1 / self.m * np.dot(E_o, self.H.T)
        dW_h = 1 / self.m * np.dot(E_h, X.T)

        # Bias derivative
        db_o = 1 / self.m * np.sum(E_o)
        db_h = 1 / self.m * np.sum(E_h)

        # Update weights
        self.W_h -= learning_rate * dW_h
        self.W_o -= learning_rate * dW_o

        # Update bias
        self.b_h -= learning_rate * db_h
        self.b_o -= learning_rate * db_o

    def train(self, X, y, learning_rate=0.1, n_iter=50, verbose=True, print_every=50):
        self.init_weights()
        self.init_bias()
        for i in range(n_iter):
            y_hat = self._forward_prop(X)
            self._backward_prop(X, y, y_hat, learning_rate=learning_rate)
            if verbose and i % print_every == 0:
                print_acc(i, accuracy(y, pred_to_label(y_hat)))

    def train_minibatch(self, X, y, learning_rate=0.1, n_iter=50, batch_size=1, verbose=True, print_every=50):
        self.init_weights()
        self.init_bias()
        if self.m is None:
            self.m = X.shape[1]

        if not 0 < batch_size <= self.m:
            raise ValueError(
                "The `batch_size` must be between 0 and the number of observations.")

        for i in range(n_iter):
            # shuffling the two arrays
            perms = self.rng.permutation(self.m)
            X_perm, y_perm = X[:, perms], y[perms]
            for start in range(0, self.m, batch_size):
                stop = start + batch_size
                X_batch, y_batch = X_perm[:, start:stop], y_perm[start:stop]
                y_hat = self._forward_prop(X_batch)
                self._backward_prop(X_batch, y_batch, y_hat,
                                   learning_rate=learning_rate)

            if verbose and i % print_every == 0:
                y_hat = self._forward_prop(X)
                print_acc(i, accuracy(y, pred_to_label(y_hat)))

    def predict(self, X):
        return pred_to_label(self._forward_prop(X))
