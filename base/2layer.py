"""
------------------------------------------------------------------
Jake Tutmaher
September 20, 2016
------------------------------------------------------------------
"""

import numpy as np
import random

"""
Two layer neural network with relu activation function. Fully connected only
"""
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Neural net archietecture
        :param input_size: int, feature vector size
        :param hidden_size: int, hidden layer size
        :param output_size: int, number of classes
        :param std: float, standard deviation of initialized weights
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Determine loss and gradient correction for batch of training data
        :param X: 2D array (floats), training data batch
        :param y: 1D array (ints), actual class (0-9)
        :param reg: float, regularization coefficient
        :return: float & dict, total loss and hashmap of gradient/bias correction
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        # Compute the forward pass
        # Input layer
        layer1 = X.dot(W1) + b1
        # RELU
        layer1[layer1 <= 0.0] = 0.0
        # Output layer - scores
        scores = layer1.dot(W2) + b2
        # Compute the loss
        num_train = scores.shape[0]
        numerator = np.exp(scores)
        denominator = np.sum(numerator, axis=1)
        # Divide 2d array (numerators) by 1d array (denominators)
        ypred_softmax = numerator / denominator[:, None]
        L_i = np.log(ypred_softmax)
        # Calculate total loss
        loss = -np.sum(L_i[np.arange(num_train), y])
        # Average loss over training set
        loss /= num_train
        # Get regularization
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        # Backward pass: compute gradients
        grads = {}
        # Calculate hidden layer corrections
        ypred_softmax[np.arange(num_train), y] -= 1
        dW2 = layer1.T.dot(ypred_softmax)
        dW2 /= num_train
        # Calculate input layer corrections
        # Move back one layer, apply relu gradient
        inter = ypred_softmax.dot(W2.T)
        inter[layer1 <= 0.0] = 0.0
        # Calculate first layer correction
        dW1 = X.T.dot(inter)
        dW1 /= num_train
        # Regularize
        dW1 += reg * W1
        dW2 += reg * W2
        # Catalogue gradients
        grads["W1"] = dW1
        grads["W2"] = dW2
        grads["b1"] = np.sum(inter / num_train, axis=0)
        grads["b2"] = np.sum(ypred_softmax / num_train, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Training method using stochastic gradient descent
        :param X: 2D array (floats), training data
        :param y: 1D array (ints), actual training classes
        :param X_val: 2D array (floats), validation data
        :param y_val: 1D array (ints), actual validation classes
        :param learning_rate: float, learning rate for gradient descent
        :param learning_rate_decay: float, decay factor for learning rate
        :param reg: float, regularization coefficient
        :param num_iters: int, number of training cycles
        :param batch_size: int, size of each training batch
        :param verbose: boolean, controls print out
        :return: dict, arrays of loss, train_accuracy, and validation accuracy
        """
        # Determine training data and corresponding epochs
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        # Initialize output arrays
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        # Set up minibatch of training indices for SGD
        minibatch = np.min([int(num_train * 0.8), batch_size])
        idx = range(num_train)
        # Main training loop
        for it in xrange(num_iters):
            # Select random index from batch
            random_idx = random.sample(idx, minibatch)
            X_batch = np.array(X[random_idx])
            y_batch = np.array(y[random_idx])
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            # Update Gradient
            self.params["W1"] -= learning_rate * grads["W1"]
            self.params["W2"] -= learning_rate * grads["W2"]
            self.params["b1"] -= learning_rate * grads["b1"].flatten()
            self.params["b2"] -= learning_rate * grads["b2"].flatten()
            # Print results
            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        # Layer 1
        H1 = X.dot(self.params["W1"]) + self.params["b1"]
        # RELU
        H1[H1 <= 0.0] = 0.0
        # Raw Output
        scores = H1.dot(self.params["W2"]) + self.params["b2"]
        # Softmax
        numerator = np.exp(scores)
        denominator = np.sum(numerator, axis=1)
        y_pred = np.argmax(numerator / denominator[:, None], axis=1)

        return y_pred