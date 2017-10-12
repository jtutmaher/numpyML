"""
------------------------------------------------------------------
Jake Tutmaher
September 20, 2016
------------------------------------------------------------------
"""

import numpy as np
import random

"""
Softmax classifier
"""

class softmax:
    def __init__(self,featurelength,classes):
        """
        Initialize weights and model for softmax class
        :param featurelength: float, length of feature vectors
        :param classes: float, number of classes
        """
        self.W = np.random.randn(featurelength,classes)*0.0001
        self.model = np.zeros((featurelength,classes))
        return

    def train(self,Xtr,Ytr,learning_rate=1e-7, reg=5e4,batch_size=100,grad="batch"):
        """
        Train softmax classifier based on training set
        :param Xtr: 2D array (floats), training data
        :param Ytr: 1D array (int), training lables
        :param learning_rate: float, correction factor to gradient
        :param reg: float, for l2 regularization
        :param batch_size: int, num images for SGD
        :param classes: int, num classes for SVM model
        :return: 2D array, final weight matrix "model"
        """
        # Training size and loss array initialization
        num_train = Xtr.shape[0]
        Weights = np.copy(self.W)
        loss_array = []
        # Iterate through batches - train model
        idx = range(num_train)
        for x in range(0,num_train,batch_size):
            # If stochastic - randomly select batch size training data
            if grad=="stochastic":
                rand = random.sample(idx,batch_size)
                batch_x = Xtr[rand]
                batch_y = Ytr[rand]
            # Else - go in order
            else:
                batch_x = Xtr[x:x+batch_size,:]
                batch_y = Ytr[x:x+batch_size]
            # Compute loss and gradients
            loss,dW = self.cross_entropy_loss(Weights,batch_x,batch_y,reg)
            loss_array.append(loss)
            Weights -= learning_rate*dW
        # Store model for testing
        self.model = Weights
        return loss_array

    def predict(self,X):
        """
        Make predictions based on Current Model
        :param X: 2D array (floats), set of test/validation images
        :return: 1D array (floats), predictions
        """
        y = X.dot(self.model)
        predict = np.argmax(y,axis=1)
        return predict

    def cross_entropy_loss(self,W,X, y, reg):
        """
        Cross entropy loss and gradient descent method
        :param W: 2D array (floats), initial weights
        :param X: 2D array (floats), batch feature vectors
        :param y: 1D array (int), list of classes
        :param reg: float, regularization factor
        :param grad: string, gradient descent method
        :return: float & 2D array (floats), batch loss and gradient
                 matrix
        """
        # Determine number of training data
        num_train = len(y)
        # Calculate softmax loss elements
        ypred = X.dot(W)
        numerator = np.exp(ypred)
        denominator = np.sum(numerator, axis=1)
        # Divide 2d array (numerators) by 1d array (denominators)
        ypred_softmax = numerator / denominator[:, None]
        L_i = np.log(ypred_softmax)
        # Calculate total loss
        loss = -np.sum([L_i[i, y[i]] for i in range(num_train)])
        # Calculate gradient
        dW = self.__batch_gradient(ypred_softmax, y, W, X)
        # Average loss over training set
        loss /= num_train
        # Get regularization
        loss += 0.5 * reg * np.sum(W * W)
        # Apply regularization correction
        dW += reg * W

        return loss, dW

    def weights(self):
        return self.weights

    def accuracy(self, Ypr, Yact):
        return np.mean(Ypr == Yact)

    def model(self):
        return self.model

    def __batch_gradient(self,ypred, yact, W, X):
        """
        Batch gradient method - average gradient over a number of
        feature vectors - no regularization
        :param ypred: 2D array (floats), predicted (softmax normalized) classes
        :param yact: 1D array (int), actual classes
        :param W: 2D array (floats), current weight matrix
        :param X: 2D array (floats), batch feature vectors
        :return: 2D array (floats), batch averaged gradient correction
        """
        # Determine batch number
        num_train = len(yact)
        # Construct actuals matrix via one-hot notation
        yact_mat = np.zeros(ypred.shape)
        yact_mat[np.arange(num_train),yact] = 1
        # Compute scaling coefficients - from gradient of loss function
        scale = ypred - yact_mat
        dW = X.T.dot(scale)
        # Average gradient matrix over batch data
        dW /= num_train

        return dW
