"""
------------------------------------------------------------------
Jake Tutmaher
September 20, 2016
------------------------------------------------------------------
"""

import numpy as np
import random

"""
Support vector machine class. Generalization of a binary support vector machine
algorithm. Utilizes the generalized loss function for arbitrary classification.
"""

class svm:
    def __init__(self,featurelength,numclasses):
        self.weights = np.random.rand(featurelength,numclasses)*0.0001
        self.model = np.zeros((featurelength,numclasses))
        return

    def train(self,Xtr,Ytr,learning_rate=1e-7, reg=5e4,batch_size=100):
        """
        Train SVM linear model based on training set
        :param Xtr: 2D array (floats), training data
        :param Ytr: 1D array (int), training lables
        :param learning_rate: float, correction factor to gradient
        :param reg: float, for l2 regularization
        :param batch_size: int, num images for SGD
        :return: 2D array, final weight matrix "model"
        """
        # Training size and loss array initialization
        num_train = Xtr.shape[0]
        Weights = np.copy(self.weights)
        loss_array = []
        # Iterate through batches - train model
        for x in range(0,num_train,batch_size):
            batch_x = Xtr[x:x+batch_size,:]
            batch_y = Ytr[x:x+batch_size]
            loss,dW = self.loss(Weights,batch_x,batch_y,reg)
            loss_array.append(loss)
            Weights -= learning_rate*dW
        # Store model for testing
        self.model = Weights
        return loss_array

    def loss(self,W,X,y,reg):
        """
        Computes total loss and gradient matrix averages for all trials
        :param W: 2D array (floats), weight matrix
        :param X: 2D array (floats), feature vectors for all trials
        :param y: 1D array (floats), correct classes for all trials
        :param reg: regularization factor for l2 correction
        :return: float & 2D array (floats), loss and gradient matrix
        """
        # Iterate through training set
        num_train = X.shape[0]
        loss,dW = self.__batch_gradient(X,y,W)
        # Average loss
        loss /= num_train
        # Average gradients as well
        dW /= num_train
        # Add regularization to the loss.
        loss += 0.5 * reg * np.sum(W * W)
        # Add regularization to the gradient
        dW += reg * W
        # Return
        return loss,dW

    def predict(self,X):
        """
        Make predictions based on Current Model
        :param X: 2D array (floats), set of test/validation images
        :return: 1D array (floats), predictions
        """
        y = X.dot(self.model)
        predict = np.argmax(y,axis=1)
        return predict

    def weights(self):
        return self.weights

    def accuracy(self,Ypr,Yact):
        return np.mean(Ypr==Yact)

    def model(self):
        return self.model

    def __batch_gradient(self,X,y,W):
        """
        Computes total loss and gradient matrix for a single training
        note: meant as private method.
        instance
        :param y: float, Correct class
        :param W: 2D array (floats), weight matrix
        :return: float & 2D array (floats), loss for trial and gradient
                 matrix
        """
        #Training Size
        num_train = X.shape[0]
        # Predictions
        scores = X.dot(W)
        # Loss term
        correct_scores = scores[np.arange(num_train),y]
        # Compute loss function for all terms
        margins = scores - correct_scores[:,None] + 1
        margins[margins < 0] = 0
        margins[np.arange(num_train),y] = 0
        # Total loss for trial
        loss = np.sum(margins)
        # Generate Gradient scaling terms
        new_margins = np.copy(margins)
        # Set finite margins to 1 - counting function
        new_margins[new_margins > 0] = 1
        new_margins[np.arange(num_train),y] = - np.sum(new_margins,axis=1)
        # Generate gradient matrix
        dW = X.T.dot(new_margins)

        return loss, dW