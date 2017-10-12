"""
------------------------------------------------------------------
Jake Tutmaher
September 20, 2016
------------------------------------------------------------------
"""

import numpy as np
import random

"""
K - nearest neighbors (KNN) class. KNN is a "memory only"
approach - analogous to a lookup method for a given training
set. Typical accuracies of 40 percent (best case).
--
"""

class knn:
    def __init__(self):
        pass

    def train(self,X,Y):
        """
        Store Training Data and Labels in Class
        :param X: 2D numpy array (floats), training images
        :param Y: 2D numpy array (ints), training labels
        :return: N/A
        """
        self.Xtr = X
        self.Ytr = Y

    def predict(self,dist,k=1):
        """
        Find Min Value Indices in distance array, return
        corresponding values for training labels
        :param dist: 2D array (floats), distance array
        :param k: Int, number of nearest neighbors
        :return: 1D array (floats), predictions
        """
        idx = np.argsort(dist,axis=1)
        predict = [np.argmax(np.bincount(self.Ytr[idx[x,:k]]))
                   for x in range(dist.shape[0])]
        return np.array(predict)

    def distance(self,X):
        """
        Compute the distance matrix for the test set
        :param X: 2D array (floats), test images
        :return: 2D array (floats), num_test x num_train
        """
        #GET NUM TEST IMAGES
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        #INITIALIZE DIST ARRAY
        darray = np.zeros((num_test,num_train))
        #COMPUTE DIST ARRAY - 1 LOOP
        for x in range(num_train):
            currIm = self.Xtr[x,:]
            slice = np.repeat(currIm[:,np.newaxis],num_test,axis=1)
            diff = np.sum(np.abs((slice - X.transpose())),axis=0)
            darray[:,x] = diff
        return darray

    def check(self,X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        #INITIALIZE DISTANCE ARRAY
        darray = np.zeros((num_test,num_train))
        for x in range(num_train):
            for y in range(num_test):
                darray[y,x] = np.sum(np.abs(self.Xtr[x,:]-X[y,:]))

        return darray

    def accuracy(self, Ypred, Yact):
        """
        Get Accuracy of KNN Training Run
        :param Ypred: 1D array (floats), predicted values for test
        :param Yact: 1D array (floats), actual values for test
        :return: Float, number of correct predictions
        """
        num_correct = np.sum(Ypred==Yact)
        return np.divide(num_correct,float(Yact.size))

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
        :param classes: int, num classes for SVM model
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