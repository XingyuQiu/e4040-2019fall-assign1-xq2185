import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                            #         
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    h = 1/(1+np.exp(-x))
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = 0

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    # There are only two dimensions; So, first we want to get dimesions
    N,D = X.shape
    C = W.shape[1]
    # We set index as i,j
    prob = np.zeros((N,C))
    dW1 = np.zeros((D,C))
    for i in range(N):
      X_i = X[i]
      y_label = y[i]
      # Get the f-function
      f_i = np.dot(X[i],W)
      e_i = np.exp(f_i)
      for j in range(C):
        # e value of every i,j
        e_ij = e_i[j]
        prob[i,j] = e_ij/e_i.sum()
        if y_label == j:
          pred_j = e_ij/e_i.sum()
      # calculate the loss
      loss_i = -(y_label*np.log(pred_j)+(1-y_label)*np.log(1-pred_j))
      loss += loss_i
    loss = loss/N +  reg * np.sum(W**2)
    # Compute the derivative 
    prob[np.arange(N),y] -= 1
    dW = np.dot(X.T, prob)/N
    dW += 2*reg * W

    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = None
    # Initialize the gradient to zero
    dW = None

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no    # 
    # explicit loops.                                       #
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                       #
    ############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    # There are only two dimensions; So, first we want to get dimesions
    N,D = X.shape
    C = W.shape[1]
    # Calculate f-function defined in homework
    f_all = np.dot(X,W)
    # Find the true value of f
    f_real = f_all[np.arange(N),y]
    # Try to find C for solving numerical instability
    e = np.exp(f_all)
    # Calculate the probability
    prob_all = e/np.sum(e,axis=1,keepdims=True)
    new_y = np.array([y,1-y])
    # Try to calculate the loss for it
    loss = - np.sum(np.dot(np.log(prob_all)[np.arange(N),y],y)+np.dot(np.log(prob_all)[np.arange(N),1-y],1-y))/N + reg * np.sum(W**2)
    # Calculate the gradient
    prob_all[np.arange(N),y] -= 1
    dW = np.dot(X.T,(prob_all))/N +  2*reg*W
    

    
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
