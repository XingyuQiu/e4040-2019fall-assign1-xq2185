import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    # Calculate the shape
    N,D = X.shape 
    C = W.shape[1]
    # Create a matrix to store variable
    prob_all = np.zeros((N,C))
    
    #Use naive method to calculate result
    for i in range(N):
      # Walk through first row
      X_i = X[i]
      # Get prediction
      f_i = X_i.dot(W)
      # Common choice for C
      log_C = -np.max(f_i)
      # Find for real y
      y_label = y[i]
      real_y = f_i[y_label]

      for j in range(C):
        f_ij = f_i[j]
        # Add C to solve for numerical instability
        C_ij = np.exp(log_C+f_ij)
        if j == y_label:
          pred_j = C_ij
        # Get probability in row i for all j
        prob_all[i,j] = C_ij
      # Get the loss in row i
      L_i = -np.log(pred_j/np.sum(prob_all[i]))
      loss+=L_i
      prob_all[i] = prob_all[i]/np.sum(prob_all[i])
      # Get the derivative for specific function
      prob_all[i,y_label] -= 1
    # Calculate the final loss
    loss = loss/N
    reg_loss = reg*np.sum(W**2)
    loss += reg_loss
    # Compute the gradient 
    dW = np.dot(X.T,(prob_all))/N + 2*reg*W
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    # Find dimensions
    N,D = X.shape
    C = W.shape[1]
    # Calculate f-function defined in homework
    f_all = np.dot(X,W)
    # Find the true value of f
    f_real = f_all[np.arange(N),y]
    # Try to find C for solving numerical instability
    log_C = -np.max(f_all, axis = 1,keepdims = True)
    e = np.exp(f_all+log_C)
    # Calculate the probability
    prob_all = e/np.sum(e,axis=1,keepdims=True)

    # Try to calculate the loss for it
    loss = np.sum(-np.log(prob_all[np.arange(N),y]))/N + reg * np.sum(W**2)
    # Calculate the gradient
    prob_all[np.arange(N),y] -= 1
    dW = np.dot(X.T,(prob_all))/N +  2*reg*W

    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
