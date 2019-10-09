import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # TODO: Implement PCA by extracting        #
    # eigenvector.You may need to sort the      #
    # eigenvalues to get the top K of them.     #
    ###############################################
    ###############################################
    #          START OF YOUR CODE         #
    ###############################################
    # Get the zero mean
    mean_val = np.mean(X,axis=0)
    new_X = X - mean_val
    # Get the covariance matrix
    cov_mat = np.cov(new_X,rowvar=0) # Here use rowvar to make sure row as a sample
    # Get the engine value
    Q,sigma,V = np.linalg.svd(cov_mat)
    # sigma is score and Q is principle components
    T = sigma[:K]
    P = Q[:,:K].T

    ###############################################
    #           END OF YOUR CODE         #
    ###############################################
    
    return (P, T)
