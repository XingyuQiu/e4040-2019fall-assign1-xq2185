from builtins import range
from builtins import object
import numpy as np

from utils.layer_funcs import *
from utils.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        ####################################################
        # TODO: Feedforward                      #
        ####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        new_X = X
        # Use for loop to calculate feedforward
        for layer in layers:
            new_X = layer.feedforward(new_X)
        loss,dout = softmax_loss(new_X,y)
        
        
        ####################################################
        # TODO: Backpropogation                   #
        ####################################################
        back_out = dout
        for reverse_layer in layers[::-1]:
            back_out = reverse_layer.backward(back_out)
        
        
        ####################################################
        # TODO: Add L2 regularization               #
        ####################################################
        reg_loss = 0
        for layer in layers:
            reg_loss += np.sum(layer.params[0]**2)
        # We can get final loss function here
        loss = loss+reg*reg_loss

        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ####################################################
        # TODO: Use SGD to update variables in layers.    #
        ####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        # At first ,initilize all the params
        params,grads = [],[]
        layers = self.layers
        num_layers = len(layers)
        # Get all params needed
        for layer in layers:
            params += layer.params
            grads += layer.gradients
        # Add regularization
        reg = self.reg
        grads = [grad + reg * params[i] for i, grad in enumerate(grads)]
        # Update params
        for i in range(len(params)):
            params[i] -= learning_rate * grads[i]
        
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #####################################################
        # TODO: Remember to use functions in class       #
        # SoftmaxLayer                          #
        #####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        # Initialze
        input_X = X
        for layer in layers:
            input_X = layer.feedforward(input_X)
        predictions = np.argmax(input_X,axis=1)
        
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc