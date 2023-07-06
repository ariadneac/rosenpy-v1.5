# -*- coding: utf-8 -*-
"""**RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks**.
*Copyright © A. A. Cruz, K. S. Mayer, D. S. Arantes*.

*License*

This file is part of RosenPy.
RosenPy is an open source framework distributed under the terms of the GNU General 
Public License, as published by the Free Software Foundation, either version 3 of 
the License, or (at your option) any later version. For additional information on 
license terms, please open the Readme.md file.

RosenPy is distributed in the hope that it will be useful to every user, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License
along with RosenPy.  If not, see <http://www.gnu.org/licenses/>.
"""
from rputils import regFunc, initFunc, actFunc
from rosenpymodel import rplayer, rpnn

class SCFFNN(rpnn.NeuralNetwork):   
    """
    Specification for the Split Complex FeedForward Neural Network to be passed 
    to the model in construction.
    This includes the feedforward, backpropagation and adding layer methods specifics.
    
    This class derives from NeuralNetwork class.    
    """
    def feedforward(self, x):
        """
        This method returns the output of the network if ``x`` is input.
        
        Parameters
        ----------
            x: array-like, shape (n_batch, n_inputs)
            
            Training vectors as real numbers, where n_batch is the
            batch and n_inputs is the number of input features.
        
        Returns
        -------
              y_pred: array-like, shape (n_batch, n_outputs) 
              
              The output of the last layer.
        
        """
        # Set the input of the first layer
        self.layers[0].input = x
        
        # Compute the sum of the products between the inputs and weights, and then add the biases of the first layer
        self.layers[0]._activ_in = self.xp.dot(self.layers[0].input, self.layers[0].weights) + self.layers[0].biases
        
        # Apply separate activation functions to the real and imaginary components of the output from the first layer
        self.layers[0]._activ_out = actFunc.splitComplex(self.layers[0]._activ_in, self.layers[0].activation, derivative=False)
        
        # Iterate through the remaining layers
        for i in range(1, len(self.layers)):
            # Set the input of the current layer as the output of the previous layer
            self.layers[i].input = self.layers[i - 1]._activ_out
            
            # Compute the sum of the products between the inputs and weights, and then add the biases of the current layer
            self.layers[i]._activ_in = self.xp.dot(self.layers[i].input, self.layers[i].weights) + self.layers[i].biases
            
            # Apply separate activation functions to the real and imaginary components of the output from the current layer
            self.layers[i]._activ_out = actFunc.splitComplex(self.layers[i]._activ_in, self.layers[i].activation, derivative=False)
        
        # Return the output of the last layer
        return self.layers[-1]._activ_out
    
    
    def backprop(self, y, y_pred, epoch):
        """
        This class provids a way to calculate the gradients of a target class output.

        Parameters
        ----------
        y : array-like, shape (n_samples, n_outputs)
            Target values are real numbers representing the desired outputs.
        y_pred : array-like, shape (n_samples, n_outputs)
            Target values are real numbers representing the predicted outputs.
        epoch : int
            Current number of the training epoch for updating the smoothing factor. 

        Returns
        -------
        None.

        """
        e = y - y_pred
       
        deltaDir, auxW = None, 0
        
        for layer in reversed(self.layers):
            # Calculate derivatives that are separately applied to the real and imaginary components
            deriv = actFunc.splitComplex(layer._activ_in, layer.activation, derivative=True)
            
            # Calculate deltaDir
            if deltaDir is not None: # others layers
                dot = self.xp.dot(deltaDir, self.xp.conjugate(auxW.T))
                
                deltaDir = self.xp.multiply(self.xp.conjugate(deriv), dot)
            
            else: # last layer
                deltaDir = self.xp.multiply(self.xp.conj(deriv), e)
            
            # Update auxW
            auxW = layer.weights
            
            # Compute the regularization l2
            regl2 = (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch))
            
            # Update weights and biases
            layer._dweights = self.xp.dot(self.xp.conj(layer.input.T), deltaDir) - (regl2 if layer.reg_strength else 0)*layer.weights
            layer._prev_dweights = layer._dweights*self.learning_rate + self.momentum*layer._prev_dweights
            layer.weights += layer._prev_dweights
                   
         
            layer._dbiases =  self.xp.divide(sum(deltaDir), deltaDir.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.biases
            layer._prev_dbiases = layer._dbiases*self.learning_rate + self.momentum*layer._prev_dbiases
            layer.biases += layer._prev_dbiases
  
    def addLayer(self, neurons, ishape=0, weights_initializer=initFunc.random_normal, bias_initializer=initFunc.random_normal, activation=actFunc.tanh, reg_strength=0.0, lambda_init=0.1):
        """
        The method is responsible for adding the layers to the neural network.

        Parameters
        ----------
        neurons : int
            The number of neurons in the hidden layer. If the ishape is different zero and 
            it is the first layer of the model, neurons represents the number of neurons 
            in the first layer (the number of input features).
        ishape : int, optional
            The number of neurons in the first layer (the number of input features). The default is 0.
        weights_initializer : str, optional
            It defines the way to set the initial random weights, as a string. The default is initFunc.random_normal.
        bias_initializer : str, optional
            It defines the way to set the initial random biases, as a string. The default is initFunc.random_normal.
            Initialization methods were defined in the file rp_utils.initFunc.
            
            * rp_utils.initFunc.zeros
            * rp_utils.initFunc.ones
            * rp_utils.initFunc.ones_real
            * rp_utils.initFunc.random_normal
            * rp_utils.initFunc.random_uniform
            * rp_utils.initFunc.glorot_normal
            * rp_utils.initFunc.glorot_uniform
            * rp_utils.initFunc.rbf_default
            
        activation : str, optional
            Select which activation function this layer should use, as a string. The default is actFunc.tanh.
            Activation methods were defined in the file rp_utils.actFunc.
            
            * rp_utils.actFunc.sinh
            * rp_utils.actFunc.atanh
            * rp_utils.actFunc.asinh
            * rp_utils.actFunc.tan
            * rp_utils.actFunc.sin
            * rp_utils.actFunc.atan
            * rp_utils.actFunc.asin
            * rp_utils.actFunc.acos
            * rp_utils.actFunc.sech
            
        reg_strength : float, optional
            It sets the regularization strength. The default is 0.0., which means that regularization is turned off
        lambda_init : float, optional
            It is the initial regularization factor strength. The default is 0.1.

        Returns
        -------
        None.

        """
        
        self.layers.append(rplayer.Layer(ishape if not len(self.layers) else self.layers[-1].neurons, neurons, 
                                              weights_initializer=weights_initializer, 
                                              bias_initializer=bias_initializer, 
                                              activation=activation, 
                                              reg_strength=reg_strength, 
                                              lambda_init=lambda_init, 
                                              cvnn=1))