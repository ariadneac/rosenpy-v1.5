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


class FCRBFNN(rpnn.NeuralNetwork):
    """
    Specification for the Fully Complex Transmittance Radial Basis Function Neural Network 
    to be passed to the model in construction.
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
        # Set layer input
        self.layers[0].input = self.xp.transpose(self.xp.tile(x, (self.layers[0].neurons,1,1)), axes=[1, 0, 2])
        
        # Calculate the the distance between the input point and each center of the radial basis function
        self.layers[0].seuc = self.layers[0].input - self.xp.tile(self.layers[0].gamma, (self.layers[0].input.shape[0], 1,1))
        
        self.layers[0].kern = self.xp.sum(self.xp.multiply(self.layers[0].sigma, self.layers[0].seuc), axis=2)
        
        # Apply activation function
        self.layers[0].phi = actFunc.sech (self.layers[0].kern)
        
        # Calculate the output of the layer
        self.layers[0]._activ_out = self.xp.dot(self.layers[0].phi, self.layers[0].weights) + self.layers[0].biases
        
        # Return the output of layer
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
        error = y - y_pred
        
        for layer in reversed(self.layers):
            phi_l = actFunc.sech(layer.kern, derivative=True)
      
            a = self.xp.multiply(self.xp.dot(error, self.xp.conj(layer.weights).T), self.xp.conj(phi_l))
            
            # Compute the regularization l2
            regl2 = (regFunc.l2_regularization(layer.lambda_init, layer.reg_strength, epoch))
            
            # Update weights and biases
            layer._dweights =self.xp.dot(layer.phi.T, error) - (regl2 if layer.reg_strength else 0)*layer.weights
            layer._prev_dweights = layer._dweights*self.learning_rate + self.momentum*layer._prev_dweights 
            layer.weights = layer.weights + layer._prev_dweights

            layer._dbiases = self.xp.divide(sum(error), error.shape[0])  - (regl2 if layer.reg_strength else 0)*layer.biases
            layer._prev_dbiases = layer._dbiases*self.learning_rate + self.momentum*layer._prev_dbiases
            layer.biases = layer.biases + layer._prev_dbiases
    
            # Update sigma and gamma
            g_a = self.xp.multiply(a[:, :, self.xp.newaxis],self.xp.tile(self.xp.conj(layer.sigma), (a.shape[0],1,1)))
            layer._dgamma = -self.xp.divide(sum(g_a), g_a.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.gamma
            layer._prev_dgamma = layer._dgamma*layer.gamma_rate + self.momentum*layer._prev_dgamma
            layer.gamma = layer.gamma + layer._prev_dgamma 
    
            s_a = a[:, :, self.xp.newaxis] * self.xp.conj(layer.seuc)
            layer._dsigma = self.xp.divide(sum(s_a), s_a.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.sigma
            layer._prev_dsigma = layer._dsigma*layer.sigma_rate + self.momentum*layer._prev_dsigma
            layer.sigma = layer.sigma + layer._prev_dsigma
    

    
    def addLayer(self, ishape, neurons, oshape, weights_initializer=initFunc.random_normal, bias_initializer=initFunc.ones, 
                 gamma_initializer=initFunc.rbf_default, sigma_initializer=initFunc.rbf_default, 
                 reg_strength=0.0, lambda_init=0.1, gamma_rate=0.01, sigma_rate=0.01):
        """
        The method is responsible for adding the layers to the neural network.

        Parameters
        ----------
        ishape : int
            The number of neurons in the first layer (the number of input features).
        neurons : int
            The number of neurons in the hidden layer. 
        oshape : int
            The oshape is a specific argument for the RBF networks; it is the number of 
            neurons in the output layer. 
        weights_initializer : str, optional
            It defines the way to set the initial random weights, as string. The default is initFunc.random_normal.
        bias_initializer : str, optional
            It defines the way to set the initial random biases, as string. The default is initFunc.ones_real.
        sigma_initializer : str, optional
            It defines the way to set the initial random sigma, as string. The default is initFunc.ones.
        gamma_initializer : str, optional
            It defines the way to set the initial random gamma, as string. The default is initFunc.rbf_default.
        reg_strength : float, optional
            It sets the regularization strength. The default value is 0.0, which means
            that regularization is turned off. The default is 0.0.
        lambda_init : float, optional
            It is the initial regularization factor strength. The default is 0.1.
        gamma_rate : float, optional
            The learning rate of matrix of the center vectors. The default is 0.01.
        sigma_rate : float, optional
            The learning rate of the vector of variance. The default is 0.01.

        Returns
        -------
        None.

        """
        self.layers.append(rplayer.Layer(ishape, neurons, oshape, 
                                          weights_initializer=weights_initializer, 
                                          bias_initializer=bias_initializer, 
                                          gamma_initializer=gamma_initializer, 
                                          sigma_initializer=gamma_initializer,
                                          reg_strength=reg_strength, 
                                          lambda_init=lambda_init, 
                                          sigma_rate=sigma_rate,
                                          gamma_rate=gamma_rate,
                                          cvnn=3,
                                          xp=self.xp))
               