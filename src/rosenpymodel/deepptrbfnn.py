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
from rputils import regFunc, initFunc
from rosenpymodel import rplayer, rpnn


class DeepPTRBFNN(rpnn.NeuralNetwork):   
    """
    Specification for the Deep Phase Transmittance Radial Basis Function Neural Network 
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
        self.layers[0].kern = self.layers[0].input - self.xp.tile(self.layers[0].gamma, (self.layers[0].input.shape[0], 1,1))
        
        # Calculate the squared Euclidean distance separately for the real and imaginary components
        aux_r = self.xp.sum(self.layers[0].kern.real**2, axis=2)
        aux_i = self.xp.sum(self.layers[0].kern.imag**2, axis=2)
        
        seuc_r = aux_r/self.layers[0].sigma.real
        seuc_i = aux_i/self.layers[0].sigma.imag
        
        self.layers[0].seuc = seuc_r + 1j*seuc_i
        
        # Activation measure for the neurons in the RBF layer, based on the proximity of the input point to the centers of the radial basis functions
        self.layers[0].phi = self.xp.exp(-seuc_r) + 1j*(self.xp.exp(-seuc_i))
        
        # Calculate the output of the layer
        self.layers[0]._activ_out = (self.xp.dot(self.layers[0].phi, self.layers[0].weights) + self.layers[0].biases)
        
        for i in range(1, len(self.layers)):
            # Set the input of the current layer as the output of the previous layer            
            self.layers[i].input = self.xp.transpose(self.xp.tile(self.layers[i - 1]._activ_out, (self.layers[i].neurons,1,1)), axes=[1, 0, 2])
            
            # Calculate the the distance between the input point and each center of the radial basis function
            self.layers[i].kern = self.layers[i].input - self.xp.tile(self.layers[i].gamma, (self.layers[i].input.shape[0], 1,1))
            
            # Calculate the squared Euclidean distance separately for the real and imaginary components
            aux_r = self.xp.sum(self.layers[i].kern.real*self.layers[i].kern.real, axis=2)
            aux_i = self.xp.sum(self.layers[i].kern.imag*self.layers[i].kern.imag, axis=2)
            
            seuc_r = aux_r/self.layers[i].sigma.real
            seuc_i = aux_i/self.layers[i].sigma.imag
            
            self.layers[i].seuc = seuc_r + 1j*seuc_i
            
            # Activation measure for the neurons in the RBF layer, based on the proximity of the input point to the centers of the radial basis functions
            self.layers[i].phi = self.xp.exp(-seuc_r) + 1j*(self.xp.exp(-seuc_i))
            
            # Calculate the output of the layer
            self.layers[i]._activ_out = (self.xp.dot(self.layers[i].phi, self.layers[i].weights) + self.layers[i].biases)
        
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
        error = y - y_pred
        last = True
        auxK = aux_r = aux_i = 0
        
        for layer in reversed(self.layers):
            psi = error if last else -self.xp.squeeze(self.xp.matmul(self.xp.transpose(auxK.real, (0, 2, 1)), aux_r[:, :, self.xp.newaxis]) + 1j * self.xp.matmul(self.xp.transpose(auxK.imag, (0, 2, 1)), aux_i[:, :, self.xp.newaxis]), axis=2)
            last = False
            auxK = layer.kern        
           
            epsilon = self.xp.dot(psi, self.xp.conj(layer.weights.T))
          
            beta_r = layer.phi.real/layer.sigma.real
            beta_i = layer.phi.imag/layer.sigma.imag
         
            aux_r  = epsilon.real * beta_r             
            aux_i  = epsilon.imag * beta_i 
            
            # Compute the regularization l2
            regl2 = (regFunc.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch))
            
            # Update weights and biases
            layer._dweights = self.xp.dot(self.xp.conj(layer.phi.T), psi) - (regl2 if layer.reg_strength else 0)*layer.weights
            layer._prev_dweights = layer._dweights*self.learning_rate + self.momentum*layer._prev_dweights 
            layer.weights = layer.weights + layer._prev_dweights
            
            layer._dbiases = self.xp.divide(sum(psi), psi.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.biases
            layer._prev_dbiases = layer._dbiases*self.learning_rate + self.momentum*layer._prev_dbiases
            layer.biases = layer.biases + layer._prev_dbiases
            
            # Update sigma and gamma
            s_a = self.xp.multiply(aux_r, layer.seuc.real) + 1j*self.xp.multiply(aux_i, layer.seuc.imag)
            layer._dsigma = self.xp.divide(sum(s_a), s_a.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.sigma
            layer._prev_dsigma = layer._dsigma*layer.sigma_rate + self.momentum*layer._prev_dsigma
            layer.sigma = layer.sigma + layer._prev_dsigma     
            
            g_a = self.xp.multiply(aux_r[:, :, self.xp.newaxis], layer.kern.real) + 1j*(self.xp.multiply(aux_i[:, :, self.xp.newaxis], layer.kern.imag))
            layer._dgamma = self.xp.divide(sum(g_a), g_a.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.gamma
            layer._prev_dgamma = layer._dgamma*layer.gamma_rate + self.momentum*layer._prev_dgamma
            layer.gamma = layer.gamma + layer._prev_dgamma 
            
            layer.sigma = self.xp.where(layer.sigma.real>0.0001, layer.sigma.real, 0.0001) + 1j*self.xp.where(layer.sigma.imag>0.0001, layer.sigma.imag, 0.0001)

    
    def addLayer(self, neurons, ishape=0, oshape=0, weights_initializer=initFunc.random_normal, bias_initializer=initFunc.ones, 
                 sigma_initializer=initFunc.ones, gamma_initializer=initFunc.rbf_default,
                 reg_strength=0.0, lambda_init=0.1, gamma_rate=0.01, sigma_rate=0.01):
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

        Parameters
        ----------
        neurons : int
            The number of neurons in the hidden layer. If the ishape is different zero and 
            it is the first layer of the model, neurons represents the number of neurons 
            in the first layer (the number of input features).
        ishape : int, optional
            The number of neurons in the first layer (the number of input features). The default is 0.
        oshape : int, optional
            The number of neurons in the last layer (the number of output features). 
        weights_initializer : str, optional
            It defines the way to set the initial random weights, as a string. The default is initFunc.random_normal.
        bias_initializer : str, optional
            It defines the way to set the initial random biases, as a string. The default is initFunc.ones.
        sigma_initializer : str, optional
            It defines the way to set the initial random sigma, as a string. The default is initFunc.ones.
        gamma_initializer : str, optional
            It defines the way to set the initial random gamma, as a string. The default is initFunc.rbf_default.
            Initialization methods were defined in the file rp_utils.initFunc.
            
            * rp_utils.initFunc.zeros
            * rp_utils.initFunc.ones
            * rp_utils.initFunc.ones_real
            * rp_utils.initFunc.random_normal
            * rp_utils.initFunc.random_uniform
            * rp_utils.initFunc.glorot_normal
            * rp_utils.initFunc.glorot_uniform
            * rp_utils.initFunc.rbf_default
        reg_strength : float, optional
            It sets the regularization strength. The default is 0.0., which means that regularization is turned off
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
        self.layers.append(rplayer.Layer(ishape if not len(self.layers) else self.layers[-1].oshape, neurons, neurons if not oshape else oshape, 
                                          weights_initializer=weights_initializer, 
                                          bias_initializer=bias_initializer, 
                                          sigma_initializer=sigma_initializer, 
                                          gamma_initializer=gamma_initializer,
                                          reg_strength=reg_strength, 
                                          lambda_init=lambda_init, 
                                          sigma_rate=sigma_rate,
                                          gamma_rate=gamma_rate,
                                          cvnn=4,
                                          xp=self.xp))
                