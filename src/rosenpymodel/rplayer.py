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
from rputils import actFunc, initFunc

class Layer():
    """
    Specification for a layer to be passed to the Neural Network during construction.  This
    includes a variety of parameters to configure each layer based on its activation type.
    """
    #  The attributes of the Layer class are initialized
    def __init__(self, ishape, neurons, oshape=0, weights_initializer=initFunc.random_normal, 
                 bias_initializer=initFunc.random_normal, gamma_initializer=initFunc.rbf_default, 
                 sigma_initializer=initFunc.ones, activation=actFunc.tanh, reg_strength=0.0, 
                 lambda_init=0.1, gamma_rate=0.0, sigma_rate=0.0, cvnn=1, xp=None):
        
        """ 
        The __init__ method is the constructor of the Layer class. 
        
        Parameters
        ----------
            ishape: int
                The number of neurons in the first layer (the number of input features).  
            neurons: int
                The number of neurons in the hidden layer. 
                
            oshape: int
                The oshape is a specific argument for the RBF networks; in shallow CVNNs, 
                as there is only one layer, the input and output dimensions and the number 
                of hidden neurons must be specified when adding the layer.
                
            weights_initializer: str
                It defines the way to set the initial random weights, as a string. 
                
            bias_initializer: str 
                It defines the way to set the initial random biases, as string.
                
            gamma_initializer: str, optional
                It defines the way to set the initial random gamma, as string.
                
            sigma_initializer: str, optional
                It defines the way to set the initial sigma biases, as string. Initialization
                methods were defined in the file rp_utils.initFunc.
                
                * rp_utils.initFunc.zeros
                * rp_utils.initFunc.ones
                * rp_utils.initFunc.ones_real
                * rp_utils.initFunc.random_normal
                * rp_utils.initFunc.random_uniform
                * rp_utils.initFunc.glorot_normal
                * rp_utils.initFunc.glorot_uniform
                * rp_utils.initFunc.rbf_default
                
            activation: str
                Select which activation function this layer should use, as a string.
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
                
            reg_strength: float, optional
                It sets the regularization strength. The default value is 0.0, which means
                that regularization is turned off.
                
            lambda_init: float, optional
                It is the initial regularization factor strength.
                
            gamma_rate: float, optional
                The learning rate of matrix of the center vectors (RBF networks).
                
            sigma_rate: float, optional
                The learning rate of the vector of variance (RBF networks).
            
            cvnn: int
                It Defines which complex neural network the layer belongs to.
                
                * 1: CVFFNN or SCFFNN
                * 2: CVRBFNN
                * 3: FCRBFNN
                * 4: Deep PTRBFNN
            xp: str
                CuPy/Numpy module. This parameter is set at the time of 
                initialization of the NeuralNetwork class.
        Returns
        -------
            None.
        """
        self.xp = xp
        self.input = None
     
        self.input = None
        self.reg_strength = reg_strength
        self.lambda_init = lambda_init
     
        self._activ_in, self._activ_out = None, None
        
        self.gamma_rate = gamma_rate
        self.sigma_rate = sigma_rate
        self.neurons = neurons
        self.oshape = oshape
        self.seuc = None
        self.phi = None
        self.kern = None
    
        ## It initializes parameters for feedforward (FF) networks (CVFFNN and SCFFNN). 
        ## This includes initializing weights, biases, activation
        if cvnn==1:
            self.weights = weights_initializer(xp, ishape, neurons)
            self.biases = bias_initializer(xp, 1, neurons)
            self.activation = activation
            self._dweights = self._prev_dweights = initFunc.zeros(xp, ishape, neurons)
            self._dbiases = self._prev_dbiases = initFunc.zeros(xp, 1, neurons)
            
        ## It initializes parameters for CVRBFNN. 
        ## This includes initializing weights, biases, gamma and sigma 
        elif cvnn==2:
            self.weights = weights_initializer(xp, neurons, oshape)
            
            self.biases = bias_initializer(xp, oshape, 1)
            
            self._dweights = self._prev_dweights = initFunc.zeros(xp, neurons, oshape)
            self._dbiases = self._prev_dbiases = initFunc.zeros(xp,oshape, 1)
            
            self.gamma = gamma_initializer(xp, neurons, ishape) 
            self.sigma = sigma_initializer(xp, 1, neurons)
            
            self._prev_dgamma = self._dgamma = initFunc.zeros(xp, neurons, ishape)
            self._prev_dsigma = self._dsigma = initFunc.zeros(xp, 1, neurons)
            
        ## It initializes parameters for FCRBFNN. 
        ## This includes initializing weights, biases, gamma and sigma 
        elif cvnn==3:
            self.weights = weights_initializer(xp, neurons, oshape)
            self.biases = bias_initializer(xp, oshape, 1)
        
            self._dweights = self._prev_dweights = initFunc.zeros(xp, neurons, oshape)
            self._dbiases = self._prev_dbiases = initFunc.zeros(xp, oshape, 1)
            
            self.gamma = initFunc.rbf_default(xp, neurons, ishape) #gpu.get_module().random.randint(2, size=[neurons,ishape])*0.7 + 1j*(gpu.get_module().random.randint(2, size=[neurons,ishape])*2-1)*0.7
            self.sigma = initFunc.rbf_default(xp, neurons, ishape) #gpu.get_module().random.randint(2, size=[neurons,ishape])*0.7 + 1j*(gpu.get_module().random.randint(2, size=[neurons,ishape])*2-1)*0.7
        
            self._prev_dgamma = self._dgamma = initFunc.zeros(xp, neurons, ishape)
            self._prev_dsigma = self._dsigma = initFunc.zeros(xp, neurons, ishape)
            
        ## It initializes parameters for DeepPTRBFNN. 
        ## This includes initializing weights, biases, gamma and sigma     
        elif cvnn==4:
            self.weights = weights_initializer(xp, neurons, oshape)
            self.biases = bias_initializer(xp, 1, oshape)
            self.gamma =  initFunc.rbf_default(xp, neurons, ishape) #gpu.get_module().random.randint(2, size=[neurons,ishape])*0.7 + 1j*(gpu.get_module().random.randint(2, size=[neurons,ishape])*2-1)*0.7
            self.sigma = initFunc.ones(xp, 1, neurons)
           
            self._ddweights = self._dweights = self._prev_dweights = initFunc.zeros(xp, neurons, oshape)
            self._ddbiases = self._dbiases = self._prev_dbiases = initFunc.zeros(xp, 1, oshape)
            
            self._prev_dgamma = self._dgamma = initFunc.zeros(xp, neurons, ishape)
            self._prev_dsigma = self._dsigma = initFunc.zeros(xp, 1, neurons)
            
    
       
       