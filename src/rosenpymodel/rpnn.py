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
from rputils import costFunc, decayFunc, batchGenFunc, gpu

        
class NeuralNetwork():
    """
    Abstract base class for wrapping all neural network functionality from RosenPy. 
    This is a superclass. 
    """
   
    def __init__(self, cost_func=costFunc.mse, learning_rate=1e-3, lr_decay_method=decayFunc.none_decay,  
                 lr_decay_rate=0.0, lr_decay_steps=1, momentum=0.0, patience=10000000000000, xp=None):     
        """ 
        The __init__ method is the constructor of the NeuralNetwork class. 
        It initializes the model with default values for various parameters, like
        cost function, learning rate, decay method, momentum e patience factor.
        
        Parameters
        ----------
            cost_func: str, optional
                The cost function to use when training the network. The default cost function, 
                MSE (Mean Square Error), defined in the file rp_utils.costFunc.
            
            learning_rate: float, optional
                Real number indicating the default/starting rate of adjustment for the weights 
                during gradient descent. It controls how quickly or slowly a neural network model 
                learns a problem. Default is ``0.001``.
            
            
            lr_decay_method: str, optional
                Learning rate schedules seek to adjust the learning rate during training by reducing 
                the learning rate according to a pre-defined schedule. This parameter defines the 
                decay method function. Three methods were implemented in this work, time-based, 
                exponential, and staircase, defined in the file rp_utils.decayFunc.
                    * rp_utils.decayFunc.none_decay: No decay method is defined. This is the 
                    default value.    
                
                    * rp_utils.decayFunc.time_based_decay: Time-based learning schedules alter 
                    the learning rate depending on the learning rate of the previous time iteration. 
                    Factoring in the decay the mathematical formula for the learning rate is: 
                    1.0/(1.0 + decay_rate*epoch)
                        
                    * rp_utils.decayFunc.exponential_decay: Exponential learning schedules are similar 
                    to step-based, but instead of steps, a decreasing exponential function is used. 
                    The mathematical formula for factoring in the decay is: 
                    learning_rate * decay_rate ** epoch
                    
                    * rp_utils.decayFunc.staircase: The decay the learning rate at discrete intervals.  
                    The mathematical formula for factoring in the decay is: 
                    learning_rate * decay_rate ** (epoch // decay_steps)
            
            lr_decay_rate: float, optional 
                It is used to initialize the learning rate at a high value that gradually decreases over 
                time, allowing the network to converge.
                
            lr_decay_steps: float, optional
                This parameter is only used in the staircase method.
            
            momentum: float, optional
                Real number indicating the momentum factor to be used for the
                learning rule 'momentum'. Default is ``0.0``
           
            patience: int, optional
                It specifies a patience factor, the CVNN will wait for a specified number of epochs 
                until the loss in the validation dataset decreases sufficiently for the training to 
                be terminated.
                
            xp: str, optional.
                The xp attribute stores the module for performing computations (e.g., NumPy or GPU module). 
                It can be set directly by the user or automatically by the system.
        Returns
        -------
        None.
        """
        
        self.xp = xp if xp is not None else gpu.get_module()
        self.layers = []
        self.cost_func = cost_func
        self.momentum = momentum
        self.learning_rate = self.lr_initial = learning_rate
       
        self.lr_decay_method = lr_decay_method
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        
        self.patience, self.waiting = patience, 0
        
        self._best_model, self._best_loss = self.layers, self.xp.inf
        self._history = {'epochs': [], 'loss': [], 'loss_val': []}
        
    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=100, verbose=10, batch_gen=batchGenFunc.batch_sequential, batch_size=1):
        """
        The fit method is used to train the model with the given training data.
        
        Parameters
        ----------
            x_train: array-like, shape (n_samples, n_inputs)
                Training vectors as real numbers, where n_samples is the number of
                samples and n_inputs is the number of input features.
                
            y_train: array-like, shape (n_samples, n_outputs)
                Target values are real numbers representing the desired outputs.
                
            x_val: array-like, shape (n_samples, n_inputs), optional
                Validation vectors as real numbers, where n_samples is the number of
                samples and n_inputs is the number of input features.
                
            y_val: array-like, shape (n_samples, n_outputs), optional
                It is representing the validation target values.
                If validation data is not provided, the method uses the training data 
                for validation.
                
            epochs: int
                Number of epochs to train the model. An epoch means training the 
                neural network with all the training data for one cycle. 
                
            verbose: int, optional 
                If the epoch number is divisible by the verbose value, 
                it calculates the loss value for the training and validation data and updates 
                the training history.
            
            batch_gen: str, optional
                It defines the batch generation function which can be sequential or 
                shuffled (defined in the file rp_utils.batchGenFunc).
                
                * rp_utils.batchGenFunc.batch_sequential
                * rp_utils.batchGenFunc.batch_shuffle
                
            batch_size: int, optional  
                It defines the number of samples to work through before updating the internal 
                model parameters.
       
        Returns
        -------
            dict 
            Return the training and validation history. For example:

                {'epochs': ('0', '100', '200'),
                 'loss': (0.41058633, 0.00749860, 0.00473513),
                 'loss_val': (0.41058633, 0.00749860, 0.00473513)}
        
        """
        # If validation data is not provided, the method uses the training data for validation
        x_val, y_val = (x_train, y_train) if (x_val is None or y_val is None) else (x_val, y_val)
        
        for epoch in range(epochs+1):
            
            # The method iterates over each epoch and updates the learning rate based on the decay method.          
            self.learning_rate = self.lr_decay_method(self.lr_initial, epochs, self.lr_decay_rate, self.lr_decay_steps)
       
            # It generates batches of training data using the specified batch generation function
            x_batch, y_batch = batch_gen(x_train, y_train, batch_size)
        
        
            # For each batch, it performs feedforward and backpropagation to update the model's parameters
            for x_batch1, y_batch1 in zip(x_batch, y_batch):
                    y_pred = self.feedforward(x_batch1) 
                
                    self.backprop(y_batch1, y_pred, epoch)   
                    
            # After each epoch, it calculates the loss value for the validation data 
            loss_val = self.cost_func(y_val, self.predict(x_val))
            
            # If the patience value is set, it checks if the loss value has improved
            if self.patience != 10000000000000:
                # If the loss has improved, it updates the best model and resets the waiting counter
                if loss_val < self._best_loss:
                    self._best_model, self._best_loss = self.layers, loss_val
                    self.waiting = 0
                # If the loss hasn't improved, it increments the waiting counter and checks if the patience limit has been reacher
                else: 
                    self.waiting +=1
                    print("not improving: [{}] current loss val: {} best: {}".format(self.waiting, loss_val, self._best_loss))
                    
                    # If the patience limit is reached, it reverts to the best model and stops training
                    if self.waiting >= self.patience:
                        self.layers = self._best_model
                        print("early stopping at epoch ", epoch)
                        return
            # If the epoch number is divisible by the verbose value, it calculates the loss value for 
            # the training data and updates the training history
            if epoch % verbose == 0:
                loss_train = self.cost_func(y_train, self.predict(x_train))
                self._history['epochs'].append(epoch)
                self._history['loss'].append(loss_train)
                self._history['loss_val'].append(loss_val)
                print("epoch: {0:=4}/{1} loss_train: {2:.8f} loss_val: {3:.8f} ".format(epoch, epochs, loss_train, loss_val))
        # It returns the training history        
        return self._history    
                
    def predict(self, x):      
        """
        Calculate predictions for specified inputs.
        
        Parameters
        ----------
            x: array-like, shape (n_samples, n_inputs)
            The input samples as real numbers.
        
        Returns
        -------
            y : array-like, shape (n_samples, n_outputs)
            The predicted values as real numbers.
        """
        return self.feedforward(x)
    
    def addLayer(self): 
        pass
    
    def getHistory(self):
        """
        The getHistory method returns the training and validation history to the training model.

        Returns
        -------
        dict 
        Return the training and validation history. For example:

            {'epochs': ('0', '100', '200'),
             'loss': (0.41058633, 0.00749860, 0.00473513),
             'loss_val': (0.41058633, 0.00749860, 0.00473513)}

        """
    
        return self._history
    
    
    