# -*- coding: utf-8 -*-
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

"""
A sample code for using the RosenPy platform with the Deep PTRBF Neural Network 
for the complex domain problem Cha and Kassan.
"""
# Import the Deep PTRBF Neural Network module from RosenPy
import rosenpymodel.deepptrbfnn as mynn
import rputils.utils as utils
import numpy as np

def setData():
    trainSetIn = []
    trainSetOut = (np.random.randint(0,2, (1,1204))*2-1)*0.7 + 1j*(np.random.randint(0,2, (1,1204))*2-1)*0.7
    a1 = -0.7 - 1j*0.7;
    a2 = -0.7 + 1j*0.7;
    
    i=0
    while i<1204 :
        a0 = trainSetOut[0][i]
        
        aux = (0.34-1j*0.27)*a0 + (0.87+1j*0.43)*a1 + (0.34-1j*0.21)*a2
        trainSetIn.append(aux + 0.1*aux**2+ 0.05*aux**3+np.sqrt(0.01)*(np.random.randn()/np.sqrt(2)+(1j*np.random.randn())/np.sqrt(2)))
        
        a2=a1
        a1=a0
        i+=1
    
    trainSetOut = trainSetOut.T[:1204-2]
        
    trainSetIn = np.array([trainSetIn[:1002-2], trainSetIn[1:1002-1], trainSetIn[2:1002]]).T
    trainSetOut = trainSetOut[:1002-2]
    return trainSetIn, trainSetOut

def main():
        # Prepare the input and output data
        input_data, output_data = setData()
                
        # Create an instance of the Deep PTRBF Neural Network     
        nn = mynn.DeepPTRBFNN(learning_rate=1e-2)
        
        # Add layers to the network
        nn.addLayer(ishape=input_data.shape[1], neurons=4, oshape=5,  gamma_rate=1e-1, sigma_rate=1e-1)
        nn.addLayer(neurons=6, oshape=output_data.shape[1], gamma_rate=1e-1, sigma_rate=1e-1)
    
        # Train the network
        nn.fit(input_data, output_data, epochs=1000, verbose=100, batch_size=100)

        # Make predictions
        y_pred = nn.predict(input_data)
        
        # Calculate and print accuracy
        print('Accuracy: {:.2f}%'.format(utils.accuracy(output_data, y_pred)))



main()


