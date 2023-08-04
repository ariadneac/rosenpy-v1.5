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

def mse(xp, y, y_pred):
    """
    Calculates the mean squared error (MSE) between the target values (y) and predicted values (y_pred).

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    y : array-like
        The target values.
    y_pred : array-like
        The predicted values..

    Returns
    -------
    float
        The mean squared error between y and y_pred.

    """
    aux = xp.square(xp.abs(y-y_pred))
    return (0.5*(xp.dot(xp.ones((1, aux.shape[0])), aux)/aux.shape[0]))[0][0]
