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
This file contains various initialization functions for initializing complex matrices.

"""

#from rputils import gpu

#xp = gpu.get_module()




def zeros(xp, rows, cols):
    """
    Initializes a complex matrix with all elements set to zero.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with all elements set to zero.

    """
    return xp.zeros((rows,cols),dtype=complex)

def ones(xp, rows, cols):
    """
    Initializes a complex matrix with all elements set to one.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with all elements set to one.

    """
    return xp.ones((rows, cols),dtype=complex)+1j

def ones_real(xp, rows, cols):
    """
    Initializes a real matrix with all elements set to one.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A real matrix of size (rows, cols) with all elements set to one.

    """
    return xp.ones((rows, cols))

def random_normal(xp, rows, cols):
    """
    Initializes a complex matrix with elements sampled from a normal distribution.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with elements sampled from a normal distribution.

    """
    real = xp.random.randn(rows, cols).astype(xp.float32) - 0.5
    imag = xp.random.randn(rows, cols).astype(xp.float32) - 0.5
    return (real + 1j * imag) / 10

def random_uniform(xp, rows, cols):
    """
    Initializes a complex matrix with elements sampled from a uniform distribution.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with elements sampled from a uniform distribution.

    """
    real = xp.random.rand(rows, cols).astype(xp.float32) 
    imag = xp.random.rand(rows, cols).astype(xp.float32) 
    return (real + 1j * imag) / 10

def glorot_normal(xp, rows, cols):
    """
    Initializes a complex matrix using the Glorot normal initialization method.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) initialized using the Glorot normal initialization method.

    """
    std_dev = xp.sqrt(2.0/(rows+cols))/10
    return (std_dev*xp.random.randn(rows, cols) + 1j*std_dev*xp.random.randn(rows, cols))/10

def glorot_uniform(xp, rows, cols):
    """
    Initializes a complex matrix using the Glorot uniform initialization method.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) initialized using the Glorot uniform initialization method.

    """
    std_dev = xp.sqrt(6.0/(rows+cols))/10
    return (2*std_dev*xp.random.randn(rows, cols)-std_dev + 1j*(std_dev*xp.random.randn(rows, cols)-std_dev))/5

def rbf_default(xp, rows, cols):
    """
    Initializes a complex matrix with elements generated from a random binary distribution.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    rows : int
        The number of rows in the matrix.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    array-like
        A complex matrix of size (rows, cols) with elements generated from a random binary distribution.

    """
    return xp.random.randint(2, size=[rows, cols])*0.7 + 1j*(xp.random.randint(2, size=[rows, cols])*2-1)*0.7
    
    
    
   


