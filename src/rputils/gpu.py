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

import numpy as np

gpu_enable = False
try:
    import cupy as cp
    cupy = cp
    gpu_enable = True
except ImportError:
    gpu_enable = False
    

def get_module():
    """
    This function is used to implement CPU/GPU generic code.  If there is a GPU, 
    the cupy module is returned, otherwise the numpy module is returned.
    
    Returns
    -------
    module
        It returns module `cupy` or `numpy`.

    """
    if not gpu_enable:
        return np
    return cp

