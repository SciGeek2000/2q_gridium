# Author: Thomas Ersevim, 2026
###################################################################################################

'''An ExpGridium test suite'''

import pytest
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/thomasersevim/QNL/2q_gridium')
from Circuit_Objs.qchard_expgridium import *

soft_ExpGridium = ExpGridium(**soft_ExpGridium_params,
                             **std_ExpGridium_operation_params,
                             **std_ExpGridium_sim_params)
hard_ExpGridium = ExpGridium(**hard_ExpGridium_params,
                         **std_ExpGridium_operation_params,
                         **std_ExpGridium_sim_params)

def test_spectrum():
    pass

def test_eigenvals():
    pass
