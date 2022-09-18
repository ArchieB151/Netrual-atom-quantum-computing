import matplotlib.pyplot as plt

import numpy as np

from qutip import *

rho = (coherent(15, 1.5) + coherent(15, -1.5)).unit()

plot_wigner(rho, figsize=(6,6));
