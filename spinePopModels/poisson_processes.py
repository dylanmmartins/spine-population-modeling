import numpy as np
import math
from numpy import random
from scipy.optimize import root_scalar
from scipy.integrate import quad

# Homogeneous Poisson Process (hpp) inter-event time sampler
def hpp(intentemp):

    total_intensity = sum(intentemp)
    # Avoid division by zero (should be handled by caller)
    return - (1 / total_intensity) * math.log(1 - random.random())

# Nonhomogeneous Poisson Process (nhpp) inter-event time sampler using inverse transform
def nhpp1(tottime, pops, inten, timeleft, transition_dict):
    # OLD VERSION, NO LONGER USED

    Y = random.random()
    def f(X):
        # Integrate sum of intensities from 0 to X
        integral, _ = quad(lambda x: sum(inten(tottime + x, pops, transition_dict)), 0, X, limit=200)
        return 1 - math.exp(-integral) - Y
    try:
        sol = root_scalar(f, bracket=[0, timeleft], method='bisect', xtol=1e-5)
        return sol.root
    except ValueError:
        # In case no root is found, return a value greater than timeleft
        return timeleft + 1


def nhpp(tottime, pops, param, calc_intensity, timeleft, maxT=96):

    Y = random.random()
    def f(X):
        # Integrate sum of intensities from 0 to X
        integral, _ = quad(lambda x: sum(calc_intensity(tottime+x % maxT, pops, param)), 0, X, limit=200)
        return 1 - math.exp(-integral) - Y
    try:
        sol = root_scalar(f, bracket=[0, timeleft], method='bisect', xtol=1e-5)
        return sol.root
    except ValueError:
        # In case no root is found, return a value greater than timeleft
        return timeleft + 1
    # except OverflowError:
    #     return timeleft + 1