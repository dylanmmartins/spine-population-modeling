import numpy as np
import math
import random
import warnings
from scipy.integrate import quad
from scipy.optimize import root_scalar

# Homogeneous Poisson Process (hpp) inter-event time sampler
def hpp(intentemp):
    total_intensity = sum(intentemp)
    # Avoid division by zero (should be handled by caller)
    return - (1 / total_intensity) * math.log(1 - random.random())

# Nonhomogeneous Poisson Process (nhpp) inter-event time sampler using inverse transform
def nhpp(tottime, N, param, inten, timeleft):
    Y = random.random()
    def f(X):
        # Integrate sum of intensities from 0 to X
        integral, _ = quad(lambda x: sum(inten(tottime + x, N, param)), 0, X, limit=200)
        return 1 - math.exp(-integral) - Y
    try:
        sol = root_scalar(f, bracket=[0, timeleft], method='bisect', xtol=1e-5)
        return sol.root
    except ValueError:
        # In case no root is found, return a value greater than timeleft
        return timeleft + 1

# Gillespie simulation (SSA with a fixed time grid) using homogeneous sampling
def gillespie(init, times, param, inten, pproc, hpp_func):
    if len(times) == 0:
        raise ValueError("No time points provided in 'times'")
    if times[0] != 0:
        raise ValueError("First time point is not 0")

    tottime = times[0]
    tinc = len(times)
    N = np.array(init, dtype=float)  # current state (assumed 1D)
    # Prepare a results array; each row will store the state at the corresponding grid time.
    results = np.zeros((tinc, len(N)))
    results[0, :] = N.copy()

    i = 1
    while i < tinc:
        results[i, :] = results[i - 1, :].copy()
        # Process events until the simulation time exceeds the grid time points[i]
        while tottime <= times[i]:
            intentemp = inten(tottime, N, param)
            if all(x == 0 for x in intentemp):
                for j in range(i, tinc):
                    results[j, :] = N.copy()
                warnings.warn("Exiting with all intensities equal to 0")
                i = tinc  # exit outer loop
                break
            elif min(intentemp) < 0:
                for j in range(i, tinc):
                    results[j, :] = np.nan
                warnings.warn("Exiting with intensity less than 0")
                i = tinc  # exit outer loop
                break
            else:
                tau = hpp_func(intentemp)
                tottime += tau
                # Choose an event type based on the probabilities proportional to intensities.
                probabilities = np.array(intentemp) / sum(intentemp)
                event_index = np.random.choice(np.arange(pproc.shape[0]), p=probabilities)
                # If the new time is beyond the grid time, record current state and apply the event once.
                if tottime > times[i]:
                    results[i, :] = N.copy()
                    N = N + pproc[event_index, :]
                    break
                else:
                    N = N + pproc[event_index, :]
        i += 1

    # Combine the time vector and results into one array (first column is time)
    return np.column_stack((times, results))

# Gillespie+ simulation using nonhomogeneous Poisson process sampling
def gillespie_plus(init, times, param, inten, pproc, nhpp_func):
    if len(times) == 0:
        raise ValueError("No time points provided in 'times'")
    if times[0] != 0:
        raise ValueError("First time point is not 0")
    
    tottime = times[0]
    tinc = len(times)
    N = np.array(init, dtype=float)
    results = np.zeros((tinc, len(N)))
    results[0, :] = N.copy()

    i = 1
    while i < tinc:
        results[i, :] = results[i - 1, :].copy()
        while tottime <= times[i]:
            intentemp = inten(tottime, N, param)
            if all(x == 0 for x in intentemp):
                for j in range(i, tinc):
                    results[j, :] = N.copy()
                warnings.warn("Exiting with all intensities equal to 0")
                i = tinc
                break
            elif min(intentemp) < 0:
                for j in range(i, tinc):
                    results[j, :] = np.nan
                warnings.warn("Exiting with intensity less than 0")
                i = tinc
                break
            else:
                # Use the nonhomogeneous inter-event sampler with the remaining time interval
                tau = nhpp_func(tottime, N, param, inten, times[-1] - tottime)
                tottime += tau
                # Recalculate intensities for the new time.
                intentemp = inten(tottime, N, param)
                probabilities = np.array(intentemp) / sum(intentemp)
                event_index = np.random.choice(np.arange(pproc.shape[0]), p=probabilities)
                if tottime > times[i]:
                    results[i, :] = N.copy()
                    N = N + pproc[event_index, :]
                    break
                else:
                    N = N + pproc[event_index, :]
        i += 1

    return np.column_stack((times, results))
