import numpy as np
from inspect import signature
import scipy.interpolate

from .helpers import make_transition_dict

def make_transition_dict():

    symbol_matrix = np.array([
        ['deltaM','gammaM2F','gammaM2H','gammaM2S','lambdaM'],
        ['deltaS','gammaS2F','gammaS2H','lambdaS','gammaS2M'],
        ['deltaH','gammaH2F','lambdaH','gammaH2S','gammaH2M'],
        ['deltaF','lambdaF','gammaF2H','gammaF2S','gammaF2M'],
        ['0','betaF','betaH','betaS','betaM']
    ])

    # Load transition probability matrices
    DtoP = np.load('transition_mats/DtoP_transition_matrix.npy')
    PtoE = np.load('transition_mats/PtoE_transition_matrix.npy')
    EtoM = np.load('transition_mats/EtoM_transition_matrix.npy')
    MtoD = np.load('transition_mats/MtoD_transition_matrix.npy')

    mats = [DtoP, PtoE, EtoM, MtoD]

    # Make a transition matrix that ciruclarly interpolates over the four timepoints of
    # transition probabilities so that any timepoint along the cycle has a ~unique
    # transition matrix.
    transition_dict = {}
    for i in range(5): # from
        for j in range(5): # to

            # Get the values across stages
            transvals_across_stages = [mat[i, j] for mat in mats]

            # Interpolate to bins
            spline_interp = scipy.interpolate.CubicSpline(
                np.arange(5)*24,
                np.append(transvals_across_stages, transvals_across_stages[0]),
                bc_type="periodic"
            )
            transvals_at_bins = spline_interp(np.arange(0,96.25,0.25))

            # add to dict
            transition_dict[symbol_matrix[i,j]] = transvals_at_bins

    return transition_dict

def get_state_change_mat():

    state_change_matrix = np.array([
        [1,0,0,0],  # 1.  New F grown
        [0,1,0,0],  # 2.  New H grown
        [0,0,1,0],  # 3.  New S grown
        [0,0,0,1],  # 4.  New M grown
        [-1,0,0,0], # 5.  Existing F pruned
        [0,-1,0,0], # 6.  Existing H pruned
        [0,0,-1,0], # 7.  Existing S pruned
        [0,0,0,-1], # 8.  Existing M pruned
        [1,-1,0,0], # 9.  H transitions to F
        [1,0,-1,0], # 10. S transitions to F
        [1,0,0,-1], # 11. M transitions to F
        [-1,1,0,0], # 12. F transitions to H
        [-1,0,1,0], # 13. F transitions to S
        [-1,0,0,1], # 14. F transitions to M
        [0,1,-1,0], # 15. S transitions to H
        [0,1,0,-1], # 16. M transitions to H
        [0,1,0,-1], # 17. H transitions to S
        [0,0,1,-1], # 18. M transitions to S
        [0,-1,0,1], # 19. H trantisions to M
        [0,0,-1,1]  # 20. S transitions to M
    ])
    return state_change_matrix

# Gillespie+ simulation using nonhomogeneous Poisson process sampling
def gillespie_plus(init, times, inten, nhpp_func, fixed_cycle_tind=None):

    transition_dict = make_transition_dict()
    pproc = get_state_change_mat()

    if len(times) == 0:
        raise ValueError("No time points provided in 'times'")
    if times[0] != 0:
        raise ValueError("First time point is not 0")
    
    tottime = times[0]
    tinc = len(times)
    pops = np.array(init, dtype=float)
    results = np.zeros((tinc, len(pops)))
    results[0, :] = pops.copy()

    # Get number of arguments for the provided Poisson process function
    nargs = len(str(signature(nhpp_func)).split(','))

    i = 1
    while i < tinc:
        results[i, :] = results[i - 1, :].copy()
        while tottime <= times[i]:

            if nargs == 4:
                tau = nhpp_func(tottime, pops, inten, times[-1]-tottime)
            
            elif nargs == 1:
                
                if fixed_cycle_tind is not None:
                    intentemp = inten(fixed_cycle_tind, pops, transition_dict)
                else:
                    intentemp = inten(tottime, pops, transition_dict)
                
                tau = nhpp_func(intentemp)
                
            tottime += tau

            # Recalculate intensities for the new time.
            intentemp = inten(tottime, pops)

            # Handle negative intensities (before normalizing by the sum) by shifting the up so the lowest
            # intensity is zero. Once scaled by the sum, they'll still sum to 1.
            if np.nanmin(intentemp) < 0:
                intentemp += - np.nanmin(intentemp)

            probabilities = np.array(intentemp) / np.nansum(intentemp)

            _choice = np.arange(pproc.shape[0])
            _probs = probabilities.flatten()
            
            event_index = np.random.choice(_choice, p=_probs)
            if tottime > times[i]:
                results[i, :] = pops.copy()
                pops = pops + pproc[event_index, :]
                break
            else:
                pops = pops + pproc[event_index, :]

            pops[pops<0] = 0
        i += 1

    return np.column_stack((times, results))