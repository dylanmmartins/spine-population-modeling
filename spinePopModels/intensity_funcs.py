import numpy as np

from .helpers import find_closest_timestamp

def inten(t, pops, transition_dict):
    # Intensity function and demographic functions combined into single func
    # Should return the values for birth*population, etc. as numbers not as rates

    # Current populations
    F, H, S, M = pops

    # Find position in estrous cycle and use the appropriate transition
    # values for that point between stages
    tind, _ = find_closest_timestamp(np.arange(0,96.25,0.25), t)

    # Given a cycle position, get the transition probabilities
    use_intensities = {}
    for key,val in transition_dict.items():
        use_intensities[key] = val[tind]

    # Ignore lambda values because these represent staibility not
    # a transition between spine classes
    deltaM = use_intensities['deltaM']*M
    gammaM2F = use_intensities['gammaM2F']*M
    gammaM2H = use_intensities['gammaM2H']*M
    gammaM2S = use_intensities['gammaM2S']*M
    # lambdaM = use_intensities['lambdaM']
    deltaS = use_intensities['deltaS']*S
    gammaS2F = use_intensities['gammaS2F']*S
    gammaS2H = use_intensities['gammaS2H']*S
    # lambdaS = use_intensities['lambdaS']
    gammaS2M = use_intensities['gammaS2M']*S
    deltaH = use_intensities['deltaH']*H
    gammaH2F = use_intensities['gammaH2F']*H
    # lambdaH = use_intensities['lambdaH']
    gammaH2S = use_intensities['gammaH2S']*H
    gammaH2M = use_intensities['gammaH2M']*H
    deltaF = use_intensities['deltaF']*F
    # lambdaF = use_intensities['lambdaF']
    gammaF2H = use_intensities['gammaF2H']*F
    gammaF2S = use_intensities['gammaF2S']*F
    gammaF2M = use_intensities['gammaF2M']*F
    betaF = use_intensities['betaF']
    betaH = use_intensities['betaH']
    betaS = use_intensities['betaS']
    betaM = use_intensities['betaM']

    # the population intensites, lambda(t), of the point process are the expected rate of occurrence
    # of events at a particular time t
    ppintens = np.array([
        betaF,
        betaH,
        betaS,
        betaM,
        deltaF,
        deltaH,
        deltaS,
        deltaM,
        gammaH2F,
        gammaS2F,
        gammaM2F,
        gammaF2H,
        gammaF2S,
        gammaF2M,
        gammaS2H,
        gammaM2H,
        gammaH2S,
        gammaM2S,
        gammaH2M,
        gammaS2M
    ])

    return ppintens



def calc_intensity(t, pops, params):
    # Intensity function and demographic functions combined into single func
    # Should return the values for birth*population, etc. as numbers not as rates
    # c == Current hormone concentrations (not including bias term

    alltimes = np.arange(0,4.25,0.25) # units of days
    ind, _ = find_closest_timestamp(alltimes, t)
    c = X_conc[ind,:]

    # augment concentrations by inserting a 1.0 to use full bias term
    c_aug = np.insert(c, 0, 1.0)

    # Current populations
    F, H, S, M = pops

    # Ignore lambda values because these represent staibility not a real transition
    # The population intensites, lambda(t), of the point process are the expected rate of occurrence
    # of events at a particular time t
    ppintens = np.array([
        params['w_betaF'] @ c_aug,              # betaF
        params['w_betaH'] @ c_aug,              # betaH
        params['w_betaS'] @ c_aug,              # betaS
        params['w_betaM'] @ c_aug,              # betaM
        (params['w_deltaF'] @ c_aug) * F,       # deltaF
        (params['w_deltaH'] @ c_aug) * H,       # deltaF
        (params['w_deltaS'] @ c_aug) * S,       # deltaF
        (params['w_deltaM'] @ c_aug) * M,       # deltaF
        (params['w_gammaH2F'] @ c_aug) * H,     # gammaHF
        (params['w_gammaS2F'] @ c_aug) * S,     # gammaSF
        (params['w_gammaM2F'] @ c_aug) * M,     # gammaMF
        (params['w_gammaF2H'] @ c_aug) * F,     # gammaFH
        (params['w_gammaF2S'] @ c_aug) * F,     # gammaFS
        (params['w_gammaF2M'] @ c_aug) * F,     # gammaFM
        (params['w_gammaS2H'] @ c_aug) * S,     # gammaSH
        (params['w_gammaM2H'] @ c_aug) * M,     # gammaMH
        (params['w_gammaH2S'] @ c_aug) * H,     # gammaHS
        (params['w_gammaM2S'] @ c_aug) * M,     # gammaMS
        (params['w_gammaH2M'] @ c_aug) * H,     # gammaHM
        (params['w_gammaS2M'] @ c_aug) * S,     # gammaSM
    ])
    return ppintens