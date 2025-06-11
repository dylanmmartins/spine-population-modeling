import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import yaml
import os

import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['svg.fonttype'] = 'none'


from .spinePopModels.estradiol import get_factor_concentrations
from .spinePopModels.gillespie import gillespie_plus
from .spinePopModels.poisson_processes import nhpp
from .spinePopModels.linalg import plot_phase_portrait
from .spinePopModels.estradiol import plot_matrix_cell_evolution, plot_transition_vs_E2
from .spinePopModels.intensity_funcs import calc_intensity



def calc_mean_trace(res_list, ind):
    mean_trace = np.zeros_like(res_list[0][:,1])
    for i in range(len(res_list)):
        mean_trace += res_list[i][:,ind] / len(res_list)
    return mean_trace


def normalize_rows(x):
    norm = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
    return x / norm


def fit_model():
    symbol_matrix = np.array([
        ['deltaM','gammaM2F','gammaM2H','gammaM2S','lambdaM'],
        ['deltaS','gammaS2F','gammaS2H','lambdaS','gammaS2M'],
        ['deltaH','gammaH2F','lambdaH','gammaH2S','gammaH2M'],
        ['deltaF','lambdaF','gammaF2H','gammaF2S','gammaF2M'],
        ['0','betaF','betaH','betaS','betaM']
    ])

    # Load transition probability matrices
    DtoP = np.load(os.path.join(os.path.realpath(os.getcwd()), 'spinePopModels/transition_mats/DtoP_transition_matrix.npy'))
    PtoE = np.load(os.path.join(os.path.realpath(os.getcwd()), 'spinePopModels/transition_mats/PtoE_transition_matrix.npy'))
    EtoM = np.load(os.path.join(os.path.realpath(os.getcwd()), 'spinePopModels/transition_mats/EtoM_transition_matrix.npy'))
    MtoD = np.load(os.path.join(os.path.realpath(os.getcwd()), 'spinePopModels/transition_mats/MtoD_transition_matrix.npy'))

    mats = [DtoP, PtoE, EtoM, MtoD]

    X_conc = get_factor_concentrations()


    fig, [ax1,ax2,ax3,ax4] = plt.subplots(4,1, figsize=(2.75,4), dpi=300)
    ax1.plot(np.arange(0,4.25,0.25), X_conc[:,0], color='k')
    ax2.plot(np.arange(0,4.25,0.25), X_conc[:,1], color='k')
    ax3.plot(np.arange(0,4.25,0.25), X_conc[:,2], color='k')
    ax4.plot(np.arange(0,4.25,0.25), X_conc[:,3], color='k')
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xlim([0,4])
        ax.vlines([1,2,3], 0, 500, color='k', alpha=0.3, ls='--', lw=1)
    ax1.set_ylim([0,0.045])
    ax2.set_ylim([0,65])
    ax3.set_ylim([0,45])
    ax4.set_ylim([0,400])
    ax1.set_ylabel('estradiol (ng/mL)')
    ax2.set_ylabel('progesterone (ng/mL)')
    ax3.set_ylabel('LH (ng/mL)')
    ax4.set_ylabel('FSH (ng/mL)')
    plt.tight_layout()
    plt.savefig('hormone_interp_concentrations_dt0p25.svg')


    weight_array = np.zeros([5,25])
    for i, k in enumerate(w.keys()):
        weight_array[:,i] = w[k]

    plt.imshow(normalize_rows(weight_array).T)
    plt.yticks(np.arange(25), labels=symbol_matrix.flatten())
    plt.xticks(np.arange(5), labels=['bias','estradiol','prog.','LH','FSH'], rotation=90)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('glm_weights_colnorm.svg')


    params = w.copy()

    # Initial population sizes
    init = [1,6,23,2]

    # Time grid: from 0 to 20 in steps of 1
    # 6 passes through full estrous cycle
    # times = np.arange(0, 96*6+.25, 0.25)
    times = np.arange(0, 20.5, 0.25) # now, step 12 hrs at a time. was doing 30 min, but compute time for a single simulation was ~200 min. not worth it.
    # Run simulation using gillespie_plus (nonhomogeneous Poisson process sampler)
    res_list = {}
    for i in tqdm(range(100)):
        res_gillespie_plus = gillespie_plus(init, times, calc_intensity, nhpp)
        res_list[i] = res_gillespie_plus

    savepath = 'res_gillespie_plus_100r_dt_0p25_init_1_6_23_2.yaml'
    with open(savepath, 'w') as outfile:
        yaml.dump(res_list, outfile, default_flow_style=False)


    sum_trace = np.zeros_like(res_list[0][:,1])
    fig, [ax1,ax2,ax3,ax4,ax5] = plt.subplots(5, 1, dpi=300, figsize=(5,7))
    for i in range(len(res_list)):
        ax1.plot(res_list[i][:,0], res_list[i][:,1], alpha=0.1)
        ax2.plot(res_list[i][:,0], res_list[i][:,2], alpha=0.1)
        ax3.plot(res_list[i][:,0], res_list[i][:,3], alpha=0.1)
        ax4.plot(res_list[i][:,0], res_list[i][:,4], alpha=0.1)
        ax5.plot(
                res_list[0][:,0],
                (res_list[i][:,1]+res_list[i][:,2]+res_list[i][:,3]+res_list[i][:,4]),
                alpha=0.1
        )
        sum_trace += (res_list[i][:,1]+res_list[i][:,2]+res_list[i][:,3]+res_list[i][:,4]) / len(res_list)
    ax1.plot(res_list[0][:,0], calc_mean_trace(res_list,1), color='tab:blue')
    ax2.plot(res_list[0][:,0], calc_mean_trace(res_list,2), color='tab:red')
    ax3.plot(res_list[0][:,0], calc_mean_trace(res_list,3), color='tab:orange')
    ax4.plot(res_list[0][:,0], calc_mean_trace(res_list,4), color='tab:green')

    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.vlines(np.arange(2.1, 20, 4), 0, 60, ls='--', lw=1, alpha=0.3, color='gray')
        ax.set_xlim([0,20])
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xticks(np.arange(0,24,4), labels=[])
    ax1.set_ylim([0,13])
    ax2.set_ylim([0,25])
    ax3.set_ylim([0,33])
    ax4.set_ylim([0,25])
    ax5.set_ylim([0,55])
    ax5.plot(res_list[0][:,0], sum_trace, color='k')
    ax1.set_ylabel('F')
    ax2.set_ylabel('H')
    ax3.set_ylabel('S')
    ax4.set_ylabel('M')
    ax5.set_ylabel('F+H+S+M')
    ax5.set_xticks(np.arange(0,24,4))
    ax5.set_xlabel('time (days)')
    fig.tight_layout()
    fig.savefig('sim_results_v2.svg')


    fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, figsize=(6,4), dpi=300)

    plot_phase_portrait(DtoP, [1,6,23,2], ['F','H'], ax=ax1)
    plot_phase_portrait(DtoP, [1,6,23,2], ['F','S'], ax=ax2)
    plot_phase_portrait(DtoP, [1,6,23,2], ['F','M'], ax=ax3)
    plot_phase_portrait(DtoP, [1,6,23,2], ['H','S'], ax=ax4)
    plot_phase_portrait(DtoP, [1,6,23,2], ['H','M'], ax=ax5)
    plot_phase_portrait(DtoP, [1,6,23,2], ['S','M'], ax=ax6)

    fig.suptitle('D to P')
    fig.tight_layout()
    fig.savefig('DtoP_vecfields.svg')

    fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, figsize=(6,4), dpi=300)

    plot_phase_portrait(PtoE, [1,6,23,2], ['F','H'], ax=ax1)
    plot_phase_portrait(PtoE, [1,6,23,2], ['F','S'], ax=ax2)
    plot_phase_portrait(PtoE, [1,6,23,2], ['F','M'], ax=ax3)
    plot_phase_portrait(PtoE, [1,6,23,2], ['H','S'], ax=ax4)
    plot_phase_portrait(PtoE, [1,6,23,2], ['H','M'], ax=ax5)
    plot_phase_portrait(PtoE, [1,6,23,2], ['S','M'], ax=ax6)

    fig.suptitle('P to E')
    fig.tight_layout()
    fig.savefig('PtoE_vecfields.svg')

    fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, figsize=(6,4), dpi=300)

    plot_phase_portrait(EtoM, [1,6,23,2], ['F','H'], ax=ax1)
    plot_phase_portrait(EtoM, [1,6,23,2], ['F','S'], ax=ax2)
    plot_phase_portrait(EtoM, [1,6,23,2], ['F','M'], ax=ax3)
    plot_phase_portrait(EtoM, [1,6,23,2], ['H','S'], ax=ax4)
    plot_phase_portrait(EtoM, [1,6,23,2], ['H','M'], ax=ax5)
    plot_phase_portrait(EtoM, [1,6,23,2], ['S','M'], ax=ax6)

    fig.suptitle('E to M')
    fig.tight_layout()
    fig.savefig('EtoM_vecfields.svg')

    fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, figsize=(6,4), dpi=300)

    plot_phase_portrait(MtoD, [1,6,23,2], ['F','H'], ax=ax1)
    plot_phase_portrait(MtoD, [1,6,23,2], ['F','S'], ax=ax2)
    plot_phase_portrait(MtoD, [1,6,23,2], ['F','M'], ax=ax3)
    plot_phase_portrait(MtoD, [1,6,23,2], ['H','S'], ax=ax4)
    plot_phase_portrait(MtoD, [1,6,23,2], ['H','M'], ax=ax5)
    plot_phase_portrait(MtoD, [1,6,23,2], ['S','M'], ax=ax6)

    fig.suptitle('M to D')
    fig.tight_layout()
    fig.savefig('MtoD_vecfields.svg')


    e_times = np.array([0, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4])
    e_vals = np.array([10, 20, 25, 40, 10, 5, 10, 5, 10])/1000 # convert to ng/mL (from pg/mL)
    p_times = np.array([0,0.5,1.25,1.5,1.6,1.75,0.25,2.0,4.0,3.0,3.5,2.5])
    p_vals = np.array([30,10,4,2,3,60,30,5,30,15,30,5])
    l_times = np.array([0,1,1.1,0.9,1.5,1.75,2.25,3.0,4.0,2,1.85,1.95])
    l_vals = np.array([2,2,2,2,2,40,2,2,2,5,15,10])
    f_times = np.array([0,.5,1,1.5,1.75,2.15,2,2.25,3.25,3.75,4,3.5,2.5,2.4,2.3,2.6,2.7])
    f_vals = np.array([100,20,20,20,375,375,375,20,20,20,100,100,20,20,20,20,20])

    plot_transition_vs_E2([DtoP,PtoE,EtoM,MtoD], e_times, e_vals, 'estradiol ng/mL')
    plot_transition_vs_E2([DtoP,PtoE,EtoM,MtoD], p_times, p_vals, 'progesterone ng/mL')
    plot_transition_vs_E2([DtoP,PtoE,EtoM,MtoD], l_times, l_vals, 'LH ng/mL')
    plot_transition_vs_E2([DtoP,PtoE,EtoM,MtoD], f_times, f_vals, 'FSH ng/mL')


    plot_matrix_cell_evolution([DtoP,PtoE,EtoM,MtoD])



if __name__ == '__main__':
    fit_model()