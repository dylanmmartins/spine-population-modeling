import numpy as np
import scipy.interpolate
import os



from .helpers import find_closest_timestamp

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_decreasing(L) or non_increasing(L)

def spline_interp_conc(t, x):

    if np.size(t) != np.size(x):
        print('Sizes not equal')
        return
    if not monotonic(t):
        sortind = np.argsort(t)
        t = t[sortind]
        x = x[sortind]

    spline_interp = scipy.interpolate.CubicSpline(
        t,
        x,
        bc_type='periodic'
    )
    conc = spline_interp(np.arange(0,4.25,0.25))

    return conc

def get_factor_concentrations():
    e_times = np.array([0, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4])
    e_vals = np.array([10, 20, 25, 40, 10, 5, 10, 5, 10])/1000 # convert to ng/mL (from pg/mL)
    p_times = np.array([0,0.5,1.25,1.5,1.6,1.75,0.25,2.0,4.0,3.0,3.5,2.5])
    p_vals = np.array([30,10,4,2,3,60,30,5,30,15,30,5])
    l_times = np.array([0,1,1.1,0.9,1.5,1.75,2.25,3.0,4.0,2,1.85,1.95])
    l_vals = np.array([2,2,2,2,2,40,2,2,2,5,15,10])
    f_times = np.array([0,.5,1,1.5,1.75,2.15,2,2.25,3.25,3.75,4,3.5,2.5,2.4,2.3,2.6,2.7])
    f_vals = np.array([100,20,20,20,375,375,375,20,20,20,100,100,20,20,20,20,20])
    print(
        e_times.shape, e_vals.shape, p_times.shape,
        p_vals.shape, l_times.shape, l_vals.shape,
        f_times.shape, f_vals.shape)

    estradiol = spline_interp_conc(e_times, e_vals)
    progesterone = spline_interp_conc(p_times, p_vals)
    luteinizing = spline_interp_conc(l_times, l_vals)
    folliclestim = spline_interp_conc(f_times, f_vals)

    # X needs to be (samples, features)
    X_conc = np.stack([estradiol, progesterone, luteinizing, folliclestim]).T
    X_conc.shape

    X_conc[X_conc<0.] = 0.

    return X_conc


def fit_glm(X, y):

    n_samples, n_features = X.shape
    assert n_features == 4, "Input must have exactly 4 features per sample."

    # Add bias (intercept) term: shape becomes (n_samples, 5)
    X_aug = np.hstack([np.ones((n_samples, 1)), X])
    
    # Closed-form solution: w = (X^T X)^(-1) X^T y
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    weights = np.linalg.inv(XtX) @ Xty
    
    return weights



def run_glm():

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

    transition_timepoints = np.arange(0, 4)

    X_conc = get_factor_concentrations()

    w = {}
    for i in range(np.size(symbol_matrix,0)):
        for j in range(np.size(symbol_matrix,1)):

            y_train = np.zeros(4) * np.nan

            for stage in range(len(mats)):

                param = symbol_matrix[i,j]
                param_at_stage = mats[stage][i,j]

                y_train[stage] = param_at_stage

            symbol_name = 'w_{}'.format(symbol_matrix[i,j])

            w_ = fit_glm(
                X = X_conc[transition_timepoints.astype(int),:],
                y = y_train
            )

            w[symbol_name] = w_




def plot_matrix_cell_evolution(matrices, ffit=None):
    """
    Plot the evolution of each cell in a series of 5x5 matrices over time.
    
    Parameters:
    matrices (list of np.ndarray): A list of four 5x5 numpy arrays representing values at four timepoints.
    """
    assert len(matrices) == 4, "Input must be a list of four 5x5 matrices"
    for mat in matrices:
        assert mat.shape == (5, 5), "All matrices must be 5x5"

    fig, axes = plt.subplots(5, 5, figsize=(6,6), dpi=300, sharex=True, sharey=True)
    fig.suptitle("Probability evolution over estrous cycle")

    timepoints = np.arange(4)
    timepoints_f = np.arange(0,3.2,0.2)

    symbol_matrix = np.array([
        [r'$\delta_M$', r'$\gamma_{M \to F}$', r'$\gamma_{M \to H}$', r'$\gamma_{M \to S}$', r'$-\alpha_M$'],
        [r'$\delta_S$', r'$\gamma_{S \to F}$', r'$\gamma_{S \to H}$', r'$-\alpha_S$', r'$\gamma_{S \to M}$'],
        [r'$\delta_H$', r'$\gamma_{H \to F}$', r'$-\alpha_H$', r'$\gamma_{H \to S}$', r'$\gamma_{H \to M}$'],
        [r'$\delta_F$', r'$-\alpha_F$', r'$\gamma_{F \to H}$', r'$\gamma_{F \to S}$', r'$\gamma_{F \to M}$'],
        [r'$0$', r'$\beta_F$', r'$\beta_H$', r'$\beta_S$', r'$\beta_M$']
    ])

    for i in range(5):
        for j in range(5):
            values = [mat[i, j] for mat in matrices]
            ax = axes[i, j]
            ax.plot(timepoints, values, marker='.', color='k')
            if ffit is not None:
                fit_over_finer_grid = np.interp(timepoints_f, np.arange(4), ffit[:, i, j])
                ax.plot(timepoints_f, fit_over_finer_grid, marker='x', color='tab:orange')
            ax.set_title(symbol_matrix[i,j])
            if j == 0:
                ax.set_ylabel('$P$')
            ax.set_xticks(range(4), labels=['D','P','E','M'])
            ax.set_xlim([-0.5,3.5])
    plt.tight_layout()
    plt.show()
    fig.savefig('probability_evolution_by_stage_stepwise.svg')




def plot_transition_vs_E2(matrices, x_data=None, y_data=None, y_label='estradiol (ng/mL)'):

    num_values = 100

    assert len(matrices) == 4, "Input must be a list of four 5x5 matrices"
    for mat in matrices:
        assert mat.shape == (5, 5), "All matrices must be 5x5"

    fig, axes = plt.subplots(5, 5, figsize=(6,5), dpi=300)#, sharex=True, sharey=True)
    # fig.suptitle("Probability evolution over estrous cycle")

    if x_data is None:
        x_data = np.array([0, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4])
    if y_data is None:
        y_data = np.array([10, 20, 25, 40, 10, 5, 10, 5, 10])
    estradiol_level = spline_interp_conc(x_data, np.linspace(0,4,num_values), y_data)
    timepoints_f = np.linspace(0,4,num_values)

    plt.figure()
    plt.plot(timepoints_f, estradiol_level)
    plt.ylabel(y_label)

    # xmax = np.max(estradiol_level*1.05)

    symbol_matrix = np.array([
        [r'$\delta_M$', r'$\gamma_{M \to F}$', r'$\gamma_{M \to H}$', r'$\gamma_{M \to S}$', r'$-\alpha_M$'],
        [r'$\delta_S$', r'$\gamma_{S \to F}$', r'$\gamma_{S \to H}$', r'$-\alpha_S$', r'$\gamma_{S \to M}$'],
        [r'$\delta_H$', r'$\gamma_{H \to F}$', r'$-\alpha_H$', r'$\gamma_{H \to S}$', r'$\gamma_{H \to M}$'],
        [r'$\delta_F$', r'$-\alpha_F$', r'$\gamma_{F \to H}$', r'$\gamma_{F \to S}$', r'$\gamma_{F \to M}$'],
        [r'$0$', r'$\beta_F$', r'$\beta_H$', r'$\beta_S$', r'$\beta_M$']
    ])
    
    new_colors = cm.get_cmap('hsv')(np.linspace(0, 1, num_values))
    # _ymax = 0
    for i in range(5):
        for j in range(5):
            values = [mat[i, j] for mat in matrices]
            ax = axes[i,j]

            # linear interpolation
            # fit_over_finer_grid = np.interp(timepoints_f, np.arange(4)*24, values)

            # circular interpolation
            x_ = np.arange(5)
            y_ = np.append(values, values[0])
            spline_interp = scipy.interpolate.CubicSpline(x_, y_, bc_type="periodic")
            fit_over_finer_grid = spline_interp(timepoints_f)
            fit_over_finer_grid[fit_over_finer_grid<0] = 0

            # if np.max(fit_over_finer_grid) > _ymax:
            #     print(np.max(fit_over_finer_grid))
            #     _ymax = np.max(fit_over_finer_grid)

            # if i==0 and j==0:
            #     print(timepoints_f, fit_over_finer_grid)
            ax.scatter(
                estradiol_level[np.arange(num_values)],
                fit_over_finer_grid,
                s=2.5, color=new_colors)
            # ax.plot(timepoints, values, marker='.', color='k')
            ax.set_title(symbol_matrix[i,j])
            # if j == 0:
                # ax.set_ylabel('$P$')
            # ax.set_xticks(range(4), labels=['D','P','E','M'])
            # ax.set_xlim([0,40])
            ax.set_ylabel('')
            # ax.set_ylim([0,_ymax])
            # ax.set_yticks(np.arange(0,_ymax,_ymax/5), labels=[])
            ax.set_xlabel('')
            # ax.set_xticks(np.arange(0,xmax,10), labels=[])
            if 'delta' in symbol_matrix[i,j] or symbol_matrix[i,j]=='$0$':
                ax.set_ylabel('Prob')
                # ax.set_yticks(np.arange(0,_ymax,_ymax/5))
            if 'beta' in symbol_matrix[i,j] or symbol_matrix[i,j]=='$0$':
                # ax.set_xticks(np.arange(0,50,10), labels=np.arange(0,50,10))
                ax.set_xlabel(y_label)
                

    fig.tight_layout()
    fig.show()
    savename = 'probability_evolution_vs_{}.svg'.format(y_label.split(' ')[0])
    print(savename)
    fig.savefig(savename)