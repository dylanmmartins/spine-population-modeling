import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import scipy.interpolate

import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['svg.fonttype'] = 'none'


def compute_C_from_P(P):

    P = np.array(P, dtype=float)
    
    # Extract the 4x4 submatrix (rows 0-3, cols 1-4) which contains the gamma parameters (off-diagonal)
    # Diagonal deltas are in P[i, i] (0 to 3)
    # We'll build a 4x4 matrix of gammas, with diagonal zeros.
    
    A = np.zeros((4,4))
    
    for i in range(4):
        delta_i = P[i, i]          # diagonal delta
        # gamma params in row i, columns 1 to 4 (index 1 to 4)
        # but only columns 1 to 4, excluding diagonal column i
        # diagonal in terms of gammas: we want to skip P[i, i] but i and column indices don't align because gamma start at col 1
        # So build row gamma vector (cols 1 to 4)
        gamma_row = P[i, 1:5]  # length 4
        
        # The diagonal of A corresponds to gamma_row element where col = i
        # Because for i=0 -> skip gamma_row[0], i=1 -> skip gamma_row[1], etc.
        denom = delta_i + np.sum(np.delete(gamma_row, i))
        
        for j in range(4):
            if j == i:
                A[i,j] = 0.0
            else:
                A[i,j] = gamma_row[j] / denom

    beta_F = P[4, 1]
    beta_H = P[4, 2]
    beta_S = P[4, 3]
    beta_M = P[4, 4]
    
    # Order: [M, S, H, F]
    B = np.array([beta_M, beta_S, beta_H, beta_F])
    
    return A, B


def compute_jacobian(P):
    # Unpack parameters from matrix P
    delta_M = P[0, 0]
    gamma_M_to_F = P[0, 1]
    gamma_M_to_H = P[0, 2]
    gamma_M_to_S = P[0, 3]

    delta_S = P[1, 0]
    gamma_S_to_F = P[1, 1]
    gamma_S_to_H = P[1, 2]
    gamma_S_to_M = P[1, 4]

    delta_H = P[2, 0]
    gamma_H_to_F = P[2, 1]
    gamma_H_to_S = P[2, 3]
    gamma_H_to_M = P[2, 4]

    delta_F = P[3, 0]
    gamma_F_to_H = P[3, 2]
    gamma_F_to_S = P[3, 3]
    gamma_F_to_M = P[3, 4]

    # Denominator terms for each row
    D_F = delta_F + gamma_F_to_H + gamma_F_to_S + gamma_F_to_M
    D_H = delta_H + gamma_H_to_F + gamma_H_to_S + gamma_H_to_M
    D_S = delta_S + gamma_S_to_F + gamma_S_to_H + gamma_S_to_M
    D_M = delta_M + gamma_M_to_F + gamma_M_to_H + gamma_M_to_S

    # Build Jacobian matrix
    J = np.array([
        [-1,                gamma_H_to_F / D_F, gamma_S_to_F / D_F, gamma_M_to_F / D_F],  # dF/d*
        [gamma_F_to_H / D_H, -1,                gamma_S_to_H / D_H, gamma_M_to_H / D_H],  # dH/d*
        [gamma_F_to_S / D_S, gamma_H_to_S / D_S, -1,                gamma_M_to_S / D_S],  # dS/d*
        [gamma_F_to_M / D_M, gamma_H_to_M / D_M, gamma_S_to_M / D_M, -1]                 # dM/d*
    ])

    return J

def plot_phase_portrait(P, init_conditions, plot_vars=['F', 'H'], ax=None):
    # Extract parameters from P matrix
    delta_M, gMF, gMH, gMS = P[0, 0], P[0, 1], P[0, 2], P[0, 3]
    delta_S, gSF, gSH, gSM = P[1, 0], P[1, 1], P[1, 2], P[1, 4]
    delta_H, gHF, gHS, gHM = P[2, 0], P[2, 1], P[2, 3], P[2, 4]
    delta_F, gFH, gFS, gFM = P[3, 0], P[3, 2], P[3, 3], P[3, 4]
    beta_F, beta_H, beta_S, beta_M = P[4, 1], P[4, 2], P[4, 3], P[4, 4]

    # Variable mapping
    var_idx = {'F': 0, 'H': 1, 'S': 2, 'M': 3}
    all_vars = ['F', 'H', 'S', 'M']

    # Determine free and fixed variables
    free_vars = plot_vars
    fixed_vars = [v for v in all_vars if v not in free_vars]

    # Generate grid for free variables
    grid_size = 10
    v1_vals = np.linspace(0, 20, grid_size)
    v2_vals = np.linspace(0, 20, grid_size)
    V1, V2 = np.meshgrid(v1_vals, v2_vals)

    # Initialize derivative arrays
    dV1 = np.zeros_like(V1)
    dV2 = np.zeros_like(V2)

    # Loop over grid points
    for i in range(grid_size):
        for j in range(grid_size):
            # Initialize full state vector from initial conditions
            state = init_conditions.copy()
            state[var_idx[free_vars[0]]] = V1[i, j]
            state[var_idx[free_vars[1]]] = V2[i, j]

            F, H, S, M = state

            # Compute time derivatives (df/dt, etc.)
            dF = beta_F - delta_F * F + gHF * H + gSF * S + gMF * M - F * (gFH + gFS + gFM)
            dH = beta_H - delta_H * H + gFH * F + gSH * S + gMH * M - H * (gHF + gHS + gHM)
            dS = beta_S - delta_S * S + gFS * F + gHS * H + gMS * M - S * (gSF + gSH + gSM)
            dM = beta_M - delta_M * M + gFM * F + gHM * H + gSM * S - M * (gMF + gMH + gMS)

            derivatives = [dF, dH, dS, dM]

            dV1[i, j] = derivatives[var_idx[free_vars[0]]]
            dV2[i, j] = derivatives[var_idx[free_vars[1]]]

    # Normalize vectors for quiver plot
    magnitude = np.sqrt(dV1**2 + dV2**2)
    dV1 /= (magnitude + 1e-8)
    dV2 /= (magnitude + 1e-8)

    J = compute_jacobian(P)
    eigvals, eigvecs = np.linalg.eig(J)

    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=300, figsize=(3,3))

    # t = np.linspace(0, 40, 1000)
    # for i in range(2):
    #     lam = eigvals[i]
    #     v = eigvecs[:, i]
    #     for alpha in [-2, -1, 1, 2]:
    #         traj = np.outer(np.exp(lam * t), alpha * v)
    #         plt.plot(traj[:, 0], traj[:, 1], 'r', lw=1)

    ax.quiver(V1, V2, dV1, dV2, angles='xy')
    ax.set_xlabel(free_vars[0])
    ax.set_ylabel(free_vars[1])
    ax.set_xlim([0,20])
    ax.set_ylim([0,20])
    ax.axis('equal')
    # plt.title(f'Phase Portrait: {free_vars[0]} vs {free_vars[1]} (others fixed)')
    plt.tight_layout()



def main():

    symbol_matrix = np.array([
        [r'$\delta_M$', r'$\gamma_{M \to F}$', r'$\gamma_{M \to H}$', r'$\gamma_{M \to S}$', r'$-\lambda_M$'],
        [r'$\delta_S$', r'$\gamma_{S \to F}$', r'$\gamma_{S \to H}$', r'$-\lambda_S$', r'$\gamma_{S \to M}$'],
        [r'$\delta_H$', r'$\gamma_{H \to F}$', r'$-\lambda_H$', r'$\gamma_{H \to S}$', r'$\gamma_{H \to M}$'],
        [r'$\delta_F$', r'$-\lambda_F$', r'$\gamma_{F \to H}$', r'$\gamma_{F \to S}$', r'$\gamma_{F \to M}$'],
        [r'$0$', r'$\beta_F$', r'$\beta_H$', r'$\beta_S$', r'$\beta_M$']
    ])

    DtoP = np.load('spinePopModels/transition_mats/DtoP_transition_matrix.npy')
    PtoE = np.load('spinePopModels/transition_mats/PtoE_transition_matrix.npy')
    EtoM = np.load('spinePopModels/transition_mats/EtoM_transition_matrix.npy')
    MtoD = np.load('spinePopModels/transition_mats/MtoD_transition_matrix.npy')

    A, B = compute_C_from_P(PtoE)

    x = np.linalg.solve((np.eye(4)-A), B)
    print("Equilibrium values:")
    print(f"F* = {x[0]:.4f}")
    print(f"H* = {x[1]:.4f}")
    print(f"S* = {x[2]:.4f}")
    print(f"M* = {x[3]:.4f}")

    J = compute_jacobian(MtoD)

    eigenvalues, eigenvectors = np.linalg.eig(J)
    print('vals:')
    print(eigenvalues.round(4))
    print('vecs:')
    for i in range(4):
        print(eigenvectors[:,i].round(4))
