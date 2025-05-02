import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

mat = pd.read_csv('./transition_matrix.csv', index_col=0)
mat = mat[['NS','filopodium','thin','stubby','mushroom']]
mat = mat.reindex(['NS','filopodium','thin','stubby','mushroom'])
mat = mat.T

# birth rates (NS -> spine type)
alpha_m = mat.loc['NS','mushroom']
alpha_s = mat.loc['NS','stubby']
alpha_t = mat.loc['NS','thin']
alpha_f = mat.loc['NS','filopodium']

# death rates (spine type -> NS)
beta_m = mat.loc['mushroom', 'NS']
beta_s = mat.loc['stubby', 'NS']
beta_t = mat.loc['thin', 'NS']
beta_f = mat.loc['filopodium', 'NS']

# transitions (filopodium -> spine type)
gamma_fm = mat.loc['filopodium', 'mushroom']
gamma_fs = mat.loc['filopodium', 'stubby']
gamma_ft = mat.loc['filopodium', 'thin']

# transitions (mushroom -> spine type)
gamma_mf = mat.loc['mushroom', 'filopodium']
gamma_ms = mat.loc['mushroom', 'stubby']
gamma_mt = mat.loc['mushroom', 'thin']

# transitions (stubby -> spine type)
gamma_sm = mat.loc['stubby', 'mushroom']
gamma_sf = mat.loc['stubby', 'filopodium']
gamma_st = mat.loc['stubby', 'thin']

# transitions (thin -> spine type)
gamma_tm = mat.loc['thin', 'mushroom']
gamma_ts = mat.loc['thin', 'stubby']
gamma_tf = mat.loc['thin', 'filopodium']

# Define the ODE system
def system(t, y):
    F, T, S, M = y
    
    # filipodium
    dF_dt = (alpha_f - beta_f*F +
             gamma_tf*T + gamma_sf*S + gamma_mf*M -
             F*(gamma_ft + gamma_fs + gamma_fm))
    
    # thin
    dF_dt = (alpha_t - beta_t*T +
            gamma_tf*T + gamma_sf*S + gamma_mf*M -
            F*(gamma_ft + gamma_fs + gamma_fm))
    
    # stubby
    dF_dt = (alpha_f - beta_f*F +
            gamma_tf*T + gamma_sf*S + gamma_mf*M -
            F*(gamma_ft + gamma_fs + gamma_fm))
    
    # mushroom
    dF_dt = (alpha_f - beta_f*F +
            gamma_tf*T + gamma_sf*S + gamma_mf*M -
            F*(gamma_ft + gamma_fs + gamma_fm))
    
    return [dF_dt, dH_dt, dS_dt, dM_dt]

# Initial conditions: F0, H0, S0, M0
y0 = [1.0, 1.0, 1.0, 1.0]

# Time span for simulation
t_span = (0, 50)
t_eval = np.linspace(*t_span, 500)

# Solve ODE
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)


plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='F')
plt.plot(sol.t, sol.y[1], label='H')
plt.plot(sol.t, sol.y[2], label='S')
plt.plot(sol.t, sol.y[3], label='M')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ODE System Solution')
plt.legend()
plt.grid(True)
plt.show()