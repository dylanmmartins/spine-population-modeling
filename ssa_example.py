
#############################
# Example application: Exponential growth model
#############################

# Initial population size
init = [100]  # using a list for one-dimensional state

# Time grid: from 0 to 20 in steps of 1
times = np.arange(0, 21, 1)

# Environment function: fluctuating environment
def fluctuating2(x):
    return (math.sin(x * 2) + 1) / 1.5

# Demographic functions for birth and death
def births(b, env_res):
    return b * env_res

def deaths(d, env_res):
    return d

# Combine parameters into a dictionary.
param = {
    'b': 0.03,
    'd': 0.027,
    'env': fluctuating2,
    'births': births,
    'deaths': deaths
}

# Intensity function using demographic functions and environment.
def inten(t, X, param):
    env_val = param['env'](t)
    b_int = X * param['births'](param['b'], env_val)
    d_int = X * param['deaths'](param['d'], env_val)
    # Return two intensities: one for birth and one for death.
    return [b_int, d_int]

# State change matrix due to events: birth increases population by 1, death decreases by 1.
pproc = np.array([[1],   # birth: add 1
                  [-1]]) # death: subtract 1

# Set a random seed for reproducibility
np.random.seed(20170915)
random.seed(20170915)

# Run simulation using gillespie (homogeneous Poisson process sampler)
res_gillespie = gillespie(init, times, param, inten, pproc, hpp)
print("Results from gillespie:")
print(res_gillespie)

# Reset the seed for reproducibility before running the second simulation
np.random.seed(20170915)
random.seed(20170915)

# Run simulation using gillespie_plus (nonhomogeneous Poisson process sampler)
res_gillespie_plus = gillespie_plus(init, times, param, inten, pproc, nhpp)
print("\nResults from gillespie_plus:")
print(res_gillespie_plus)
