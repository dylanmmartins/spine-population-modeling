import numpy as np
import scipy.interpolate

from .helpers import find_closest_timestamp

# environmental function
def estradiol_lvl(t):

    x_data = np.array([0, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4])
    y_data = np.array([10, 20, 25, 40, 10, 5, 10, 5, 10])
    spline_interp = scipy.interpolate.CubicSpline(x_data*24, y_data, bc_type='periodic')
    estradiol_level_spline = spline_interp(np.arange(0,96.25,0.25))

    # `t` must be aligned so that it always begins at the start
    # of diestrus stage.

    # Mod so that any number of hours falls into the 0 to 96 hour range
    t = t%(4*24)
    search_range = np.arange(0,96.25,0.25)
    cycle_ind, _ = find_closest_timestamp(search_range, t)

    # return the current estradiol level
    return estradiol_level_spline[cycle_ind]