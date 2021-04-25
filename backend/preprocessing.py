import numpy as np
from scipy.ndimage import shift
from sklearn.preprocessing import StandardScaler
from gaussianize import Gaussianize 

def rolling_window(x, k, sparse=True):
    """compute rolling windows from timeseries

    Args:
        x ([2d array]): x contains the time series in the shape (timestep, sample).
        k ([int]): window length.
        sparse (bool): Cut off the final windows containing NA. Defaults to True.

    Returns:
        [3d array]: array of rolling windows in the shape (window, timestep, sample).
    """    
    arr = np.tile(x, (k, 1, 1))
    for i in range(k):
            arr[i] = shift(arr[i], (-i, 0), order=0, cval=np.nan)
            
    if not sparse:
        return arr

    return arr[:, :-(k-1)]