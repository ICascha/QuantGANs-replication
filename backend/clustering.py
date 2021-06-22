import matplotlib.pyplot as plt
import numpy as np
from backend.preprocessing import rolling_window
from sklearn.cluster import KMeans


def clustering(real_returns, synth_returns, window_length, n_clusters, random_state=0):
    """cluster real and synthetic returns. Synthetic returns are used to determine cluster
    

    Args:
        real_returns (2d-array): real returns in shape (timesteps, 1)
        synth_returns (2d-array): synthetic returns in shape (timesteps, samples)
        window_length (int): length of moving window, skips across 2*window_length when moving
        n_clusters (int): number of clusters
        random_state (int, optional): seed of kmeans RNG. Defaults to 0.

    Returns:
        tuple: tuple of real and synthetic samples, their corresponding bins, 
        array of total frequency of both real and synthetic returns and a fitted kmeans object
    """    

    real_samples = rolling_window(real_returns, window_length).T.reshape((-1, window_length))
    # We cluster based on synth samples, so we skip ahead twice the window length
    # to avoid interdependencies of windows
    synth_samples = rolling_window(synth_returns, window_length)[:, ::window_length*2].T.reshape((-1, window_length))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(synth_samples)

    real_bins = kmeans.predict(real_samples)
    synth_bins = kmeans.predict(synth_samples)

    real_freq = np.bincount(real_bins)
    synth_freq = np.bincount(synth_bins)

    total_freq = np.stack([real_freq, synth_freq])
    total_freq = total_freq / total_freq.sum(axis=1, keepdims=True)

    return real_samples, synth_samples, real_bins, synth_bins, total_freq, kmeans

def plot_clustering(synth_samples, synth_bins, total_freq, alpha, bins, figsize):
    """Plut 9 clusters on a 3x3 grid

    Args:
        synth_samples (2d-array): synthetic windows used (non-overlapping)
        synth_bins (1d-array): array of bins
        alpha (float): transparancy of plotted returns, 1 = opaque, 0 = invisible
        bins (int): bins considered in plotting
        figsize(tuple): tuple of (length, height) of figure

    Returns:
        tuple: tuple of matplotlib.pyplot figure and axes object
    """    
    fig, axs = plt.subplots(figsize=figsize, nrows=3, ncols=3, sharex=True, sharey=True)

    i = 0
    j = 0
    for bin in bins:
        axs[i, j].plot(np.moveaxis(synth_samples[synth_bins == bin], 0, -1), color='orange', alpha=alpha)
        axs[i, j].set_title('# paths: {}'.format(int((synth_bins == bin).sum())))
        axs[i, j].grid(axis='y', which='major')

        j += 1
        j = j % 3
        if j == 0:
            i += 1

    plt.setp(axs[-1, :], xlabel='time (days)')
    plt.setp(axs[:, 0], ylabel='log return')

    return fig, axs
