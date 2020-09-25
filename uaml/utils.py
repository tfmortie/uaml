"""
Some important functions that are used throughout the module

Author: Thomas Mortier
"""
import numpy as np

def get_most_common_el(x):
    """Function which returns most common element.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Array of elements.

    Returns
    -------
    y : object
        Most common element from x.
    """ 
    (values, counts) = np.unique(x,return_counts=True)
    return values[np.argmax(counts)]

def entropy(p, eps=0.000001):
    """Function which calculates the Shannon entropy.

    Parameters
    ----------
    p : ndarray, shape (n_samples,)
        Array of probabilities.
    eps : float, default=1e-5
        Avoid division by zero in calculation of log2.

    Returns
    -------
    ent : float
        Shannon entropy of p.
    """ 
    ent = -np.sum(p*np.log2(p.clip(min=eps)))
    return ent

def calculate_uncertainty_jsd(P):
    """Function which calculates aleatoric and epistemic uncertainty based on Jensen-Shannon divergence.

    See https://arxiv.org/abs/1910.09457 and https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    Parameters
    ----------
    P : ndarray, shape (n_samples, n_mc_samples, n_classes) 
        Array of probability distributions.

    Returns
    -------
    u_a : ndarray, shape (n_samples,)
        Array of aleatoric uncertainty estimates for each sample.
    u_e : ndarray, shape (n_samples,)
        Array of epistemic uncertainty estimates for each sample.
    """ 
    u_t = np.apply_along_axis(entropy, 1, np.mean(P, axis=1))
    u_a = np.mean(np.apply_along_axis(entropy, 2, P), axis=1) 
    u_e = u_t - u_a
    return u_a, u_e
