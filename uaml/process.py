"""
Contains model-specific functions that are parallelizable.

Author: Thomas Mortier
Date: June 2022
"""
import multiprocessing as mp
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.exceptions import NotFittedError
from .utils import calculate_uncertainty_jsd
from joblib import Parallel, delayed, parallel_backend

""" some state vars that are needed """
FITSTATE = {"model": None, "X": None, "results": []}
PREDICTSTATE = {"model": None, "X": None, "results": []}
UNCERTAINTYSTATE = {"P": None, "results": []}


def fit(model):
    """Represents a general fit process.

    Parameters
    ----------
    model : UAClassifier
        Represents the fitted uncertainty-aware model.

    Returns
    -------
    ensemble : list
        Returns a list of fitted base estimators.
    """
    ensemble = []
    # check how many workers we need
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    try:
        # start fitting
        with parallel_backend("loky"):
            ensemble = Parallel(n_jobs=model.n_jobs)(
                delayed(_fit)(len(a), model)
                for a in np.array_split(range(model.ensemble_size), num_workers)
            )
        ensemble = [ens for s_ensemble in ensemble for ens in s_ensemble]
    except NotFittedError as e:
        raise NotFittedError("Error {}, model fitting failed!".format(e))

    return ensemble


def predict(model, X):
    """Represents a general predict process.

    Parameters
    ----------
    model : UAClassifier
        Represents the fitted uncertainty-aware model.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input samples.

    Returns
    -------
    preds : ndarray, shape (n_samples, model.ensemble_size)
        Returns an array of predicted class labels.
    """
    preds = np.array([])
    # check how many workers we need
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    try:
        # construct config for predict
        pr_conf, start_ind = [], 0
        for a in np.array_split(range(len(model.ensemble_)), num_workers):
            pr_conf.append((start_ind, len(a)))
            start_ind += len(a)
        # obtain predictions
        with parallel_backend("loky"):
            preds = Parallel(n_jobs=model.n_jobs)(
                delayed(_predict)(X, s_i, la, model) for (s_i, la) in pr_conf
            )
        preds.sort(key=lambda x: x[0])
        preds = np.hstack([p[1] for p in preds])
    except RuntimeError as e:
        raise RuntimeError("Error {}, obtaining predictions failed!".format(e))

    return preds


def predict_proba(model, X):
    """Represents a general predict probabilities process.

    Parameters
    ----------
    model : uncertainty-aware model
        Represents the fitted uncertainty-aware model.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input samples.

    Returns
    -------
    probs : ndarray
        Returns the probability of the sample for each class in the model.
    """
    probs = np.array([])
    # check how many workers we need
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    try:
        # construct config for predict_proba
        pr_conf, start_ind = [], 0
        for a in np.array_split(range(len(model.ensemble_)), num_workers):
            pr_conf.append((start_ind, len(a)))
            start_ind += len(a)
        # obtain probabilities
        with parallel_backend("loky"):
            probs = Parallel(n_jobs=model.n_jobs)(
                delayed(_predict_proba)(X, s_i, la, model) for (s_i, la) in pr_conf
            )
        probs.sort(key=lambda x: x[0])
        probs = np.concatenate([p[1] for p in probs], axis=1)
    except RuntimeError as e:
        raise RuntimeError("Error {}, obtaining probabilities failed!".format(e))

    return probs


def get_uncertainty_jsd(P, n_jobs):
    """Represents a general jsd uncertainty calculation process.

    Parameters
    ----------
    P : ndarray, shape (n_samples, ensemble_size, n_classes)
        Array of probability distributions.
    n_jobs : int
        Number of cores to use.

    Returns
    -------
    u_a : ndarray, shape (n_samples,)
        Array of aleatoric uncertainty estimates for each sample.
    u_e : ndarray, shape (n_samples,)
        Array of epistemic uncertainty estimates for each sample.
    """
    u_a, u_e = np.array([]), np.array([])
    # check how many workers we need
    if not n_jobs is None:
        num_workers = max(min(mp.cpu_count(), n_jobs), 1)
    else:
        num_workers = 1
    try:
        # construct config for get_uncertainty_jsd
        pr_conf, start_ind = [], 0
        for a in np.array_split(range(P.shape[0]), num_workers):
            pr_conf.append((start_ind, len(a)))
            start_ind += len(a)
        # obtain probabilities
        with parallel_backend("loky"):
            u = Parallel(n_jobs=n_jobs)(
                delayed(_get_uncertainty_jsd)(P, s_i, la) for (s_i, la) in pr_conf
            )
        u.sort(key=lambda x: x[0])
        u_a, u_e = np.concatenate([ui[1] for ui in u]), np.concatenate(
            [ui[2] for ui in u]
        )
    except RuntimeError as e:
        raise RuntimeError("Error {}, calculating uncertainties failed!".format(e))

    return u_a, u_e


def _fit(n_models, m):
    models = []
    ss = StratifiedShuffleSplit(
        n_splits=n_models,
        train_size=m.b_size_,
        random_state=m.random_state,
    )
    try:
        for train_index, _ in ss.split(m.X_, m.y_):
            model = {}
            # create estimator, given parameters of base estimator and bootstrap sample indices
            model["clf"] = clone(m.estimator)
            model["ind"] = train_index
            model["clf"].fit(
                m.X_[model["ind"], :],
                m.y_[model["ind"]],
            )
            models.append(model)
    except Exception as e:
        print("Exception caught while fitting ensemble: {0}".format(e), flush=True)

    return models


def _predict(X, i, n_models, model):
    batch_preds = []
    for m_i in range(i, i + n_models):
        batch_preds.append(model.ensemble_[m_i]["clf"].predict(X).reshape(-1, 1))
    batch_preds = np.hstack(batch_preds)

    return (i, batch_preds)


def _predict_proba(X, i, n_models, model):
    batch_probs = []
    for m_i in range(i, i + n_models):
        batch_probs.append(
            np.expand_dims(
                model.ensemble_[m_i]["clf"].predict_proba(X),
                axis=1,
            )
        )
    batch_probs = np.concatenate(batch_probs, axis=1)

    return (i, batch_probs)


def _get_uncertainty_jsd(P, i, n_samples):
    batch_ua, batch_ue = calculate_uncertainty_jsd(P[i : i + n_samples])

    return (i, batch_ua, batch_ue)
