"""
Contains model-specific functions that are parallelizable.

Author: Thomas Mortier
Date: June 2022
"""
import multiprocessing as mp
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from .utils import calculate_uncertainty_jsd

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
    global FITSTATE
    # set global state
    FITSTATE["model"] = model
    FITSTATE["results"] = []
    # check how many workers we need
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    # intialize the pool with workers
    fit_pool = mp.Pool(num_workers)
    # add fit tasks to pool
    num_models_per_worker = [
        len(a) for a in np.array_split(range(model.ensemble_size), num_workers)
    ]
    for i in range(num_workers):
        fit_pool.apply_async(_fit, args=(num_models_per_worker[i],), callback=_add_fit)
    fit_pool.close()
    fit_pool.join()
    ensemble = FITSTATE["results"]

    return ensemble


def predict(model, X):
    """Represents a general predict process.

    Parameters
    ----------
    model : uncertainty-aware model
        Represents the fitted uncertainty-aware model.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input samples.

    Returns
    -------
    preds : ndarray, shape (n_samples, model.ensemble_size)
        Returns an array of predicted class labels.
    """
    global PREDICTSTATE
    # set global state
    PREDICTSTATE["model"] = model
    PREDICTSTATE["X"] = X
    PREDICTSTATE["results"] = []
    # check how many workers we need
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    # intialize the pool with workers
    predict_pool = mp.Pool(num_workers)
    # add predict tasks to pool
    num_models_per_worker = [
        len(a) for a in np.array_split(range(len(model.ensemble_)), num_workers)
    ]
    start_ind = 0
    for i in range(num_workers):
        predict_pool.apply_async(
            _predict, args=(start_ind, num_models_per_worker[i]), callback=_add_predict
        )
        start_ind += num_models_per_worker[i]
    predict_pool.close()
    predict_pool.join()
    # get predictions, sort and stack
    preds = PREDICTSTATE["results"]
    preds.sort(key=lambda x: x[0])
    preds = np.hstack([p[1] for p in preds])

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
    global PREDICTSTATE
    # set global state
    PREDICTSTATE["model"] = model
    PREDICTSTATE["X"] = X
    PREDICTSTATE["results"] = []
    # check how many workers we need
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    # intialize the pool with workers
    predict_proba_pool = mp.Pool(num_workers)
    # add predict tasks to pool
    num_models_per_worker = [
        len(a) for a in np.array_split(range(len(model.ensemble_)), num_workers)
    ]
    start_ind = 0
    for i in range(num_workers):
        predict_proba_pool.apply_async(
            _predict_proba,
            args=(start_ind, num_models_per_worker[i]),
            callback=_add_predict_proba,
        )
        start_ind += num_models_per_worker[i]
    predict_proba_pool.close()
    predict_proba_pool.join()
    # get predictions, sort and stack
    probs = PREDICTSTATE["results"]
    probs.sort(key=lambda x: x[0])
    probs = np.concatenate([p[1] for p in probs], axis=1)

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
    global UNCERTAINTYSTATE
    # set global state
    UNCERTAINTYSTATE["P"] = P
    UNCERTAINTYSTATE["results"] = []
    # check how many workers we need
    if not n_jobs is None:
        num_workers = max(min(mp.cpu_count(), n_jobs), 1)
    else:
        num_workers = 1
    # intialize the pool with workers
    get_uncertainty_pool = mp.Pool(num_workers)
    # add uncertainty tasks to pool
    num_samples_per_worker = [
        len(a) for a in np.array_split(range(P.shape[0]), num_workers)
    ]
    start_ind = 0
    for i in range(num_workers):
        get_uncertainty_pool.apply_async(
            _get_uncertainty_jsd,
            args=(start_ind, num_samples_per_worker[i]),
            callback=_add_get_uncertainty_jsd,
        )
        start_ind += num_samples_per_worker[i]
    get_uncertainty_pool.close()
    get_uncertainty_pool.join()
    # get uncertainties, sort and stack
    u = UNCERTAINTYSTATE["results"]
    u.sort(key=lambda x: x[0])
    u_a, u_e = np.concatenate([ui[1] for ui in u]), np.concatenate([ui[2] for ui in u])

    return u_a, u_e


def _add_fit(models):
    global FITSTATE
    FITSTATE["results"].extend(models)


def _fit(n_models):
    global FITSTATE
    models = []
    ss = StratifiedShuffleSplit(
        n_splits=n_models,
        train_size=FITSTATE["model"].b_size_,
        random_state=FITSTATE["model"].random_state,
    )
    try:
        for train_index, _ in ss.split(FITSTATE["model"].X_, FITSTATE["model"].y_):
            model = {}
            # create estimator, given parameters of base estimator and bootstrap sample indices
            model["clf"] = clone(FITSTATE["model"].estimator)
            model["ind"] = train_index
            model["clf"].fit(
                FITSTATE["model"].X_[model["ind"], :],
                FITSTATE["model"].y_[model["ind"]],
            )
            models.append(model)
    except Exception as e:
        print("Exception caught while fitting ensemble: {0}".format(e), flush=True)

    return models


def _add_predict(batch_preds):
    global PREDICTSTATE
    PREDICTSTATE["results"].append(batch_preds)


def _predict(i, n_models):
    global PREDICTSTATE
    batch_preds = []
    for m_i in range(i, i + n_models):
        batch_preds.append(
            PREDICTSTATE["model"]
            .ensemble_[m_i]["clf"]
            .predict(PREDICTSTATE["X"])
            .reshape(-1, 1)
        )
    batch_preds = np.hstack(batch_preds)

    return (i, batch_preds)


def _add_predict_proba(batch_probs):
    global PREDICTSTATE
    PREDICTSTATE["results"].append(batch_probs)


def _predict_proba(i, n_models):
    global PREDICTSTATE
    batch_probs = []
    for m_i in range(i, i + n_models):
        batch_probs.append(
            np.expand_dims(
                PREDICTSTATE["model"]
                .ensemble_[m_i]["clf"]
                .predict_proba(PREDICTSTATE["X"]),
                axis=1,
            )
        )
    batch_probs = np.concatenate(batch_probs, axis=1)

    return (i, batch_probs)


def _add_get_uncertainty_jsd(batch_u):
    global UNCERTAINTYSTATE
    UNCERTAINTYSTATE["results"].append(batch_u)


def _get_uncertainty_jsd(i, n_samples):
    global UNCERTAINTYSTATE
    batch_ua, batch_ue = calculate_uncertainty_jsd(
        UNCERTAINTYSTATE["P"][i : i + n_samples]
    )

    return (i, batch_ua, batch_ue)
