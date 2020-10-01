"""
Contains model-specific functions that are parallelizable

Author: Thomas Mortier
"""
import multiprocessing as mp
import numpy as np
import uaml.utils as u

fit_state = {"model" : None,
        "X" : None,
        "results": []}
predict_state = {"model": None, 
        "X" : None,
        "results": []}
uncertainty_state = {"P": None,
        "results" : []}

def _add_fit(models):
    global fit_state
    fit_state["results"].extend(models)

def _fit(n_models):
    global fit_state
    models = []
    for _ in range(n_models):
        model = {}
        # Create estimator, given parameters of base estimator and bootstrap sample indices
        model["clf"] = type(fit_state["model"].estimator)(**fit_state["model"].estimator.get_params())
        model["ind"] = fit_state["model"].random_state_.randint(0, fit_state["model"].X_.shape[0], size=fit_state["model"].n_samples_)
        model["clf"].fit(fit_state["model"].X_[model["ind"], :], fit_state["model"].y_[model["ind"]])
        models.append(model)

    return models

def _add_predict(batch_preds):
    global predict_state
    predict_state["results"].append(batch_preds)

def _predict(i, n_models):
    global predict_state
    batch_preds = []
    for m_i in range(i, i+n_models):
        if predict_state["model"].n_outputs_ > 1:
            batch_preds.append(np.expand_dims(predict_state["model"].ensemble_[m_i]["clf"].predict(predict_state["X"]), axis=1))
        else:
            batch_preds.append(predict_state["model"].ensemble_[m_i]["clf"].predict(predict_state["X"]).reshape(-1, 1))
    if predict_state["model"].n_outputs_ > 1:
        batch_preds = np.concatenate(batch_preds, axis=1) 
    else:
        batch_preds = np.hstack(batch_preds)
    
    return (i, batch_preds)

def _add_predict_proba(batch_probs):
    global predict_state
    predict_state["results"].append(batch_probs)

def _predict_proba(i, n_models):
    global predict_state
    batch_probs = []
    for m_i in range(i, i+n_models):
        if predict_state["model"].n_outputs_ > 1:
            probs_list = predict_state["model"].ensemble_[m_i]["clf"].predict_proba(predict_state["X"])
            batch_probs.append(np.expand_dims(np.concatenate([np.expand_dims(p, axis=1) for p in probs_list], axis=1), axis=1))
        else:
            batch_probs.append(np.expand_dims(predict_state["model"].ensemble_[m_i]["clf"].predict_proba(predict_state["X"]), axis=1))
    batch_probs = np.concatenate(batch_probs, axis=1)

    return (i, batch_probs)

def _add_get_uncertainty_jsd(batch_u):
    global uncertainty_state
    uncertainty_state["results"].append(batch_u)

def _get_uncertainty_jsd(i, n_samples):
    global uncertainty_state
    batch_ua, batch_ue = u.calculate_uncertainty_jsd(uncertainty_state["P"][i:i+n_samples])

    return (i, batch_ua, batch_ue) 

def fit(model):
    """Represents a general fit process.

    Parameters
    ----------
    model : uncertainty-aware model
        Represents the fitted uncertainty-aware model.

    Returns
    -------
    ensemble : list
        Returns a list of fitted base estimators.
    """
    global fit_state
    # Set global state 
    fit_state["model"] = model
    fit_state["results"] = []
    # Check how many workers we need 
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    # Intialize the pool with workers
    fit_pool = mp.Pool(num_workers)
    # Add fit tasks to pool
    num_models_per_worker = [len(a) for a in np.array_split(range(model.n_mc_samples), num_workers)]
    for i in range(num_workers):
        fit_pool.apply_async(_fit, args=(num_models_per_worker[i],), callback=_add_fit)
    fit_pool.close()
    fit_pool.join()
    ensemble = fit_state["results"]

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
    preds : ndarray
        Returns an array of predicted class labels.
    """
    global predict_state
    # Set global state 
    predict_state["model"] = model
    predict_state["X"] = X
    predict_state["results"] = []
    # Check how many workers we need 
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    # Intialize the pool with workers
    predict_pool = mp.Pool(num_workers)
    # Add predict tasks to pool
    num_models_per_worker = [len(a) for a in np.array_split(range(len(model.ensemble_)), num_workers)]
    start_ind = 0
    for i in range(num_workers):
        predict_pool.apply_async(_predict, args=(start_ind, num_models_per_worker[i]), callback=_add_predict)
        start_ind += num_models_per_worker[i]
    predict_pool.close()
    predict_pool.join()
    # Get predictions, sort and stack
    preds = predict_state["results"]
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
    global predict_state
    # Set global state 
    predict_state["model"] = model
    predict_state["X"] = X
    predict_state["results"] = []
    # Check how many workers we need 
    if not model.n_jobs is None:
        num_workers = max(min(mp.cpu_count(), model.n_jobs), 1)
    else:
        num_workers = 1
    # Intialize the pool with workers
    predict_proba_pool = mp.Pool(num_workers)
    # Add predict tasks to pool
    num_models_per_worker = [len(a) for a in np.array_split(range(len(model.ensemble_)), num_workers)]
    start_ind = 0
    for i in range(num_workers):
        predict_proba_pool.apply_async(_predict_proba, args=(start_ind, num_models_per_worker[i]), callback=_add_predict_proba)
        start_ind += num_models_per_worker[i]
    predict_proba_pool.close()
    predict_proba_pool.join()
    # Get predictions, sort and stack
    probs = predict_state["results"]
    probs.sort(key=lambda x: x[0])
    probs = np.concatenate([p[1] for p in probs], axis=1)
    
    return probs

def get_uncertainty_jsd(P, n_jobs):
    """Represents a general jsd uncertainty calculation process.

    Parameters
    ----------
    P : ndarray, shape (n_samples, n_mc_samples, n_classes) 
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
    global uncertainty_state
    # Set global state 
    uncertainty_state["P"] = P
    uncertainty_state["results"] = []
    # Check how many workers we need 
    if not n_jobs is None:
        num_workers = max(min(mp.cpu_count(), n_jobs), 1)
    else:
        num_workers = 1
    # Intialize the pool with workers
    get_uncertainty_pool = mp.Pool(num_workers)
    # Add uncertainty tasks to pool
    num_samples_per_worker = [len(a) for a in np.array_split(range(P.shape[0]), num_workers)]
    start_ind = 0
    for i in range(num_workers):
        get_uncertainty_pool.apply_async(_get_uncertainty_jsd, args=(start_ind, num_samples_per_worker[i]), callback=_add_get_uncertainty_jsd)
        start_ind += num_samples_per_worker[i]
    get_uncertainty_pool.close()
    get_uncertainty_pool.join()
    # Get uncertainties, sort and stack
    u = uncertainty_state["results"]
    u.sort(key=lambda x: x[0])
    u_a, u_e = np.concatenate([ui[1] for ui in u]), np.concatenate([ui[2] for ui in u])
    
    return u_a, u_e
