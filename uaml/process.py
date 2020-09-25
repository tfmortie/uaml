import multiprocessing as mp

fit_state = {"X": None, 
        "y": None, 
        "results": []}
predict_state = {"X": None, 
        "ensemble": None, 
        "results": []}

def _add_fit(self, models):
    global fit_state
    fit_state["results"].extend(models)

def _fit(estimator, n_models, n_samples, random_state):
    global fit_state
    models = []
    for _ in range(n_models):
        model = {}
        # Create estimator, given parameters of base estimator and bootstrap sample indices
        model["clf"] = type(estimator)(**estimator.get_params())
        model["ind"] = random_state.randint(0, fit_state["X"].shape[0], size=n_samples)
        model["clf"].fit(fit_state["X"][model["ind"], :], fit_state["y"][model["ind"]])
        models.append(model)

    return models

def _add_predict(self, batch_preds):
    global predict_state
    predict_state["results"].append(batch_preds)

def _predict(self, i, n_models):
    global predict_state
    batch_preds = []
    for m_i in range(i, i+n_models):
        batch_preds.append(predict_state["ensemble"][m_i]["clf"].predict(X).reshape(-1, 1))
    batch_preds = np.hstack(batch_preds)
    
    return batch_preds

def _add_predict_proba(self, batch_probs):
    global predict_state
    predict_state["results"].append(batch_probs)

def _predict_proba(self, i, n_models):
    global predict_state
    batch_probs = []
    for m_i in range(i, i+n_models):
        batch_probs.append(np.expand_dims(predict_state["ensemble"][m_i]["clf"].predict_proba(X), axis=1))
    batch_probs = np.vstack(batch_probs)

    return batch_probs

def fit(estimator, X, y, n_jobs, n_tasks, n_samples, random_state=None):
    """Represents a general fit process.

    Parameters
    ----------
    estimator : scikit-learn base estimator
        Represents the base estimator for the fit task.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The class labels
    n_jobs : int
        Number of cores to use.
    n_tasks : int
        Number of tasks (in this case fits) to be divided among the cores.
    n_samples : int
        Number of samples to consider for each task.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.

    Returns
    -------
    ensemble : list
        Returns a list of fitted base estimators.
    """
    global fit_state
    # Set global state 
    fit_state["X"] = X
    fit_state["y"] = y
    fit_state["results"] = []
    # Check how many workers we need 
    if not n_jobs is None:
        num_workers = max(min(mp.cpu_count(), n_jobs), 1)
    else:
        num_workers = 1
    # Intialize the pool with workers
    fit_pool = mp.Pool(num_workers)
    # Add fit tasks to pool
    num_models_per_worker = [len(a) for a in np.array_split(range(n_tasks), num_workers)]
    for i in range(num_workers):
        fit_pool.apply_async(_fit, args=(estimator, num_models_per_worker[i], n_samples, random_state), callback=_add_fit).get()
    fit_pool.close()
    fit_pool.join()
    ensemble = fit_state["results"]

    return ensemble

def predict(ensemble, X, n_jobs, random_state):
    """Represents a general predict process.

    Parameters
    ----------
    ensemble : list of scikit-learn base estimators
        Represents the ensemble consisting of fitted scikit-learn base estimators.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input samples.
    n_jobs : int
        Number of cores to use.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.

    Returns
    -------
    preds : ndarray
        Returns an array of predicted class labels.
    """
    global predict_state
    # Set global state 
    predict_state["X"] = X
    predict_state["ensemble"] = ensemble
    predict_state["results"] = []
    # Check how many workers we need 
    if not n_jobs is None:
        num_workers = max(min(mp.cpu_count(), n_jobs), 1)
    else:
        num_workers = 1
    # Intialize the pool with workers
    predict_pool = mp.Pool(num_workers)
    # Add predict tasks to pool
    num_models_per_worker = [len(a) for a in np.array_split(range(len(ensemble)), num_workers)]
    start_ind = 0
    for i in range(num_workers):
        predict_pool.apply_async(self._predict, args=(start_ind, num_models_per_worker[i]), callback=self._add_predict).get()
        start_ind += num_models_per_worker[i]
    predict_pool.close()
    predict_pool.join()
    preds = predict_state["results"]
    preds = np.vstack(preds)

    return preds
    
def predict_proba(ensemble, X, n_jobs, random_state):
    """Represents a general predict probabilities process.

    Parameters
    ----------
    ensemble : list of scikit-learn base estimators
        Represents the ensemble consisting of fitted scikit-learn base estimators.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input samples.
    n_jobs : int
        Number of cores to use.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.

    Returns
    -------
    probs : ndarray
        Returns the probability of the sample for each class in the model.
    """
    global predict_state
    # Set global state 
    predict_state["X"] = X
    predict_state["ensemble"] = ensemble
    predict_state["results"] = []
    # Check how many workers we need 
    if not n_jobs is None:
        num_workers = max(min(mp.cpu_count(), n_jobs), 1)
    else:
        num_workers = 1
    # Intialize the pool with workers
    predict_proba_pool = mp.Pool(num_workers)
    # Add predict tasks to pool
    num_models_per_worker = [len(a) for a in np.array_split(range(len(ensemble)), num_workers)]
    start_ind = 0
    for i in range(num_workers):
        predict_proba_pool.apply_async(self._predict_proba, args=(start_ind, num_models_per_worker[i]), callback=self._add_predict_proba).get()
        start_ind += num_models_per_worker[i]
    predict_proba_pool.close()
    predict_proba_pool.join()
    probs = np.concatenate(predict_state["results"], axis=1)
    
    return probs
