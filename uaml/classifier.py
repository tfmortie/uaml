"""
Code for uncertainty-aware classifiers

Author: Thomas Mortier
Date: June 2022
"""
import time
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels, is_multilabel
from sklearn.utils import _message_with_time
from sklearn.utils.validation import check_X_y, check_array, check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from .process import fit, predict, predict_proba, get_uncertainty_jsd
from .utils import get_most_common_el

class UAClassifier(BaseEstimator, ClassifierMixin):
    """Generic uncertainty-aware classifier.

    Parameters
    ----------
    estimator : scikit-learn base estimator
        Represents the base estimator for the classification task.
    mc_sample_size : float, default=0.5
        Percentage of training samples used for each bootstrap.
    n_mc_samples : int, default=10
        Number of Monte Carlo samples or in other words the size of ensemble.
    n_jobs : int, default=None
        The number of jobs to run in parallel. Currently this applies to fit, predict and predict_proba.  
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the random permutations generator.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
        
    Attributes
    ----------
    TODO

    Examples
    --------
    >>> from uaml import UAClassifier
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> clf = UAClassifier()
    >>> clf.fit(X,y)
    """

    def __init__(self, estimator, mc_sample_size, n_mc_samples=10, n_jobs=None, random_state=None, verbose=0):
        self.estimator = estimator
        self.mc_sample_size = mc_sample_size
        self.n_mc_samples = n_mc_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Implementation of the fitting function for the uncertainty-aware classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_state(self.random_state)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True) # multi-output not supported (yet) 
        # Check whether base estimator supports probabilities
        if not hasattr(self.estimator, 'predict_proba'):
            raise NotFittedError("{0} does not support \
                    probabilistic predictions.".format(self.estimator))
        # Check if mc_sample_size is float
        if not isinstance(self.mc_sample_size, float):
            raise TypeError("Parameter mc_sample_size must be of type float.")
        # Check if n_mc_samples is integer
        if not isinstance(self.n_mc_samples, int):
            raise TypeError("Parameter n_mc_samples must be of type int.")
        # Check if n_jobs is integer
        if not self.n_jobs is None:
            if not isinstance(self.n_jobs, int):
                raise TypeError("Parameter n_jobs must be of type int.")
        # Store the number of outputs, classes for each output and complete data seen during fit
        if is_multilabel(y):
            self.n_outputs_ = y.shape[1]
        else:
            self.n_outputs_ = 1
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Now initialize and fit the ensemble
        self.n_samples_ = int(X.shape[0]*self.mc_sample_size)
        start_time = time.time()
        self.ensemble_ = fit(self)
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("UAClassifier", "fitting", stop_time-start_time))

        return self

    def predict(self, X, avg=True):
        """Return class predictions.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        avg : boolean, default=True
            Return model average when true, and array of predictions otherwise.

        Returns
        -------
        TODO: check shapes for MC and MLC
        preds : ndarray, shape (n_samples, self.n_mc_samples) or (n_samples,) in case of avg=True
            Returns an array of predicted class labels.
        """
        # Check input
        X = check_array(X)
        start_time = time.time()
        try:
            preds = predict(self, X)
        except NotFittedError as e:
            print("Error {}, this model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.".format(e))
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("UAClassifier", "predicting", stop_time-start_time))
        if avg:
            preds = np.apply_along_axis(get_most_common_el, 1, preds)

        return preds

    def predict_proba(self, X, avg=True):
        """Return probability estimates.

        Important: the returned estimates for all classes are ordered by self.classes_.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        avg : boolean, default=True
            Return model average when true, and array of probability estimates otherwise.

        Returns
        -------
        TODO: check shapes for MC and MLC
        preds : ndarray, shape (n_samples, self.n_mc_samples, n_classes) or (n_samples,) in case of avg=True
            Returns the probability of the sample for each class in the model, where classes are ordered by self.classes_.
        """
        # Check input
        X = check_array(X)
        start_time = time.time()
        try:
            probs = predict_proba(self, X)
        except NotFittedError as e:
            print("Error {}, this model is not fitted yet. Cal 'fit' with \
                    appropriate arguments before using this method.".format(e))
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("UAClassifier", "predicting probabilities", stop_time-start_time))
        if avg:
            probs = np.mean(probs, axis=1)
        
        return probs

    def get_uncertainty(self, X):
        """Return uncertainty estimates.

        Calculate estimates for epistemic and aleatoric uncertainty based on Jensen-Shannon divergence.

        See paper about Aleatoric and Epistemic uncertainty in Machine Learning: An Introduction to Concepts and Methods (https://arxiv.org/abs/1910.09457) and https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        u_a : ndarray, shape (n_samples, n_outputs)
            Array of aleatoric uncertainty estimates for each sample and output.
        u_e : ndarray, shape (n_samples, n_outputs)
            Array of epistemic uncertainty estimates for each sample and output.
        """
        # Check input
        X = check_array(X)
        start_time = time.time()
        try:
            # Obtain probabilities
            P = self.predict_proba(X, avg=False)
        except NotFittedError as e:
            print("Error {}, this model is not fitted yet. Cal 'fit' with \
                    appropriate arguments before using this method.".format(e))
        if self.n_outputs_ > 1:
            u_a, u_e = [], []
            for di in range(P.shape[2]):
                u_a_di, u_e_di = get_uncertainty_jsd(P[:,:,di,:], self.n_jobs)
                u_a.append(u_a_di.reshape(-1,1))
                u_e.append(u_e_di.reshape(-1,1))
            u_a, u_e = np.hstack(u_a), np.hstack(u_e)
        else:
            u_a, u_e = get_uncertainty_jsd(P, self.n_jobs)
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("UAClassifier", "calculating uncertainty", stop_time-start_time))
        
        return u_a, u_e 

    def score(self, X, y, normalize=True, sample_weight=None):
        """Return mean accuracy score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        TODO: check if this works
        normalize : bool, optional (default=True)
            If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
       
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True)
        start_time = time.time()
        try:
            preds = predict(self, X)
        except NotFittedError as e:
            print("Error {}, this model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.".format(e))
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("UAClassifier", "calculating score", stop_time-start_time))
        preds = np.apply_along_axis(get_most_common_el, 1, preds)
        score = accuracy_score(y, preds, normalize=normalize, sample_weight=sample_weight) 

        return score
