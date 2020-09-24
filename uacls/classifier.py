import time

import numpy as np

from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils import _message_with_time
from sklearn.exceptions import NotFittedError, FitFailedWarning

import uacls.utils as u

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
    verbose : integer, default=0
        Controls the verbosity: the higher, the more messages
    random_state : RandomState or an int seed (None by default)
        A random number generator instance to define the state of the
        random permutations generator.

    Examples
    --------
    >>> from uacls import UAClassifier
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> clf = UAClassifier()
    >>> clf.fit(X,y)
    """

    def __init__(self, estimator, mc_sample_size, n_mc_samples=10, verbose=0, random_state=None):
        self.estimator = estimator
        self.mc_sample_size = mc_sample_size
        self.n_mc_samples = n_mc_samples
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """Implementation of the fitting function for the uncertainty-aware classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The class labels

        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_state(self.random_state)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
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
        # Now initialize and fit the ensemble
        self.ensemble = []
        self.n_samples_ = int(X.shape[0]*self.mc_sample_size)
        start_time = time.time()
        for _ in range(self.n_mc_samples):
            model = {}
            # Create estimator, given parameters of base estimator and bootstrap sample indices
            model["clf"] = type(self.estimator)(**self.estimator.get_params())
            model["ind"] = self.random_state_.randint(0, X.shape[0], size=self.n_samples_)
            model["clf"].fit(X[model["ind"], :], y[model["ind"]])
            self.ensemble.append(model)
        stop_time = time.time()
        if self.verbose >= 1:
            _message_with_time("UAClassifier", "fitting model", stop_time-start_time)
        # Store the classes and complete data seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

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
        y : ndarray
            Returns an array of predicted class labels.
        """
        # Check input
        X = check_array(X)
        # Obtain predictions
        preds = []
        for model in self.ensemble:
            try:
                # Append predictions to prediction list
                preds.append(model["clf"].predict(X).reshape(-1, 1))
            except NotFittedError as e:
                print("This {0} instance is not fitted yet. Cal 'fit' \
                        with appropriate arguments before using this \
                        method.".format(type(model["clf"])))
        preds = np.hstack(preds)
        if avg:
            return np.apply_along_axis(u.get_most_common_el, 1, preds)
        return preds

    def predict_proba(self, X, avg=True):
        """Return probability estimates.

        Important: the returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        avg : boolean, default=True
            Return model average when true, and array of probability estimates otherwise.

        Returns
        -------
        P : ndarray
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        # Check input
        X = check_array(X)
        # Obtain predictions
        probs = []
        for model in self.ensemble:
            try:
                # Append predictions to prediction list
                probs.append(np.expand_dims(model["clf"].predict_proba(X), axis=1))
            except NotFittedError as e:
                print("This {0} instance is not fitted yet. Cal 'fit' with \
                        appropriate arguments before using this method.".format(type(model["clf"])))
        probs = np.concatenate(probs, axis=1)
        if avg:
            return np.mean(probs, axis=1)
        else:
            return probs

    def get_uncertainty(self, X):
        """Return uncertainty estimates.

        Calculate estimates for epistemic and aleatoric uncertainty
        based on Jensen-Shannon divergence.

        See paper about Aleatoric and Epistemic uncertainty in Machine Learning:
        An Introduction to Concepts and Methods (https://arxiv.org/abs/1910.09457)
        and https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        u_a : ndarray, shape (n_samples,)
            Array of aleatoric uncertainty estimates for each sample.
        u_e : ndarray, shape (n_samples,)
            Array of epistemic uncertainty estimates for each sample.
        """
        # Check input
        X = check_array(X)
        # Obtain probabilities
        P = self.predict_proba(X, avg=False)
        return u.calculate_uncertainty_jsd(P)
