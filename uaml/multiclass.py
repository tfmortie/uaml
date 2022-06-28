"""
Code for uncertainty-aware classifiers.

Author: Thomas Mortier
Date: June 2022
"""
import time
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
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
    ensemble_size : int, default=10
        Number of Monte Carlo samples or in other words the size of ensemble.
    train_ratio : float, default=0.5
        Percentage of training samples used for each bootstrap.
    n_jobs : int, default=None
        The number of jobs to run in parallel. Currently this applies to fit, predict and predict_proba.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the random permutations generator.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    estimator : scikit-learn base estimator
        Represents the base estimator for the classification task.
    ensemble_size : int, default=10
        Number of Monte Carlo samples or in other words the size of ensemble.
    train_ratio : float, default=0.5
        Percentage of training samples used for each bootstrap.
    n_jobs : int, default=None
        The number of jobs to run in parallel. Currently this applies to fit, predict and predict_proba.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the random permutations generator.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
    classes_ : ndarray of shape (n_classes,)
        A list of class labels known to the classifier.
    X_ : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training input samples seen during fit.
    y_ : array-like, shape (n_samples,)
        The class labels seen during fit.
    b_size_ : int
        Size of bootstrap samples.
    ensemble_ : list, size self.ensemble_size
        List which represents the fitted ensemble.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from uaml.classifier import UAClassifier
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> clf = UAClassifier(LogisticRegression())
    >>> clf.fit(X,y)
    """

    def __init__(
        self,
        estimator,
        ensemble_size=10,
        train_ratio=0.5,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.estimator = estimator
        self.train_ratio = train_ratio
        self.ensemble_size = ensemble_size
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
        self.random_state = check_random_state(self.random_state)
        # check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=False)  # multi-output not supported (yet)
        # check whether base estimator supports probabilities
        if not hasattr(self.estimator, "predict_proba"):
            raise NotFittedError(
                "{0} does not support probabilistic predictions.".format(self.estimator)
            )
        # check if train_ratio is float
        if not isinstance(self.train_ratio, float):
            raise TypeError("Parameter train_ratio must be of type float.")
        # check if ensemble_size is integer
        if not isinstance(self.ensemble_size, int):
            raise TypeError("Parameter ensemble_size must be of type int.")
        # check if n_jobs is integer
        if not self.n_jobs is None:
            if not isinstance(self.n_jobs, int):
                raise TypeError("Parameter n_jobs must be of type int.")
        # store the classes and complete data seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # now initialize and fit the ensemble
        self.b_size_ = int(X.shape[0] * self.train_ratio)
        start_time = time.time()
        self.ensemble_ = fit(self)
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("UAClassifier", "fitting", stop_time - start_time))

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
        preds : ndarray, shape (n_samples, self.ensemble_size) or (n_samples,) in case of avg=True
            Returns an array of predicted class labels.
        """
        # check input
        X = check_array(X)
        start_time = time.time()
        try:
            preds = predict(self, X)
        except NotFittedError as e:
            print(
                "Error {}, this model is not fitted yet. Cal 'fit' with appropriate arguments before using this method.".format(
                    e
                )
            )
        stop_time = time.time()
        if self.verbose >= 1:
            print(
                _message_with_time("UAClassifier", "predicting", stop_time - start_time)
            )
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
        preds : ndarray, shape (n_samples, self.ensemble_size, n_classes) or (n_samples,) in case of avg=True
            Returns the probability of the sample for each class in the model, where classes are ordered by self.classes_.
        """
        # check input
        X = check_array(X)
        start_time = time.time()
        try:
            probs = predict_proba(self, X)
        except NotFittedError as e:
            print(
                "Error {}, this model is not fitted yet. Cal 'fit' with appropriate arguments before using this method.".format(
                    e
                )
            )
        stop_time = time.time()
        if self.verbose >= 1:
            print(
                _message_with_time(
                    "UAClassifier", "predicting probabilities", stop_time - start_time
                )
            )
        if avg:
            probs = np.mean(probs, axis=1)

        return probs

    def get_uncertainty(self, X):
        """Return uncertainty estimates.

        Calculate estimates for aleatoric and epistemic uncertainty based on Jensen-Shannon divergence.

        See paper about Aleatoric and Epistemic uncertainty in Machine Learning: An Introduction to Concepts and Methods (https://arxiv.org/abs/1910.09457) and https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

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
        # check input
        X = check_array(X)
        start_time = time.time()
        try:
            # obtain probabilities
            P = self.predict_proba(X, avg=False)
        except NotFittedError as e:
            print(
                "Error {}, this model is not fitted yet. Cal 'fit' with appropriate arguments before using this method.".format(
                    e
                )
            )
        u_a, u_e = get_uncertainty_jsd(P, self.n_jobs)
        stop_time = time.time()
        if self.verbose >= 1:
            print(
                _message_with_time(
                    "UAClassifier", "calculating uncertainty", stop_time - start_time
                )
            )

        return u_a, u_e

    def score(self, X, y, normalize=True, sample_weight=None):
        """Return mean accuracy score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        normalize : bool, optional (default=True)
            If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        try:
            preds = predict(self, X)
        except NotFittedError as e:
            print(
                "Error {}, this model is not fitted yet. Cal 'fit' with appropriate arguments before using this method.".format(
                    e
                )
            )
        stop_time = time.time()
        if self.verbose >= 1:
            print(
                _message_with_time(
                    "UAClassifier", "calculating score", stop_time - start_time
                )
            )
        preds = np.apply_along_axis(get_most_common_el, 1, preds)
        score = accuracy_score(
            y, preds, normalize=normalize, sample_weight=sample_weight
        )

        return score
