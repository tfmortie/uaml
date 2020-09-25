# Uncertainty-Aware Machine Learning

**uaml** is a Python module for easy yet highly effective uncertainty-aware machine learning. Currently, it is built on top of scikit-learn and supports all probabilistic base classifiers. More precisely, the following models are currently implemented:

* Uncertainty estimation based on probabilistic classification ensembles
* MORE TO COME

## Installation 

### Dependencies 

Following packages are required:

* numpy 
* scikit-learn

### User installation

TODO

## Basic usage

```python
import numpy as np

from sklearn.svm import svc
from uaml.classifier import UAClassifier

# Some example data
X_train, y_train = np.random.randn(1000,100), np.random.randint(0,5,1000)
X_test = np.random.randn(100,100)

# Use SVC as base (probabilistic) estimator
estm = SVC(gamma=2, C=1, probability=True) 

# Constuct and fit an uncertainty-aware classifier with 100 estimators and parellilize over 5 cores 
clf = UAClassifier(estm, mc_sample_size=0.5, n_mc_samples=500, n_jobs=5)
clf.fit(X_train, y_train)

# Obtain predictions by means of majority voting and get corresponding uncertainties
yhat = clf.predict(X_test, avg=True)
ua, ue = clf.get_uncertainty(X_test)
```
