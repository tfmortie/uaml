# Uncertainty-aware machine learning [![build](https://github.com/tfmortie/uaml/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/tfmortie/uaml/actions/workflows/build.yml) [![pypi version](https://badge.fury.io/py/uaml.svg)](https://pypi.org/project/uaml/) [![license](https://img.shields.io/github/license/tfmortie/uaml)](https://github.com/tfmortie/uaml/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/uaml)](https://pepy.tech/project/uaml)

Python package for uncertainty-aware classification built on top of Scikit-learn. 

## Description

**uaml** is a Python package for uncertainty-aware machine learning based on probabilistic ensembles and the Jensen–Shannon divergence. Currently, it is built on top of Scikit-learn and supports all probabilistic base classifiers. 

## Installation

Clone this repository [`tfmortie/uaml`](https://github.com/tfmortie/uaml.git) and run `pip install . -r requirements.txt`
or install by means of `pip install uaml`.

## Example

The uncertainty-aware classifier is provided through `uaml.multiclass.UAClassifier`. Below we show a minimal working and more elaborate example.

### Basic usage

We start by importing some packages that we will need throughout the example:

```python
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# some example data
X, y = make_moons(n_samples=100, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
```

Creating an uncertainty-aware classifier, with `LogisticRegression` as underlying probabilistic model, is done as follows:

```python
from uaml.multiclass import UAClassifier

# use LogisticRegression as base (probabilistic) estimator
est = LogisticRegression(solver="liblinear")

# construct and fit an uncertainty-aware classifier with 500 estimators and parallelize over 5 cores 
clf = UAClassifier(est, ensemble_size=500, train_ratio=0.5, n_jobs=5)
```

`UAClassifier` follows the Scikit-learn API, as illustrated below: 

```python
# fit our classifier
clf.fit(X_train, y_train)

# obtain predictions by means of majority voting
preds = clf.predict(X_test, avg=True)

# obtain probabilities
probs = clf.predict_proba(X_test, avg=True) 
```
Finally, let's calculate aleatoric and epistemic uncertainty:

```python
ua, ue = clf.get_uncertainty(X_test)
```

### Visualisation

In a next example, let's see how aleatoric and epistemic uncertainty evaluate in the feature space of the "two moons" dataset for different classifiers:

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uaml.multiclass import UAClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

# different estimators for UAClassifier
classifiers = {
    "5-NN": KNeighborsClassifier(5),
    "Linear SVM": SVC(kernel="linear", C=0.025, probability=True),
    "RBF SVM": SVC(gamma=1, C=1, probability=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Simple Neural Network" : MLPClassifier(alpha=1, max_iter=1000),
    "QDA": QuadraticDiscriminantAnalysis()
}

# create dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=0)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# create plot
cm = plt.cm.viridis
fig,ax = plt.subplots(len(classifiers), 3, figsize=(10,10))
for i, clf in enumerate(classifiers.keys()):
    # fit classifiers and obtain predictions and uncertainty estimates
    model = classifiers[clf]
    clf = UAClassifier(model, 500, 0.8, n_jobs=5, verbose=1)
    clf.fit(X_train, y_train)
    Zp = clf.predict(np.c_[xx.ravel(), yy.ravel()], avg=True)
    Za, Ze = clf.get_uncertainty(np.c_[xx.ravel(), yy.ravel()])

    # construct contour plot
    Zp = Zp.reshape(xx.shape)
    Za = Za.reshape(xx.shape)
    Ze = Ze.reshape(xx.shape)
    ax[i,0].contourf(xx, yy, Zp, cmap=cm, alpha=.8)
    if i == 0:
        ax[i, 0].set_title("Prediction")

    # prediction plot
    # plot the training points
    ax[i,0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
    # plot the testing points
    ax[i,0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
    ax[i,0].set_xlim(xx.min(), xx.max())
    ax[i,0].set_ylim(yy.min(), yy.max())

    # aleatoric uncertainty plot
    ax[i,1].contourf(xx, yy, Za, cmap=cm, alpha=.8)
    if i == 0:
        ax[i, 1].set_title("Aleatoric uncertainty")
    # plot the training points
    ax[i,1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
    # plot the testing points
    ax[i,1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
    ax[i,1].set_xlim(xx.min(), xx.max())
    ax[i,1].set_ylim(yy.min(), yy.max())

    # epistemic uncertainty plot
    ax[i,2].contourf(xx, yy, Ze, cmap=cm, alpha=.8)
    if i == 0:
        ax[i, 2].set_title("Epistemic uncertainty")
    # plot the training points
    ax[i,2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
    # plot the testing points
    ax[i,2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
    ax[i,2].set_xlim(xx.min(), xx.max())
    ax[i,2].set_ylim(yy.min(), yy.max())
```

![Aleatoric and epistemic uncertainty in classification](uncertainty.png "Aleatoric and epistemic uncertainty")

## References

* _Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods, Hüllermeier et al., Machine learning (2021)_
