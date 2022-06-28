"""
Pytest test code.

Author: Thomas Mortier
Date: June 2022
"""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from uaml.multiclass import UAClassifier
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA


def test_basic():
    # first load data and get training and validation sets
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, random_state=2021, stratify=y
    )
    print(X_tr.shape)
    # create base estimator
    est = LogisticRegression(solver="liblinear")
    # create model
    model = UAClassifier(est)
    # start fitting model
    model.fit(X_tr, y_tr)
    # obtain predictions and probabilities on validation sets
    model_preds = model.predict(X_te)
    model_probs = model.predict_proba(X_te)
    # check performance with score function
    print(model.score(X_te, y_te))
    # check performance based on top-1 preds
    print(np.mean(model_preds == y_te))
    # also check performance based on top-1 probs
    print(np.mean(model.classes_[np.argmax(model_probs, axis=1)] == y_te))
    # get uncertainties
    u_a, u_e = model.get_uncertainty(X_te)
    print(u_a)
    print(u_e)


def test_advanced():
    classifiers = {
        "LR": LogisticRegression(solver="liblinear"),
        "5-NN": KNeighborsClassifier(5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Simple Neural Network": MLPClassifier(alpha=1, max_iter=1000),
        "QDA": QuadraticDiscriminantAnalysis(),
    }
    twomoons = True
    if twomoons:
        X, y = make_moons(n_samples=100, noise=0.1, random_state=0)
    else:
        X, y = make_classification(
            n_samples=100,
            n_features=100,
            n_informative=10,
            n_classes=2,
            n_clusters_per_class=1,
            flip_y=0.05,
            class_sep=5,
            hypercube=True,
        )
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # just plot the dataset first
    cm = plt.cm.viridis
    fig, ax = plt.subplots(len(classifiers), 3, figsize=(10, 10))
    for i, clf in enumerate(classifiers.keys()):
        # Plot the training points
        ax[i, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
        ax[i, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
        ax[i, 2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
        # Plot the testing points
        ax[i, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
        ax[i, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
        ax[i, 2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
        ax[i, 0].set_xlim(xx.min(), xx.max())
        ax[i, 0].set_ylim(yy.min(), yy.max())
        ax[i, 1].set_xlim(xx.min(), xx.max())
        ax[i, 1].set_ylim(yy.min(), yy.max())
        ax[i, 2].set_xlim(xx.min(), xx.max())
        ax[i, 2].set_ylim(yy.min(), yy.max())
        ax[i, 0].set_ylabel(clf)
        model = classifiers[clf]
        clf = UAClassifier(model, 500, 0.8, n_jobs=5, verbose=1)
        clf.fit(X_train, y_train)
        Zp = clf.predict(np.c_[xx.ravel(), yy.ravel()], avg=True)
        Za, Ze = clf.get_uncertainty(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Zp = Zp.reshape(xx.shape)
        Za = Za.reshape(xx.shape)
        Ze = Ze.reshape(xx.shape)
        ax[i, 0].contourf(xx, yy, Zp, cmap=cm, alpha=0.8)
        if i == 0:
            ax[i, 0].set_title("Prediction")
        # Plot the training points
        ax[i, 0].scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=cm,
        )
        # Plot the testing points
        ax[i, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
        ax[i, 0].set_xlim(xx.min(), xx.max())
        ax[i, 0].set_ylim(yy.min(), yy.max())
        ax[i, 1].contourf(xx, yy, Za, cmap=cm, alpha=0.8)
        if i == 0:
            ax[i, 1].set_title("Aleatoric uncertainty")
        # Plot the training points
        ax[i, 1].scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=cm,
        )
        # Plot the testing points
        ax[i, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
        ax[i, 1].set_xlim(xx.min(), xx.max())
        ax[i, 1].set_ylim(yy.min(), yy.max())
        ax[i, 2].contourf(xx, yy, Ze, cmap=cm, alpha=0.8)
        if i == 0:
            ax[i, 2].set_title("Epistemic uncertainty")
        # Plot the training points
        ax[i, 2].scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=cm,
        )
        # Plot the testing points
        ax[i, 2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
        ax[i, 2].set_xlim(xx.min(), xx.max())
        ax[i, 2].set_ylim(yy.min(), yy.max())
    plt.savefig("uncertainty.png", bbox_inches="tight")
    plt.close(fig)
