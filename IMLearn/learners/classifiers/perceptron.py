from __future__ import annotations
from typing import Callable
from typing import NoReturn
from base import BaseEstimator
from IMLearn.metrics.loss_functions import misclassification_error
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


def add_ones_row(X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    X = np.column_stack((ones, X))
    return X


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    training_loss_: array of floats
        holds the loss value of the algorithm during training.
        training_loss_[i] is the loss value of the i'th training iteration.
        to be filled in `Perceptron.fit` function.

    """

    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        max_iter): int, default = 1000
            Maximum number of passes over training data

        callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by Perceptron. To be set in `Perceptron.fit` function.
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to the given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        self.coefs_ = np.zeros(X.shape[1])
        m = y.shape[0]
        _X = X  # is the X with ones if the intercept is true
        if self.include_intercept_:  # in include intercept then adding a ones row to X, and w.shape = d + 1
            self.coefs_ = np.zeros(X.shape[1] + 1)  # include W0 intercept
            _X = add_ones_row(X)

        # perceptron
        for t in range(self.max_iter_):
            if self._loss(X, y) == 0:  # checks loss over the original X
                return
            for i in range(m):  # breaks after first correction
                if y[i] * _X[i, :].T @ self.coefs_ <= 0:  # y_i * x_i.T @ W : check if prediction is true, if not:
                    self.coefs_ = self.coefs_ + _X[i, :] * y[i]  # corrects the W vector
                    self.callback_(self)
                    break

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            X = add_ones_row(X)
        pred = X @ self.coefs_
        y_pred = np.sign(pred)  # for every sample, project it onto the W vector, and find the sign
        return y_pred

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
