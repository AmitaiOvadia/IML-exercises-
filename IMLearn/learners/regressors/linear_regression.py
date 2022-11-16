from __future__ import annotations
import re
import sys
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import mean_square_error
from numpy.linalg import pinv

def add_ones_row(X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    X = np.column_stack((ones, X))
    return X


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            X = add_ones_row(X)
        X_dager = np.linalg.pinv(X)  # pseodo inverse using SVD
        W = X_dager @ y  # page 33 in course book, how to solve the least-squares problem
        self.coefs_ = W  # set the coeffiwions to be the result

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
        Y_pred = X @ self.coefs_  # Xw = y
        return Y_pred

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        Y_pred = self._predict(X)  # predict the new Y responces using the W vector that was trained
        loss = mean_square_error(y, Y_pred)  # check the new Y_responces vs the true Y values   
        return loss
