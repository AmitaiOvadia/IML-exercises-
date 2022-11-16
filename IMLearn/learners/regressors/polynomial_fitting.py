from __future__ import annotations
from typing import NoReturn
from . import LinearRegression
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import mean_square_error


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self.linear_regression = LinearRegression(include_intercept=False)
        self.k = k

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # X is a van der munde matrix of degree k
        self.linear_regression._fit(self.__transform(X), y)  # fit X, y. now polinomial_regression holds a W vector


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
        Y_pred = self.linear_regression._predict(self.__transform(X))  # will perform X @ W = y_predicted (W was calculated by fit)
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
        loss = mean_square_error(y, Y_pred)
        return loss

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        vander_X = np.vander(X, N=self.k + 1, increasing=True)  # creates a data X that fits k the polynomial degree
        return vander_X
