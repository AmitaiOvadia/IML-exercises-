from __future__ import annotations
from typing import Tuple, NoReturn

import numpy

from base import BaseEstimator
import numpy as np
# from metrics.loss_functions import misclassification_error
from itertools import product


def weited_misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    mult = y_true * y_pred
    mult[mult > 0] = 0
    misclass_error = float(np.sum(mult))  # sum all the true (wrong predictions) values
    return np.abs(misclass_error)


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm
    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split
    self.j_ : int
        The index of the feature by which to split the data
    self.sign_: int
        The label to predict for samples where the value of the j'th feature is above the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        d = X.shape[1]  # number of features
        best_j, best_thr, best_err, best_sign = -1, -1, np.inf, -1
        for j in range(d):
            thr_j_minus, err_j_minus = self._find_threshold(X[:, j], y, 1)
            thr_j_plus, err_j_plus = self._find_threshold(X[:, j], y, -1)
            errors = {-1: [err_j_minus, thr_j_minus], 1: [err_j_plus, thr_j_plus]}
            if err_j_minus > err_j_plus:
                sign_i = 1
            else:
                sign_i = -1
            if best_err > errors[sign_i][0]:
                best_err = errors[sign_i][0]
                best_thr = errors[sign_i][1]
                best_sign = sign_i
                best_j = j
        self.threshold_, self.j_, self.sign_ = best_thr, best_j, best_sign
        # if self.threshold_ == np.max(X[:, self.j_]):
        #     self.threshold_ = np.inf
        # if self.threshold_ == np.min(X[:, self.j_]):
        #     self.threshold_ = -np.inf

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        X_j = X[:, self.j_]
        m = X_j.shape[0]
        new_labels = np.ones((m,))
        response = new_labels
        if self.sign_ == 1:
            response[X_j < self.threshold_] = -1
        elif self.sign_ == -1:
            response[X_j >= self.threshold_] = -1
        return response

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature
        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for
        labels: ndarray of shape (n_samples,)
            The labels to compare against
        sign: int
            Predicted label assigned to values equal to or above threshold
        Returns
        -------
        thr: float
            Threshold by which to perform split
        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold
        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        m = values.shape[0]  # number of samples
        val_labels = np.vstack((values, labels))  # stack values and labels togather
        val_labels = val_labels[:, val_labels[0].argsort()]  # sort by values
        values = val_labels[0, :]
        labels = val_labels[1, :]
        threshold_error = np.ones(m)  # weited_misclassification_error of every values if threshold went through them
        new_labels = sign * np.ones(m)   # the new labels
        error_val = weited_misclassification_error(labels, new_labels)  # initial error
        for i in range(m):
            new_labels[i] = -sign  # switch the i'th label
            if np.sign(labels[i]) != new_labels[i]:  # the last label was better for this value
                error_val += np.abs(labels[i])  # increase the error
            else:
                error_val -= np.abs(labels[i])  # new label was right and the previous was wrong
            threshold_error[i] = error_val  # record error
        best_threshold_idx = np.argmin(threshold_error)  # the threshold is the where we had the smallest error

        best_threshold, best_split_error = values[best_threshold_idx], threshold_error[best_threshold_idx]

        return best_threshold, best_split_error

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
        return weited_misclassification_error(y, self.predict(X))


if __name__ == '__main__':
    X = np.arange(10).reshape((10, 1))
    y = np.array([1, 1, -1, 1, 1, 1, 1, -1, 1, -1])
    stump = DecisionStump()
    stump.fit(X, y)
    print(stump.j_, stump.sign_, stump.threshold_)
    print(stump.predict(X))
    print(stump.loss(X,y))
