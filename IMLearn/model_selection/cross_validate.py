from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator
    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data
    X: ndarray of shape (n_samples, n_features)
       Input data to fit
    y: ndarray of shape (n_samples, )
       Responses of input data to fit to
    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.
    cv: int
        Specify the number of folds.
    Returns
    -------
    train_score: float
        Average train score over folds
    validation_score: float
        Average validation score over folds
    """

    _X = np.column_stack((X, y))  # combine X and y
    S = np.array_split(_X, cv)
    validation_score = 0
    train_score = 0
    for i in range(cv):
        X_i_test, X_i_train, y_i_test, y_i_train = split_Si(S, i)
        estimator.fit(X_i_train, y_i_train)
        validation_score += scoring(estimator.predict(X_i_test), y_i_test)
        train_score += scoring(estimator.predict(X_i_train), y_i_train)
    return train_score/cv, validation_score/cv


def split_Si(S, i):
    """
    split S to train and test
    :param S: a list of split data
    :param i: the index of the test data
    :return:
    """
    S_i_test = S[i]
    S_i_train = list(S)
    del S_i_train[i]
    S_i_train = np.concatenate(S_i_train, axis=0)
    X_i_test = S_i_test[:, :-1]
    y_i_test = S_i_test[:, -1]
    X_i_train = S_i_train[:, :-1]
    y_i_train = S_i_train[:, -1]
    if X_i_train.shape[1] == 1:
        return X_i_test.reshape((X_i_test.shape[0],)), X_i_train.reshape((X_i_train.shape[0],)), y_i_test, y_i_train
    return X_i_test, X_i_train, y_i_test, y_i_train
