from typing import Tuple

import numpy as np
import pandas as pd

def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    m = X.shape[0]
    indexes_shuffled = np.arange(m)  # array of all rows' indexes
    np.random.shuffle(indexes_shuffled)  # shuffle indexes
    train_number = int(m * train_proportion)  # get number of train samples
    train_indexes = indexes_shuffled[:train_number]  # get train rows' indexes
    test_indexes = indexes_shuffled[train_number:]  # get test rows' indexes

    train_X = X.iloc[train_indexes]  # sample only train indexes
    train_y = y.iloc[train_indexes]  # sample only train indexes

    test_X = X.iloc[test_indexes]  # sample only test indexes
    test_y = y.iloc[test_indexes]  # sample only test indexes

    return train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
