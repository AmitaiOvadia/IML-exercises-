from typing import NoReturn
from base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.learners.gaussian_estimators import MultivariateGaussian


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_, self.class_mult_gaussian = None, None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # get all the different classes, assumes that all the classes are present in y
        self.classes_, self.pi_ = np.unique(y, return_counts=True)
        self.pi_ = self.pi_/ X.shape[0]  # normalize the frequencies
        self.class_mult_gaussian = []
        self.mu_ = []
        m = X.shape[0]
        K = self.classes_.shape[0]
        d = X.shape[1]
        _X = np.column_stack((X, y))
        # _X = _X[_X[:, -1].argsort()]
        for k in range(self.classes_.shape[0]):
            X_k = _X[_X[:, -1] == k]  # X_k is a sub matrix of all the samples that belong to class k
            X_k = X_k[:, :-1]  # remove the y column
            # append the multivariate gaussian that is the gaussian of the k'th class samples
            self.class_mult_gaussian.append(MultivariateGaussian().fit(X_k))
            self.mu_.append(self.class_mult_gaussian[k].mu_)  # calculate the mean of every column
        self.mu_ = np.array(self.mu_)
        sum = 0
        for i in range(m):
            k = int(_X[i, -1])  # the class k
            vec = _X[i, :-1] - self.mu_[k, :]
            vec = vec.reshape((d, 1))
            sum += vec @ vec.T
        self.cov_ = 1/(m - K) * sum
        self._cov_inv = np.linalg.inv(self.cov_)


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
        apply_on_each_line = 1
        return np.argmax(self.likelihood(X), axis=apply_on_each_line)


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        M, d = X.shape
        K = self.classes_.shape[0]
        constant = np.sqrt(1/(((2*np.pi)**d) * np.linalg.det(self.cov_)))  # exponent constant
        # likelihoods[i, k] = the probability of the sample X[i] to be classified as k
        likelihoods = np.zeros((M, K))
        for i_m in range(M):
            for k in range(K):
                vec = X[i_m, :] - self.mu_[k, :]
                exp_arg = -0.5 * vec.T @ self._cov_inv @ vec
                exponent = constant * np.exp(exp_arg)
                i_k_likelyhood = exponent * self.pi_[k]
                likelihoods[i_m, k] = i_k_likelyhood
        return likelihoods


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
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
