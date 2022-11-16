from typing import NoReturn
from base import BaseEstimator
import numpy as np
from IMLearn.learners.gaussian_estimators import MultivariateGaussian

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_,  self.class_mult_gaussian = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # get all the different classes, assumes that all the classes are present in y
        self.classes_, self.pi_ = np.unique(y, return_counts=True)
        self.pi_ = self.pi_ / X.shape[0]
        self.class_mult_gaussian = []
        self.mu_ = []
        K = self.classes_.shape[0]
        if len(X.shape) == 1:
            d = 1
        else:
            d = X.shape[1]
        # self.vars_ = []
        self.vars_ = np.zeros((K, d))
        _X = np.column_stack((X, y))
        for k in range(self.classes_.shape[0]):
            X_k = _X[_X[:, -1] == k]  # X_k is a sub matrix of all the samples that belong to class k
            X_k = X_k[:, :-1]  # remove the y column
            # append the multivariate gaussian that is the gaussian of the k'th class samples
            gauss_k = MultivariateGaussian().fit(X_k)
            self.class_mult_gaussian.append(gauss_k)
            self.vars_[k, :] = np.var(X_k, axis=0)
            self.mu_.append(self.class_mult_gaussian[k].mu_)  # calculate the mean of every column
        self.mu_, self.vars_ = np.array(self.mu_), np.array(self.vars_)


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
        likelihoods = np.zeros((M, K))
        for k in range(K):
            var_k = self.vars_[k, :]  # (d,) of feature's variances that are classified as k
            const = -0.5 * np.log(2 * np.pi * var_k)  # (d,) the log of the constant inside the exponent
            # log(exp((X - mu)**2/2*sigma**2)) element wise divide (M, d) / (d,)
            exp_arg = - np.power(X - self.mu_[k, :], 2) / (2 * var_k)  # (M ,d)
            log_of_exp = const + exp_arg    # (M, d) the log of the exp(const * exp_arg)
            sum_for_all_features = 1
            k_likelihood_col = np.sum(log_of_exp, axis=sum_for_all_features) + np.log(self.pi_[k])
            likelihoods[:, k] = k_likelihood_col  # the k'th column
        #  likelihoods[i, k] = the probability of the sample X[i] to be classified as k
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
        return misclassification_error(y_pred, y)
