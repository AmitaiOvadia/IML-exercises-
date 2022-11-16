from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet



class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters
        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator
        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.
        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples
        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data
        Returns
        -------
        self : returns an instance of self.
        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        m = len(X)
        self.mu_ = (1/m)*np.sum(X)  # 1/m(the sum of all the elements of X)
        ddf = 0  # degrees of freedom 
        if self.biased_ == True:
            ddf = 1
        self.var_ = 1/(m - ddf)*np.sum(np.power(X - self.mu_, 2))  # 1/(m-1)(the sum of all the elements of (X - mu_)**2)
        self.fitted_ = True
        return self

    

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators
        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for
        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)
        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        exp_arg = -((X-self.mu_)**2)/(2*self.var_)  # np array of the exponent arguments
        constant = np.sqrt(1/(2*np.pi*self.var_))   # the constant that multiplies the exponent
        pdf = constant * np.exp(exp_arg)  # the gaussian pdf 
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        return pdf


    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model
        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with
        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        real_sigma = np.sqrt(sigma)
        m = len(X)  # number of samples 
        log_likelihood = -(m/2)*np.log(2*np.pi*(real_sigma**2)) - (1/(2*(real_sigma**2))) * np.sum( (X - mu)**2 )  # log likelyhood after developing formula  
        return log_likelihood





class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator
        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.
        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False


    @staticmethod
    def return_pdf(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> np.ndarray:
        det_cov = np.linalg.det(cov)  # the determinant of the covariance matrix
        inv_cov = np.linalg.inv(cov)  # the inverse of the covariance matrix
        d = len(mu)
        constant = np.sqrt(1/(((2*np.pi)**d) * det_cov))

        def calculate_each_sample_row(X_row):
            # x_row is one row in the matrix X
            # for each row apply the following functions and return a scalar
            # returning the pdf of each row of X
            vec = X_row - mu
            exp_arg = -0.5 * vec.T @ inv_cov @ vec
            pdf_per_row = constant * np.exp(exp_arg)
            return pdf_per_row

        for_each_row = 1
        pdf = np.apply_along_axis(calculate_each_sample_row, for_each_row, X)
        return pdf

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data
        Returns
        -------
        self : returns an instance of self
        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        for_every_column = 0
        self.mu_ = np.mean(X, axis=for_every_column)  # calculate the mean of every column of the matrix X: (n,) vector
        self.cov_ = np.cov(X.T)  # 2-d matrix of covariance (X.T because each column of original X is a differt feature)
        self.fitted_ = True
        return self


    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for
        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)
        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        pdf = MultivariateGaussian.return_pdf(self.mu_, self.cov_, X)
        return pdf


    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model
        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with
        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        det_cov = np.linalg.det(cov)  # the determinant of the covariance matrix
        inv_cov = np.linalg.inv(cov)  # the inverse of the covariance matrix
        d = len(mu)
        m = X.shape[0]
        constant = (m/2) * np.log(((2*np.pi)**d) * det_cov)
        X = X - mu
        mat = X @ inv_cov * X
        log_likelyhood = - constant - 0.5 * np.sum(mat)   
        return log_likelyhood
