from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
import plotly.express as px

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree
    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate
    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    f_x = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = f_x + eps
    x_df = pd.Series(x)
    y_df_series = pd.Series(y)
    train_X, train_y, test_X, test_y = split_train_test(x_df, y_df_series, 2/3)

    # show data: real vs noisy train and test
    fig = go.Figure(
        [go.Scatter(x=x, y=f_x, mode="markers+lines", name="true values",
                    marker=dict(color="green", opacity=.7), ),

         go.Scatter(x=train_X, y=train_y, fill=None, mode="markers", name="train values",),

         go.Scatter(x=test_X, y=test_y, fill='tonexty', mode="markers", name="test values",), ],

        layout=go.Layout(title=f"True, test and train values for f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps",
                         xaxis_title=f"x values",
                         yaxis_title="y valuse"))
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    degrees = np.arange(0, 10 + 1)
    train_errors = np.zeros(10 + 1)
    validation_errors = np.zeros(10 + 1)

    # for every polynomial degree do cross validation and fill train errors and validation errors
    for deg in degrees:
         train_errors[deg], validation_errors[deg] = cross_validate.cross_validate(PolynomialFitting(k=deg),
                                                                                   train_X,
                                                                                   train_y, mean_square_error)
    # show for each degree of the polynomial the train vs validation errors
    fig = go.Figure(
        [go.Scatter(x=degrees, y=train_errors, mode="markers", name="train errors", ),

         go.Scatter(x=degrees, y=validation_errors, mode="markers", name="validation errors")],

        layout=go.Layout(title=f"Train vs validation errors for every degree of the polynomail",
                         xaxis_title=f"Degrees",  yaxis_title="Errors"))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    # find best polynomial degree and fit model using this degree
    best_degree = np.argmin(validation_errors)
    best_poly_fit = PolynomialFitting(k=best_degree).fit(train_X, train_y)
    test_error = mean_square_error(best_poly_fit.predict(test_X), test_y)
    print(best_degree, test_error)



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions
    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate
    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X_df, y_df = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_df, y_df, train_size=n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    lambdas = np.linspace(0, 3, n_evaluations)

    # arrays of train errors and validation errors for each lambda
    Ridge_train_errors = np.zeros(n_evaluations)
    Ridge_validation_errors = np.zeros(n_evaluations)
    Lasso_train_errors = np.zeros(n_evaluations)
    Lasso_validation_errors = np.zeros(n_evaluations)

    # cross validate models for each lambda and get train and validation errors
    for i, lam in enumerate(lambdas):
        Ridge_train_errors[i], Ridge_validation_errors[i] = cross_validate.cross_validate(RidgeRegression(lam=lam),
                                                                                          X_train,
                                                                                          y_train,
                                                                                          mean_square_error)

        Lasso_train_errors[i], Lasso_validation_errors[i] = cross_validate.cross_validate(Lasso(lam),
                                                                                          X_train,
                                                                                          y_train,
                                                                                          mean_square_error)
    # show Lasso vs Ridge validations and train errors
    fig = go.Figure(
        [go.Scatter(x=lambdas, y=Ridge_train_errors, mode="lines", name="Ridge train errors", ),
        go.Scatter(x=lambdas, y=Ridge_validation_errors, mode="lines", name="Ridge validation errors"),
        go.Scatter(x=lambdas, y=Lasso_train_errors, mode="lines", name="Lasso train errors", ),
        go.Scatter(x=lambdas, y=Lasso_validation_errors, mode="lines", name="Lasso validation errors")],
        layout=go.Layout(title=f"Lasso vs Ridge validations and train errors", xaxis_title=f"lambda", yaxis_title="error"))
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    # get best lambdas
    best_lambda_Ridge = lambdas[np.argmin(Ridge_validation_errors)]
    best_lambda_Lasso = lambdas[np.argmin(Lasso_validation_errors)]

    # fit models according to best lambda
    best_Lasso_model = Lasso(best_lambda_Lasso).fit(X_train, y_train)
    best_Ridge_model = RidgeRegression(best_lambda_Ridge).fit(X_train, y_train)
    best_regretion_model = LinearRegression().fit(X_train, y_train)

    # caculate loss
    loss_Lasso = mean_square_error(best_Lasso_model.predict(X_test), y_test)
    loss_Ridge = mean_square_error(best_Ridge_model.predict(X_test), y_test)
    loss_Regretion = mean_square_error(best_regretion_model.predict(X_test), y_test)

    print("Lasso error: ", loss_Lasso, "Ridge error: ", loss_Ridge,"Regression error: ", loss_Regretion)



if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree(n_samples=100, noise=5)
    # select_polynomial_degree(n_samples=100, noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
