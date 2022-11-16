from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics.loss_functions import mean_square_error

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
from IMLearn.utils import utils

original =  ["date", "waterfront", "view", "yr_renovated", "zipcode", "lat", "long"]

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename, index_col=0)
    exclude_cols = ["date", "yr_renovacoted",  "long", "sqft_lot15", "lat","yr_built"]
    df = df.loc[:, ~df.columns.isin(exclude_cols)]  # get rid of certain features
    df = pd.get_dummies(df, columns=['zipcode'])  # add new feature for every different zipcode
    df = df.apply(pd.to_numeric, errors='coerce')  # remove al invalid data
    df = df.dropna()

    df = df[df['price'].values > 100]  # prices must be positive
    df = df[df['bedrooms'].values < 15]  # no more then 15 bedrooms
    df = df[df['condition'].values <= 5]   # condition is discrete between 1 and 5
    df = df[df['condition'].values > 0]

    prices = df["price"]   # dataBase of prices
    df = df.loc[:, ~df.columns.isin(["price"])]  # remove proces column
    return df, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearson = pearson_correlation(X, y)
    i = 0
    for feature in X:
        x_axis = X[feature]
        go.Figure([go.Scatter(x=x_axis.values, y=y, mode='markers', name=r'$\widehat\mu - \mu$')],
        layout=go.Layout(title=f"prices as a function of {feature}, pearson correlation = {pearson[i]}",
        xaxis_title=f"{feature}",
        yaxis_title="prices")).show()
        i += 1


def pearson_correlation(X, y):
    X, Y = X.to_numpy(), y.to_numpy()
    cov_y = []
    for i in range(X.shape[1]):
        col = X[:, i]
        cov = np.cov(col, Y, True, dtype=Y.dtype)
        stddev = np.sqrt(np.diag(cov))
        cov /= stddev[:, None]
        cov /= stddev[None, :]
        cov_y.append(cov)
    return np.round(np.array(cov_y)[:, 1, 0], 3)


def train_subset_fit(df, prices):

    initial_per = 10
    my_nice_linear_model = LinearRegression()
    train_X, train_y, test_X, test_y = utils.split_train_test(df, prices)
    losses = np.zeros((100,))
    stdds = np.zeros((100,))

    for percentage in range(initial_per, 100):
        frac = percentage / 100
        losses_i = np.zeros((10,))
        for i in range(10):
            # generate training indexes
            X_train_sample, Y_train_sample = get_training_sample(frac, train_X, train_y)
            my_nice_linear_model._fit(X_train_sample, Y_train_sample)
            loss = my_nice_linear_model._loss(test_X, test_y)
            losses_i[i] = loss

        mean_loss = np.mean(losses_i)
        std = np.std(losses_i)
        stdds[percentage] = std
        losses[percentage] = mean_loss
    percentage =10 + np.arange(100)
    losses = losses[10:]
    stdds = stdds[10:]
    fig = go.Figure(
        [go.Scatter(x=percentage, y=losses, mode="markers+lines", name="mean loss prediction", line=dict(dash="dash"),
                    marker=dict(color="green", opacity=.7), ),
         go.Scatter(x=percentage, y=losses - (2 * stdds), fill=None, mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=percentage, y=losses + 2 * stdds, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),],
        layout= go.Layout(title=f" house prices as function of percentage of training set used for fit",
                    xaxis_title=f"percetages",
                    yaxis_title="loss") )
    fig.show()


def get_training_sample(frac, train_X, train_y):
    m = train_X.shape[0]
    size_of_train_sample = int(frac * m)
    indexes_shuffled = np.arange(m)
    np.random.shuffle(indexes_shuffled)  # shuffle the indexes
    train_indexes = indexes_shuffled[:size_of_train_sample]  # take only a subset of the indexes
    # train and fit
    train_i_x = train_X[train_indexes]  #
    train_i_y = train_y[train_indexes]
    return train_i_x, train_i_y

def quiz():
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    loss = mean_square_error(y_true, y_pred)
    print(loss)

if __name__ == '__main__':
    np.random.seed(0)
    filename = r"C:\Users\amita\PycharmProjects\pythonProject\IML.HUJI\datasets\house_prices.csv"
    # Question 1 - Load and preprocessing of housing prices dataset
    df, prices = load_data(filename)

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(df, prices)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = utils.split_train_test(df, prices)

    # Question 4 - Fit model over increasing percentages of the overall training data

    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    train_subset_fit(df, prices)

    # quiz()


