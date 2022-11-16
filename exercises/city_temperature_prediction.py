import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from IMLearn.utils import utils
import plotly.graph_objects as go
from datetime import datetime, date

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=["Date"])
    df = df[df['Temp'].values > -15]  # cant be colder then -15
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


def plot_temperatures(df):
    # Plot a scatter plot showing this relation, and color code the dots by the different years
    df_israel = df.loc[df["City"] == "Tel Aviv"]
    DayOfYear = "DayOfYear"
    marker_size = 3
    color_by = "Year"
    temp = "Temp"
    fig1 = px.scatter(x=df_israel[DayOfYear], y=df_israel[temp], color=df_israel[color_by].astype(str),
                      title="temperatures as a function of day of year")
    fig1.update_traces(marker={'size': marker_size})
    fig1.show()
    #  graph of the stds of each month's temperatures
    df_israel_gb_months = df_israel.groupby("Month")
    temps = df_israel_gb_months[temp]
    temps_std = temps.std().rename("standard deviation")
    fig2 = px.bar(temps_std, title="std of each month")
    fig2.show()


def q_3_temps_by_countries(df):
    # 3. Mean temperatures as a function of month in different countries.
    df_groupby_C_M = df.groupby(["City", "Month"])
    df_groupby_C_M = df_groupby_C_M.agg({"Temp": ["mean", "std"]}).reset_index()
    df_groupby_C_M.columns = ["City", "Month", "mean", "std"]
    fig_temps = px.line(df_groupby_C_M, x="Month", y="mean", color=df_groupby_C_M["City"].astype(str),
                        title="mean temperatures of different countries", error_y="std").show()


def find_best_polynome_fit(df):
    # for a model fit of temperatures in tel aviv vs day of year
    # loss as a function of k, the polynomial degree
    df_israel = df.loc[df["City"] == "Tel Aviv"]
    train_X, train_y, test_X, test_y = utils.split_train_test(df_israel["DayOfYear"], df_israel["Temp"])
    losses = np.zeros((10,))
    for k in range(1, 11):
        poly_k = PolynomialFitting(k)
        poly_k.fit(train_X, train_y)
        loss_k = poly_k.loss(test_X, test_y)
        losses[k - 1] = loss_k
    fig = px.bar(x=np.arange(1, 11), y=losses, title="for a model fit of temperatures in tel aviv vs day of year:\n "
                                                     "loss as a function of k, the polynomial degree")
    fig.show()


def fit_other_counties_israel_train(df):
    # 5. loss a a funtion of the different country data set, after fitted by a model trained on Israel data
    df_israel = df.loc[df["City"] == "Tel Aviv"]
    train_x, train_y = df_israel["DayOfYear"].to_numpy(), df_israel["Temp"].to_numpy()
    k = 5
    israel_fit = PolynomialFitting(k)
    israel_fit.fit(train_x, train_y)
    df_netherlands = df.loc[df["City"] == "Amsterdam"]
    df_jordan = df.loc[df["City"] == "Amman"]
    df_south_africa = df.loc[df["City"] == "Capetown"]
    temps_by_countries = [df_netherlands[["Temp", "DayOfYear"]], df_jordan[["Temp", "DayOfYear"]],
                          df_south_africa[["Temp", "DayOfYear"]]]

    losses = []
    for country in temps_by_countries:
        loss = israel_fit.loss(country["DayOfYear"].to_numpy(), country["Temp"].to_numpy())
        losses.append(loss)
    fig = px.bar(x=["Netherlands", "Jordan", "South Africa"], y=losses,
                 title="loss a a function country, fit by israel data")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    address = r"C:\Users\amita\PycharmProjects\pythonProject\IML.HUJI\datasets\City_Temperature.csv"
    df = load_data(address)

    # Question 2
    plot_temperatures(df)

    # Question 3 - Exploring differences between countries
    q_3_temps_by_countries(df)

    # Question 4 - Fitting model for different values of `k`
    find_best_polynome_fit(df)

    # Question 5 - Evaluating fitted model on different countries
    fit_other_counties_israel_train(df)
