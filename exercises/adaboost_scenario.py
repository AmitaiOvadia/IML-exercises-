import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def best_accuracy_iterations(accuracies_iterations):
    best_accuracy = 0
    iterations = -1
    for accuracy, iters in accuracies_iterations:
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            iterations = iters
    return best_accuracy, iterations


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case

    Ada1 = AdaBoost(DecisionStump, n_learners)
    Ada1.fit(train_X, train_y)
    Training_losses = np.zeros(n_learners)
    Test_losses = np.zeros(n_learners)
    for t in range(1, n_learners + 1):
        Training_losses[t - 1] = Ada1.partial_loss(train_X, train_y, t)
        Test_losses[t - 1] = Ada1.partial_loss(test_X, test_y, t)

    X_axis = np.arange(1, n_learners + 1)
    fig = go.Figure(
        [go.Scatter(x=X_axis, y=Training_losses, mode="markers+lines", name="Train loss",
                    marker=dict(color="green", opacity=.7), ),
         go.Scatter(x=X_axis, y=Test_losses, fill=None, mode="lines", name="Test loss",
                    showlegend=False)],
        layout=go.Layout(title=f"AdaBoost Train loss and Test loss as a function of number of learners ",
                         xaxis_title=f"learnears",
                         yaxis_title="loss"))
    fig.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    model_names = T
    symbols = np.array(["circle", "x"])
    title = "Q2"
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    accuracies_iterations = []

    for i, iterations in enumerate(T):
        def partial_pred(X):
            return Ada1.partial_predict(X, iterations)

        def partial_loss(X, y):
            return Ada1.partial_loss(X, y, iterations)

        # get the accuracy of each model
        loss = partial_loss(test_X, test_y)
        accuracy = 1 - loss
        accuracies_iterations.append([accuracy, iterations])

        fig.add_traces([decision_surface(partial_pred, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"Q2 Decision Boundaries Of Models according to number of iterations", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble

    # find which was the best model

    iterations = 0
    best_accuracy = 1
    for t in range(1, 1 + n_learners):
        accuracy = Ada1.partial_loss(test_X, test_y, t)
        if accuracy < best_accuracy:
            best_accuracy = accuracy
            iterations = t

    def partial_pred(X):
        return Ada1.partial_predict(X, iterations)

    fig = go.Figure(layout=go.Layout(title=rf"Q3"))

    fig.add_traces([decision_surface(partial_pred, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, symbol=symbols[test_y.astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])

    fig.update_layout(title=rf"best model is performing {iterations} iterations. with accuracy {best_accuracy}",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 4: Decision surface with weighted samples

    fig = go.Figure(layout=go.Layout(title=rf"Q4 "))
    D = (Ada1.D_ / np.max(Ada1.D_)) * 5
    # D[D > 3] = 3
    fig.add_traces([decision_surface(partial_pred, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, size=D,
                                           symbol=symbols[train_y.astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))],)

    fig.update_layout(
        title=rf"Decision Boundaries, Markers' Size as an Indicator for the Error Weight",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()





if __name__ == '__main__':
    np.random.seed(0)
    # fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
