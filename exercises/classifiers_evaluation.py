from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    y = (data[:, -1]).astype(int)
    X = data[:, 0:-1]
    return X, y


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    root_adr = r"C:\Users\amita\PycharmProjects\pythonProject\IML.HUJI\datasets"
    for n, f in [("Linearly Separable", "\linearly_separable.npy"), ("Linearly Inseparable", "\linearly_inseparable.npy")]:
        # Load dataset
        # C:\Users\amita\PycharmProjects\pythonProject\IML.HUJI\datasets
        filename = root_adr + f
        X, y = load_dataset(filename)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def callback(cur_perc: Perceptron):
            losses.append(cur_perc._loss(X, y))

        perc = Perceptron(callback=callback, include_intercept=True)
        perc._fit(X, y)

        # Plot figure

        go.Figure([go.Scatter(x=np.arange(1, len(losses) + 1), y=losses, mode="markers+lines", name=r'$\widehat\mu - \mu$')],
                  layout=go.Layout(title=f"loss as a function of W changes",
                                   xaxis_title=f"change iteration",
                                   yaxis_title="losses")).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    root_adr = r"C:\Users\amita\PycharmProjects\pythonProject\IML.HUJI\datasets"
    for f in ["\gaussian1.npy", "\gaussian2.npy"]:
        # Load dataset
        filename = root_adr + f
        X, Y = load_dataset(filename)

        # Fit models and predict over training set
        lda = LDA().fit(X, Y)
        gnb = GaussianNaiveBayes().fit(X, Y)
        y_pred_LDA = lda.predict(X)
        y_pred_GNB = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        LDA_accuracy = np.round(accuracy(y_pred_LDA, Y), 3)
        GNB_accuracy = np.round(accuracy(y_pred_GNB, Y), 3)

        # plots
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"Plot Of Gaussian Naive Bias classifier, accuracy = {GNB_accuracy}",
                                            f"Plot Of Linear Discriminant Analysis classifier, accuracy = {LDA_accuracy}"))
        feature_1 = X[:, 0]
        feature_2 = X[:, 1]
        # each scatter dot is
        K = lda.classes_.size
        fig.add_trace(go.Scatter(x=feature_1, y=feature_2, mode="markers", showlegend=False,
                                 marker=dict(color=y_pred_GNB, symbol=Y, opacity=.8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=feature_1, y=feature_2, mode="markers", showlegend=False,
                                 marker=dict(color=y_pred_LDA, symbol=Y, opacity=.8)), row=1, col=2)
        fig.update_layout(
            title="Gaussian Naive Bias classifier vs  Linear Discriminant Analysis classifier",
            xaxis_title="feature 1",
            yaxis_title="feature 2",)

        for k in range(K):
            # add ellipse of LDA graph
            LDA_mu_k = lda.mu_[k]  # (a, b) of center of ellipse of the k'th class LDA
            LDA_cov_k = lda.cov_  # the shape of ellipse of the k'th class LDA
            fig.add_trace(get_ellipse(LDA_mu_k, LDA_cov_k), row=1, col=2)

            # add ellipse of GNB graph
            GNB_mu_k = gnb.mu_[k]  # (a, b) of center of ellipse of the k'th class GNB
            # take the two independent variances, and make covariance matrix
            GNB_cov_k = np.diag(gnb.vars_[k])  # the shape of ellipse of the k'th class GNB
            fig.add_trace(get_ellipse(GNB_mu_k, GNB_cov_k), row=1, col=1)

            # add center of LDA ellipse as black 'X'
            a_LDA = LDA_mu_k[0]  # x of center
            b_LDA = LDA_mu_k[1]  # y of center
            fig.add_trace(go.Scatter(x=[a_LDA], y=[b_LDA], mode="markers", showlegend=False,
                                     marker=dict(color="black", symbol="x", opacity=.8, size=12)), row=1, col=2)

            # add center of GNB ellipse as black 'X'
            a_GNB = GNB_mu_k[0]  # x of center
            b_GNB = GNB_mu_k[1]  # y of center
            fig.add_trace(go.Scatter(x=[a_GNB], y=[b_GNB], mode="markers", showlegend=False,
                                     marker=dict(color="black", symbol="x", opacity=.8, size=12)), row=1, col=1)
        fig.show()

def quiz():
    S1 = np.array([[0, 0], [1, 0], [2, 1], [3, 1], [4, 1], [5, 1], [6, 2], [7, 2]])
    X = S1[:, 0]
    Y = S1[:, 1]
    gnb1 = GaussianNaiveBayes().fit(X, Y)
    # Q1
    print("Q1")
    print("pi = ", gnb1.pi_)
    print("mu = \n", gnb1.mu_)

    # Q2
    print("\nQ2")
    X = np.array([[1,1],[1,2],[2,3],[2,4],[3,3],[3,4]])
    Y = np.array([0,0,1,1,1,1])
    gnb2 = GaussianNaiveBayes().fit(X, Y)
    print("vars = \n", gnb2.vars_)

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    # quiz()