# from sklearn.covariance import log_likelihood
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model
    mu = 10  
    var = 1
    sigma = np.sqrt(var)  # the numpy function axcepts sigma as an argument
    m = 1000
    X = np.random.normal(mu, sigma, m)  # a vectoor of random floats normaly distributed
    uni_gaussian = UnivariateGaussian()
    uni_gaussian = uni_gaussian.fit(X)
    est_mu = uni_gaussian.mu_  # mean as caculated in the UnivariateGaussian class over the vectoer X
    est_var = uni_gaussian.var_  # variance as caculated in the UnivariateGaussian class over the vectoer X
    print(est_mu, est_var)

    # Question 2 - Empirically showing sample mean is consistent
    X_axis = np.linspace(10,1000, 100)  
    abs_dis = np.zeros((100,))

    for i in range(1, 100):
        num = i * 10
        samples = X[:num]
        uni_gaussian = UnivariateGaussian()
        uni_gaussian = uni_gaussian.fit(samples)
        est_mu = uni_gaussian.mu_
        distance = np.abs(est_mu - mu)
        abs_dis[i - 1] = distance

    go.Figure([go.Scatter(x=X_axis, y=abs_dis, mode='markers+lines', name=r'$\widehat\mu - \mu$')],
          layout=go.Layout(title=r"$\text{Absolute distance between the estimated - "
                                 r"and true value of the expectation, as a function of the sample size}$",
                  xaxis_title="$\\text{number of samples}$",
                  yaxis_title="r$\hat\|expected mu - mu|$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = uni_gaussian.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdf, mode='markers', name=r'$\widehat\mu - $\mu$')],
          layout=go.Layout(title=r"$\text{Every sample x (taken from ~N(10,1)) and it's pdf(x)}$", 
                  xaxis_title="sample x", 
                  yaxis_title="gaussian pdf(x)",
                  height=300)).show()


def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model

    mu = np.array([0,0,4,0])
    cov = np.array([  [1,   0.2, 0, 0.5],
                      [0.2, 2,   0, 0  ],
                      [0,   0,   1, 0  ],
                      [0.5, 0,   0, 1  ]]) 
    num_of_samples = 1000
    X = np.random.multivariate_normal(mu, cov, num_of_samples)
    multyvar_gauss = MultivariateGaussian()
    multyvar_gauss = multyvar_gauss.fit(X)
    est_mu = multyvar_gauss.mu_
    print("mu = ", est_mu)
    est_cov =  multyvar_gauss.cov_
    print("cov = ", est_cov)

    # Question 5 - Likelihood evaluation
    d = len(mu)
    size = 2000
    limit = 10
    F1 = np.linspace(-limit, limit, size)
    F3 = np.linspace(-limit, limit, size)
    # all the possible combinations of [F1, 0, F2, 0] for every F1, F3
    mu_combinations = np.array(np.meshgrid(F1, 0, F3, 0)).T.reshape(size*size, d)  # a array of shape (size * size, 4) 

    # get the log likelyhood of every differet combination of [F1, 0, F3, 0] (float) and put it in an array of shape (size * size,)
    log_likelyhoods = np.apply_along_axis(MultivariateGaussian.log_likelihood, 1, mu_combinations, cov, X)  
    log_likelyhoods = log_likelyhoods.reshape(size, size)  # create a 2-d array the first 'size' elements are in the 0'th row and so on
    # create a heatmap: F1 as X axis and F3 as Y axis, the color is by the value of log_likelyhoods assosiated with the specific (f3, f1) value
    fig = go.Figure(data = [go.Heatmap(x = F1, y = F3, z = log_likelyhoods, type='heatmap', colorscale = 'Viridis')],
          layout=go.Layout(title=r"Heat Map of log-likelyhoods as a function of f1, f3 : as mu = [f1, 0, f3, 0]", 
                  xaxis_title="f1", 
                  yaxis_title="f3"))
    fig.show()

    # Question 6 - Maximum likelihood
    max_log_likelyhood_value = log_likelyhoods.argmax()
    f1_inx_best, f3_inx_best = np.unravel_index(max_log_likelyhood_value, log_likelyhoods.shape)  # find the index of the max likelyhood
    f1_best = np.round(F1[f1_inx_best], 4)
    f3_best = np.round(F3[f3_inx_best], 4)
    print(f"the best f1 is {f1_best}, and the best f3 is {f3_best} ")
    # print(log_likelyhoods[i,j])


def ex1_quiz():

    X = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    mu_1, sigma_1 = 1, 1
    
    log_likelihood_1 = UnivariateGaussian.log_likelihood(mu_1, sigma_1, X)
    print(log_likelihood_1)

    mu_2, sigma_2 = 10, 1
    log_likelihood_2 = UnivariateGaussian.log_likelihood(mu_2, sigma_2, X)
    print(log_likelihood_2)


if __name__ == '__main__':
    np.random.seed(0)
    # ex1_quiz()
    # test_univariate_gaussian()
    test_multivariate_gaussian()
