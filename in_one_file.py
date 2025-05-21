import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools
from scipy import linalg
from sklearn import mixture
from scipy.stats import norm
from scipy.stats import gaussian_kde
import pints
from scipy.special import gamma, loggamma
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

def K_h (x, h):
    return (1 / (2 * np.pi) ** 0.5) * (np.exp(-((x/h) ** 2) / 2)) / h

class MyLogPDF(pints.LogPDF):

    def __init__(self):
        pass

    def n_parameters(self):
        # Tell the inference method how many parameters there are
        return 2

    def __call__(self, p):
        # Extract the parameter x from the parameter vector
        h = p[0]
        sigma = p[1]
        n = X_train.shape[0]

        if h <= 0 or sigma <= 0:
            return -np.inf

        # start = time.time()

        k_h = K_h(euro_difs, h)
        np.fill_diagonal(k_h, 0)
        # print(time.time() - start)

        res = (k_h * y_train).sum(axis = 0) / (k_h.sum(axis = 0) + 1e-100)
        # print(time.time() - start)

        difs = (res - y_train.flatten()) ** 2
        # print(time.time() - start)

        const = - 0.5 * np.log(np.sqrt(2 * np.pi)) - n * np.log(sigma)
        sum_part = - difs.sum() / (2 * (sigma ** 2))

        # Calculate and return the log pdf
        return const + sum_part

class MyLogPrior(pints.LogPrior):

    def __init__(self):
        pass

    def n_parameters(self):
        # Tell the inference method how many parameters there are
        return 2

    def __call__(self, p):
        # Extract the parameter x from the parameter vector
        h = p[0]
        sigma = p[1]

        if h <= 0 or sigma <= 0:
            return -np.inf

        # Calculate and return the log pdf
        return - np.log(sigma)

data = pd.read_excel('euro_dollar.xlsx')
data = data.sort_values(by = ['curs_euro'])

data = data[data['curs_euro'] > -0.14]

list_of_euro = list(data['curs_euro'])
list_of_dollar = list(data['curs_dollar'])
n_samples = data['curs_euro'].size

np_data_dollar = np.array(data['curs_dollar']).reshape(-1, 1)
np_data_euro = np.array(data['curs_euro']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    np_data_euro,
    np_data_dollar,
    test_size=0.2,  # Доля тестовой выборки (можно указать train_size)
    random_state=2,  # Для воспроизводимости
)

XY_train = np.concatenate([X_train, y_train], axis = 1)
XY_test = np.concatenate([X_test, y_test], axis = 1)

n_samples_train = X_train.shape[0]
n_samples_test = X_test.shape[0]

euro_difs = X_train - X_train.reshape(1, X_train.shape[0])
h = 0.0017321346403785167

logpdf = MyLogPDF()
logprior = MyLogPrior()
logposterior = pints.LogPosterior(logpdf, logprior)

n_chains = 1
xs = [[h, 0.005055160417958599]]
mcmc = pints.MCMCController(logposterior, n_chains, xs, sigma0= [1e-7, 2e-8],  method= pints.MetropolisRandomWalkMCMC)
# mcmc.set_parallel(parallel = True)

# for sampler in mcmc.samplers():
#     sampler.set_target_acceptance_rate()

mcmc.set_max_iterations(100000)

chains = mcmc.run()

result = pd.DataFrame(chains[0, :, :], columns= ['h', 'sigma'])
result.to_csv(f'result8', index=False)