# r packages
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects

pandas2ri.activate()

# python packages
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import riskparityportfolio as rp
import datetime as dt
import matplotlib.pyplot as plt

yf.pdr_override()
utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)

# install r packages if not already installed
packageNames = ("fitHeavyTail", "sparseIndexTracking")
packnames_to_install = [x for x in packageNames if not rpackages.isinstalled(x)]
if len(packnames_to_install) > 0:
    utils.install_packages(StrVector(packnames_to_install))


def get_t(
    tickers,
    start=dt.datetime.now() - dt.timedelta(days=365 * 1),
    end=dt.datetime.now(),
):
    """get ticker data"""
    tickers = sorted(tickers)
    t_names = [
        ticker + ": " + yf.Ticker(ticker).info["shortName"] for ticker in tickers
    ]
    t_prices = pdr.get_data_yahoo(tickers, start, end)["Close"]
    t_returns = t_prices.resample("D").ffill().pct_change().dropna(axis=0)

    return (t_names, t_prices, t_returns)


def fit_mvt(t_returns):
    """fit mvt to get mean and covar of tickers"""
    # convert t to r dataframe
    data = pandas2ri.py2rpy_pandasdataframe(t_returns)

    # import r package
    fitHeavyTail = rpackages.importr("fitHeavyTail")

    # fit mvt to data
    results_r = fitHeavyTail.fit_mvt(data)

    # get results_r to python
    result_names = results_r.names[:5]
    results = dict()
    for idx, key in enumerate(result_names):
        results[key] = results_r[idx]

    return results


def construct_rbp(sigma, b):
    """construct risk parity/budgeting portfolio"""
    return rp.vanilla.design(sigma, b).T


def plot_portfolio(weights):
    """plot portfolio"""
    _, ax = plt.subplots(figsize=(5, 5))
    weights.plot.pie(autopct="%1.1f%%")


def design_sparse(X_train, r_train, l=1e-7, u=0.5, measure="ete"):
    # convert to r matrix
    X_train = robjects.r["as.matrix"](X_train)
    r_train = robjects.r["as.matrix"](r_train)

    # import r package
    spIndexTrack = rpackages.importr("sparseIndexTracking")

    # design sparse portfolio
    return spIndexTrack.spIndexTrack(X_train, r_train, l, u, measure)
