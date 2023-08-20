# %%
# r packages
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# python packages
import yfinance as yf
from pandas_datareader import data as pdr
import riskparityportfolio as rp
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

yf.pdr_override()
utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)

# install r packages if not already installed
packageNames = ("fitHeavyTail", "riskParityPortfolio")
packnames_to_install = [x for x in packageNames if not rpackages.isinstalled(x)]
if len(packnames_to_install) > 0:
    utils.install_packages(StrVector(packnames_to_install))


# get ticker data
def get_t(
    tickers,
    start=dt.datetime.now() - dt.timedelta(days=10 * 365),
    end=dt.datetime.now(),
):
    t_names = [
        ticker + ": " + yf.Ticker(ticker).info["shortName"] for ticker in tickers
    ]
    t_prices = pdr.get_data_yahoo(tickers, start, end)["Close"]
    t_returns = t_prices.resample("D").ffill().pct_change().dropna(axis=0)
    return (t_names, t_prices, t_returns)


# fit mvt to get mean and covar of tickers
def fit_mvt(t_returns):
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


# construct risk parity/budgeting portfolio
def construct_rbp(sigma, b):
    return rp.vanilla.design(sigma, b).T


if __name__ == "__main__":
    # choose tickers
    tickers = ["XRSG.L", "SPXP.L", "IWDG.L", "G500.L", "SWLD.L", "EQGB.L", "SGLN.L"]

    # get ticker data
    t_names, t_prices, t_returns = get_t(tickers=tickers)

    # get mean and covar
    results = fit_mvt(t_returns)

    # construct risk parity portfolio
    b = np.ones((len(tickers), 1)) * 1 / len(tickers)
    weights = pd.Series(construct_rbp(results["cov"], b).flatten(), index=t_names).T

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    weights.plot.pie(autopct="%1.1f%%")
