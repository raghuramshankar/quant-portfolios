# %%
# r packages
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# python packages
import yfinance as yf
from pandas_datareader import data as pdr
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
def get_t(tickers, start, end):
    t_prices = pdr.get_data_yahoo(tickers, start, end)["Close"]
    t_returns = t_prices.resample("D").ffill().pct_change().dropna(axis=0)
    return (t_prices, t_returns)


# fit mvt to get mean and covar of tickers
def fit_mvt(
    tickers,
    start=dt.datetime.now() - dt.timedelta(days=365 * 10),
    end=dt.datetime.now(),
):
    t_prices, t_returns = get_t(tickers=tickers, start=start, end=end)

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


if __name__ == "__main__":
    # get ticker data
    tickers = ["XRSG.L", "SPXP.L", "IWDG.L", "G500.L", "SWLD.L", "EQGB.L", "SGLN.L"]

    results = fit_mvt(tickers)

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        pd.DataFrame(results["scatter"], index=tickers, columns=tickers),
        annot=True,
        ax=ax,
    )
