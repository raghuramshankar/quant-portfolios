# %%
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import yfinance as yf
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import datetime as dt

yf.pdr_override()
utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)

# install r packages if not already installed
packageNames = ("fitHeavyTail", "riskParityPortfolio")
packnames_to_install = [x for x in packageNames if not rpackages.isinstalled(x)]
if len(packnames_to_install) > 0:
    utils.install_packages(StrVector(packnames_to_install))


# get ticker data
def get_t(tickers, start, end=dt.datetime.now()):
    t = pdr.get_data_yahoo(tickers, start, end)["Close"]
    return t


if __name__ == "__main__":
    tickers = ["VUSA.L", "IDTG.L"]
    end = dt.datetime.now()
    start = end - dt.timedelta(days=365 * 10)
    t = get_t(tickers=tickers, start=start, end=end)
    t = t.resample("D").ffill().pct_change().dropna(axis=0)
