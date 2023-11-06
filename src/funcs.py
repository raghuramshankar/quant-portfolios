# r packages
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects

pandas2ri.activate()

# python packages
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import riskparityportfolio as rp
import datetime as dt
import itertools

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
    start=dt.datetime.now() - dt.timedelta(days=365 * 10),
    end=dt.datetime.now(),
):
    """get ticker data"""
    tickers = sorted(tickers)
    # t_names = [ticker + ": " + yf.Ticker(ticker).info["longName"] for ticker in tickers]
    t_names = tickers
    t_prices = pdr.get_data_yahoo(tickers, start, end, progress=True)["Close"]
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


def design_sparse(X_train, r_train, l=1e-7, u=0.5, measure="ete"):
    """design sparse portfolio to track index returns"""
    # convert to r matrix
    X_train = robjects.r["as.matrix"](X_train)
    r_train = robjects.r["as.matrix"](r_train)

    # import r package
    spIndexTrack = rpackages.importr("sparseIndexTracking")

    # design sparse portfolio
    return spIndexTrack.spIndexTrack(X_train, r_train, l, u, measure)


def backtest_portfolio(t_portfolio_returns, weights, portfolio_name, PLOT, ax=None):
    """backtest portfolio returns with weights"""
    # create dictionary
    portfolio_returns = dict()
    portfolio_returns[portfolio_name] = (
        np.array(
            (
                1
                + np.dot(
                    t_portfolio_returns.to_numpy().reshape((-1, len(weights))),
                    np.matrix(weights).reshape((-1, 1)),
                )
            ).cumprod()
        )
        .flatten()
        .tolist()
    )

    # add index
    portfolio_returns = pd.DataFrame(portfolio_returns, index=t_portfolio_returns.index)

    if PLOT:
        portfolio_returns.plot.line(figsize=(20, 6), ylabel="Return", ax=ax)

    return portfolio_returns


def build_sparse(
    ticker_index,
    tickers_portfolio,
    num_tickers,
    t_all_returns,
    t_index_returns,
    end_train,
):
    # get all combinations of n tickers
    tickers_comb = list(itertools.combinations(tickers_portfolio, num_tickers))
    c = ["ticker" + "_" + str(idx) for idx in range(num_tickers)]
    c = c + ["weight" + "_" + str(idx) for idx in range(num_tickers)]
    c = c + ["crmse"]
    crmse_df = pd.DataFrame(columns=c)

    for comb_idx, comb in enumerate(tickers_comb):
        print("Combination %d/%d: %s" % (comb_idx + 1, len(tickers_comb), comb))
        # get comb ticker data
        t_portfolio_returns = t_all_returns.loc[:, comb]

        # ensure the same index
        starting_idx = max(t_portfolio_returns.index[0], t_index_returns.index[0])
        ending_idx = min(t_portfolio_returns.index[-1], t_index_returns.index[-1])
        t_index_returns = t_index_returns.loc[starting_idx:ending_idx]
        t_portfolio_returns = t_portfolio_returns.loc[starting_idx:ending_idx]

        try:
            # design sparse portfolio
            w_sparse = design_sparse(
                X_train=t_portfolio_returns.loc[:end_train],
                r_train=t_index_returns.loc[:end_train],
                l=1e-9,
                u=0.9,
                measure="ete",
            )

            # get dataframe with cumulative returns for all the data
            sparse_portfolio = backtest_portfolio(
                t_portfolio_returns=t_portfolio_returns,
                weights=w_sparse,
                portfolio_name="sparse_" + ticker_index[0],
                PLOT=False,
            )
            sparse_index = backtest_portfolio(
                t_portfolio_returns=t_index_returns,
                weights=np.array([1.0]),
                portfolio_name=ticker_index[0],
                PLOT=False,
            )

            # get crmse
            crmse = np.sqrt(
                np.sum(
                    np.square(
                        sparse_portfolio["sparse_" + ticker_index[0]]
                        - sparse_index[ticker_index[0]]
                    )
                )
            )

            # add to results only if
            if crmse < 3.0:
                # get tracking error dictionary
                crmse_d = dict()
                for idx in range(num_tickers):
                    crmse_d["ticker_" + str(idx)] = comb[idx]
                    crmse_d["weight_" + str(idx)] = w_sparse[idx]

                crmse_d["crmse"] = crmse

                # add to crmse dataframe
                crmse_df = pd.concat(
                    (crmse_df, pd.DataFrame(crmse_d, index=[0])), ignore_index=True
                )
            else:
                print("Skipping because CRMSE is too high: %.2f\n" % (crmse))

        except:
            pass

    return crmse_df
