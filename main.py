# %%
import numpy as np
import pandas as pd
from src.funcs import get_t, fit_mvt, construct_rbp, plot_portfolio

if '__ipython__':
    %load_ext autoreload
    %autoreload 2

if __name__ == "__main__":
    # choose tickers
    # tickers = [ "XRSG.L", "SPXP.L", "IWDG.L", "G500.L", "SWLD.L", "EQGB.L", "SGLN.L"]
    tickers = [ "XRSG.L", "SPXP.L", "IWDG.L", "G500.L", "SWLD.L", "EQGB.L"]
    # tickers = [ "SWLD.L", "RSP"]
    # tickers = ["AAPL", "TSLA", "NVDA", "XRSG.L"]
    # tickers = ["CSH2.L", "SPXP.L", "SGLN.L"]
    # tickers = ["CSH2.L", "RSP"]

    # get ticker data
    t_names, t_prices, t_returns = get_t(tickers=tickers)

    # get mean and covar
    results = fit_mvt(t_returns)

    # construct risk parity portfolio and plot
    b = np.ones((len(tickers), 1)) * 1 / len(tickers)
    weights = pd.Series(construct_rbp(results["cov"], b), index=t_names).T
    plot_portfolio(weights=weights)
    print("Risk Parity porfolio:\n", weights.to_string())

    # construct risk budgeting portfolio and plot
    risk_1 = 0.5
    sort_indices = np.argsort(tickers)
    b = np.vstack(
        [risk_1, np.ones((len(tickers) - 1, 1)) * (1 - risk_1) / (len(tickers) - 1)]
    )
    b = b[sort_indices]
    weights = pd.Series(construct_rbp(results["cov"], b), index=t_names).T
    plot_portfolio(weights=weights)
    print("Risk Budgeting porfolio:\n", weights.to_string())