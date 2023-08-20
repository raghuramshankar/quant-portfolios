# %%
import numpy as np
import pandas as pd
from src.funcs import get_t, fit_mvt, construct_rbp, plot_portfolio

if __name__ == "__main__":
    # choose tickers
    tickers = ["XRSG.L", "SPXP.L", "IWDG.L", "G500.L", "SWLD.L", "EQGB.L", "SGLN.L"]

    # get ticker data
    t_names, t_prices, t_returns = get_t(tickers=tickers)

    # get mean and covar
    results = fit_mvt(t_returns)

    # construct risk parity portfolio and plot
    b = np.ones((len(tickers), 1)) * 1 / len(tickers)
    weights = pd.Series(construct_rbp(results["cov"], b).flatten(), index=t_names).T
    plot_portfolio(weights=weights)

    # construct risk budgeting portfolio and plot
    b = np.vstack(
        [0.5, np.ones((len(tickers) - 1, 1)) * (1 - 0.5) / (len(tickers) - 1)]
    )
    weights = pd.Series(construct_rbp(results["cov"], b).flatten(), index=t_names).T
    plot_portfolio(weights=weights)
