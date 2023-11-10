# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.funcs import get_t, fit_mvt, construct_rbp

if __name__ == "__main__":
    # choose tickers
    tickers = ["XRSG.L", "IWDG.L", "UC15.L"]
    # tickers = ["XRSG.L", "IWDG.L"]

    # get ticker data
    t_names, t_prices, t_returns = get_t(tickers=tickers)

    # get mean and covar
    mvt_results = fit_mvt(t_returns)

    # construct risk parity portfolio
    b = np.ones((len(tickers), 1)) * 1 / len(tickers)
    weights_parity = pd.Series(construct_rbp(mvt_results["cov"], b), index=t_names).T

    # visualize weights
    _, ax = plt.subplots(figsize=(12, 6))
    weights_parity.plot.pie(autopct="%1.1f%%", ax=ax, title="Parity Portfolio")

    # construct risk budgeting portfolio
    b = np.array((0.8, 0.1, 0.1)).reshape((-1, 1))
    sort_indices = np.argsort(tickers)
    b = b[sort_indices]
    weights_budget = pd.Series(construct_rbp(mvt_results["cov"], b), index=t_names).T

    # visualize weights
    _, ax = plt.subplots(figsize=(12, 6))
    weights_budget.plot.pie(autopct="%1.1f%%", ax=ax, title="Budget Portfolio")

    # construct equal weight portfolio
    b = (np.ones((1, len(tickers))) * 1 / len(tickers)).flatten()
    weights_equal = pd.Series(b, index=t_names).T

    # visualize weights
    _, ax = plt.subplots(figsize=(12, 6))
    weights_equal.plot.pie(autopct="%1.1f%%", ax=ax, title="Equal Weight Portfolio")

    # get portfolio returns df
    portfolio_returns = pd.DataFrame(
        (t_returns.to_numpy()),
        index=t_returns.index,
        columns=t_names,
    )

    # add portfolio backtest
    portfolio_returns["Parity Portfolio"] = (t_returns.to_numpy()) * np.matrix(
        weights_parity.to_numpy().reshape((-1, 1))
    )
    portfolio_returns["Budget Portfolio"] = (t_returns.to_numpy()) * np.matrix(
        weights_budget.to_numpy().reshape((-1, 1))
    )
    portfolio_returns["Equal Weight Portfolio"] = (t_returns.to_numpy()) * np.matrix(
        weights_equal.to_numpy().reshape((-1, 1))
    )

    # plot normalized backtest performance
    _, ax = plt.subplots(figsize=(12, 6))
    (1 + portfolio_returns).cumprod().plot.line(ax=ax)
