# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from src.funcs import get_t, fit_mvt, construct_rbp, backtest_portfolio

if __name__ == "__main__":
    # choose tickers
    tickers = ["VUSA.L", "FRXE.L", "UC90.L"]

    # get ticker data
    t_names, t_prices, t_returns = get_t(
        tickers=tickers, start=dt.datetime.now() - dt.timedelta(days=365 * 10)
    )

    # get mean and covar
    mvt_results = fit_mvt(t_returns)

    # construct risk parity portfolio
    b = np.ones((len(tickers), 1)) * 1 / len(tickers)
    weights_parity = pd.Series(construct_rbp(mvt_results["cov"], b), index=t_names).T

    # visualize weights
    _, ax = plt.subplots(figsize=(12, 6))
    weights_parity.plot.pie(autopct="%1.1f%%", ax=ax, title="Parity Portfolio")

    # construct risk budgeting portfolio
    b = np.array((0.6, 0.2, 0.2)).reshape((-1, 1))
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

    # backtest portofolio performance
    _, ax = plt.subplots(figsize=(12, 6))
    _ = backtest_portfolio(
        t_portfolio_returns=t_returns,
        weights=weights_parity,
        portfolio_name="Parity Portfolio",
        PLOT=True,
        ax=ax,
    )
    _ = backtest_portfolio(
        t_portfolio_returns=t_returns,
        weights=weights_budget,
        portfolio_name="Budget Portfolio",
        PLOT=True,
        ax=ax,
    )
    _ = backtest_portfolio(
        t_portfolio_returns=t_returns,
        weights=weights_equal,
        portfolio_name="Equal Weight Portfolio",
        PLOT=True,
        ax=ax,
    )

    # backtest asset performance
    for asset in t_returns.columns:
        _ = backtest_portfolio(
            t_portfolio_returns=t_returns.loc[:, asset].to_frame(),
            weights=[1.0],
            portfolio_name=asset,
            PLOT=True,
            ax=ax,
        )
