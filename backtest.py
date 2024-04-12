# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from src.funcs import get_t, backtest_portfolio, plot_weights, get_stats

if "__ipython__":
    %load_ext autoreload
    %autoreload 2
    # %matplotlib widget

if __name__ == "__main__":
    # allocate weights
    portfolio_name = "Three Fund Portfolio"
    tickers = ["VUSA.L", "CSH2.L", "UC90.L"]
    weights = pd.Series([0.7, 0.3, 0.0], index=tickers)
    sort_indices = np.argsort(tickers)
    weights = weights[sort_indices]

    # get ticker data
    t_names, t_prices, t_returns = get_t(
        tickers=tickers, start=dt.datetime.now() - dt.timedelta(days=365 * 10)
    )

    # visualize weights
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    plot_weights(weights=weights, title="Three Fund Portfolio", ax=ax)

    # backtest asset performance
    _, ax = plt.subplots(1, 1, figsize=(12, 9))
    for asset in t_returns.columns:
        _ = backtest_portfolio(
            t_portfolio_returns=t_returns.loc[:, asset].to_frame(),
            weights=[1.0],
            portfolio_name=asset,
            PLOT=True,
            ax=ax,
        )

    # backtest portfolio
    portfolio_prices_normalized = backtest_portfolio(
        t_portfolio_returns=t_returns,
        weights=weights,
        portfolio_name=portfolio_name,
        PLOT=True,
        ax=ax,
    )

    # tickers and portfolio stats
    stats = get_stats(t_prices=t_prices.loc[:, "CSH2.L"])
    stats = pd.concat((stats, get_stats(t_prices=t_prices.loc[:, "UC90.L"])), axis=0)
    stats = pd.concat((stats, get_stats(t_prices=t_prices.loc[:, "VUSA.L"])), axis=0)
    stats = pd.concat(
        (stats, get_stats(t_prices=portfolio_prices_normalized.iloc[:, 0])), axis=0
    )
    print(stats.to_string())
