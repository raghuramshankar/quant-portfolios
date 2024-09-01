# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from src.funcs import get_t, backtest_portfolio, plot_weights, get_stats

if __name__ == "__main__":
    # allocate weights
    portfolio_name = "Portfolio"
    tickers = ["G500.L", "XRSG.L", "SGLN.L", "CSH2.L"]
    weights = pd.Series([0.3, 0.15, 0.3, 0.25], index=tickers)

    sort_indices = np.argsort(tickers)
    weights = weights[sort_indices]

    # get ticker data
    (t_names, t_prices, t_returns, t_cum_returns) = get_t(
        tickers=tickers, start=dt.datetime.now() - dt.timedelta(days=365 * 10)
    )

    # visualize weights
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    plot_weights(weights=weights, title=portfolio_name, ax=ax)

    # backtest asset performance
    _, ax = plt.subplots(1, 1, figsize=(12, 9))
    for ticker in t_returns.columns:
        _ = backtest_portfolio(
            t_returns=t_returns.loc[:, ticker].to_frame(),
            weights=[1.0],
            portfolio_name=ticker,
            PLOT=True,
            ax=ax,
        )

    # backtest portfolio
    portfolio_cum_return = backtest_portfolio(
        t_returns=t_returns,
        weights=weights,
        portfolio_name=portfolio_name,
        PLOT=True,
        ax=ax,
    )

    # tickers and portfolio stats
    stats = pd.DataFrame()
    for ticker in tickers:
        stats = pd.concat(
            (stats, get_stats(t_prices=t_cum_returns.loc[:, ticker])), axis=0
        )
    stats = pd.concat(
        (stats, get_stats(t_prices=portfolio_cum_return.iloc[:, 0])), axis=0
    )
    print(stats.to_string())

    # save backtest
    plt.savefig("outputs/all_weather_portfolio_backtest" + ".png")
