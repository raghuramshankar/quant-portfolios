# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from src.funcs import get_t, backtest_portfolio, plot_weights


if __name__ == "__main__":
    portfolio_name = "Three Fund Portfolio"
    tickers = ["VUSA.L", "CSH2.L", "UC90.L"]
    # tickers = ["IWDG.L", "VUSA.L", "V3AM.L"]
    weights = pd.Series([0.5, 0.3, 0.2])

    # get ticker data
    t_names, t_prices, t_returns = get_t(
        tickers=tickers, start=dt.datetime.now() - dt.timedelta(days=365 * 10)
    )

    # visualize weights
    _, ax = plt.subplots(1, 2, figsize=(12, 3))
    plot_weights(weights=weights, title="Three Fund Portfolio", ax=ax[0])

    # backtest asset performance
    for asset in t_returns.columns:
        _ = backtest_portfolio(
            t_portfolio_returns=t_returns.loc[:, asset].to_frame(),
            weights=[1.0],
            portfolio_name=asset,
            PLOT=True,
            ax=ax[1],
        )

    # backtest portfolio
    backtest_portfolio(
        t_portfolio_returns=t_returns,
        weights=weights,
        portfolio_name=portfolio_name,
        PLOT=True,
        ax=ax[1],
    )
