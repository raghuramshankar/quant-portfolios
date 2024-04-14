# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from src.funcs import get_t, fit_mvt, construct_rbp, backtest_portfolio, plot_weights


def risk_parity(tickers):
    # get ticker data
    (t_names, t_prices, t_returns, t_cum_returns) = get_t(
        tickers=tickers, start=dt.datetime.now() - dt.timedelta(days=365 * 10)
    )

    # get mean and covar
    mvt_results = fit_mvt(t_returns)

    # construct risk parity portfolio
    b = np.ones((len(tickers), 1)) * 1 / len(tickers)
    weights_parity = pd.Series(construct_rbp(mvt_results["cov"], b), index=t_names).T

    # visualize weights
    _, ax = plt.subplots(1, 3, figsize=(12, 6))
    plot_weights(weights=weights_parity, title="Parity Portfolio", ax=ax[0])

    # construct risk budgeting portfolio
    b = np.array((0.2, 1e-5, 0.6)).reshape((-1, 1))
    sort_indices = np.argsort(tickers)
    b = b[sort_indices]
    weights_budget = pd.Series(construct_rbp(mvt_results["cov"], b), index=t_names).T

    # visualize weights
    plot_weights(weights=weights_budget, title="Budget Portfolio", ax=ax[1])

    # construct equal weight portfolio
    b = (np.ones((1, len(tickers))) * 1 / len(tickers)).flatten()
    weights_equal = pd.Series(b, index=t_names).T

    # visualize weights
    plot_weights(weights=weights_equal, title="Equal Weight Portfolio", ax=ax[2])

    # save pie plot
    plt.savefig(
        "outputs/risk_parity_weights__" + "_".join(tickers).replace(".", "_") + ".png"
    )

    # backtest portofolio performance
    _, ax = plt.subplots(figsize=(12, 6))
    _ = backtest_portfolio(
        t_returns=t_returns,
        weights=weights_parity,
        portfolio_name="Parity Portfolio",
        PLOT=True,
        ax=ax,
    )
    _ = backtest_portfolio(
        t_returns=t_returns,
        weights=weights_budget,
        portfolio_name="Budget Portfolio",
        PLOT=True,
        ax=ax,
    )
    _ = backtest_portfolio(
        t_returns=t_returns,
        weights=weights_equal,
        portfolio_name="Equal Weight Portfolio",
        PLOT=True,
        ax=ax,
    )

    # backtest asset performance
    for asset in t_returns.columns:
        _ = backtest_portfolio(
            t_returns=t_returns.loc[:, asset].to_frame(),
            weights=[1.0],
            portfolio_name=asset,
            PLOT=True,
            ax=ax,
        )

    # save backtest
    plt.savefig(
        "outputs/risk_parity_backtest__" + "_".join(tickers).replace(".", "_") + ".png"
    )


if __name__ == "__main__":
    tickers = ["VUSA.L", "CSH2.L", "UC90.L"]
    risk_parity(tickers=tickers)
