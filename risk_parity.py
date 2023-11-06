# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.funcs import get_t, fit_mvt, construct_rbp

if __name__ == "__main__":
    # choose tickers
    tickers = [
        "XRSG.L",
        "LGGG.L",
        "UC15.L",
    ]

    # get ticker data
    t_names, t_prices, t_returns = get_t(tickers=tickers)

    # get mean and covar
    results = fit_mvt(t_returns)

    # construct risk parity portfolio
    b = np.ones((len(tickers), 1)) * 1 / len(tickers)
    weights_parity = pd.Series(construct_rbp(results["cov"], b), index=t_names).T

    # plot
    print("Risk Parity porfolio:\n", weights_parity.to_string())
    _, ax = plt.subplots(figsize=(12, 6))
    weights_parity.plot.pie(autopct="%1.1f%%", ax=ax)

    # construct risk budgeting portfolio
    b = np.array((0.5, 0.4, 0.1)).reshape((-1, 1))
    sort_indices = np.argsort(tickers)
    b = b[sort_indices]
    weights_budget = pd.Series(construct_rbp(results["cov"], b), index=t_names).T

    # plot weights
    print("Risk Budgeting porfolio:\n", weights_budget.to_string())
    _, ax = plt.subplots(figsize=(12, 6))
    weights_budget.plot.pie(autopct="%1.1f%%", ax=ax)

    # get returns_df
    returns_df = pd.DataFrame(
        (1 + t_returns.to_numpy().transpose()).cumprod(axis=1).transpose(),
        index=t_returns.index,
        columns=t_names,
    )

    # add portfolio backtest
    returns_df["Parity Portfolio"] = (
        (
            (1 + t_returns.to_numpy())
            * np.matrix(weights_parity.to_numpy().reshape((-1, 1)))
        )
        .cumprod()
        .transpose()
    )
    returns_df["Budget Portfolio"] = (
        (
            (1 + t_returns.to_numpy())
            * np.matrix(weights_budget.to_numpy().reshape((-1, 1)))
        )
        .cumprod()
        .transpose()
    )

    # plot returns_df
    _, ax = plt.subplots(figsize=(12, 6))
    returns_df.plot.line(ax=ax)
