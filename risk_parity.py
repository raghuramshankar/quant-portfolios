# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.funcs import get_t, fit_mvt, construct_rbp

if __name__ == "__main__":
    # choose tickers
    tickers = ["XRSG.L", "SPXP.L", "IWDG.L", "G500.L", "SWLD.L", "EQGB.L"]

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
    risk_1 = 0.5
    sort_indices = np.argsort(tickers)
    b = np.vstack(
        [risk_1, np.ones((len(tickers) - 1, 1)) * (1 - risk_1) / (len(tickers) - 1)]
    )
    b = b[sort_indices]
    weights_budget = pd.Series(construct_rbp(results["cov"], b), index=t_names).T

    # plot weights
    print("Risk Budgeting porfolio:\n", weights_budget.to_string())
    _, ax = plt.subplots(figsize=(12, 6))
    weights_budget.plot.pie(autopct="%1.1f%%", ax=ax)

    # plot t_returns
    _, ax = plt.subplots(figsize=(12, 6))
    pd.DataFrame(
        (1 + t_returns.to_numpy().transpose()).cumprod().transpose(),
        index=t_returns.index,
    ).plot.line(ax=ax)

    # plot backtest
    _, ax = plt.subplots(figsize=(12, 6))
    pd.DataFrame(
        (1 + t_returns.to_numpy() * np.matrix(weights_budget.to_numpy()).transpose())
        .cumprod()
        .transpose()
    ).plot.line(ax=ax)