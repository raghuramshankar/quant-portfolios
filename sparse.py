# %%
import numpy as np
import pandas as pd
import datetime as dt
from src.funcs import get_t, design_sparse

if __name__ == "__main__":
    # choose tickers
    tickers_portfolio = [
        "SPXP.L",
        "XRSG.L",
        "CNX1.L",
        "SGLN.L",
        "EQQQ.L",
        "FRIN.L",
        "FTAL.L",
    ]
    ticker_index = ["RSP"]

    # get ticker data
    start = dt.datetime(year=2020, month=1, day=1)
    t_portfolio_names, _, t_portfolio_returns = get_t(
        tickers=tickers_portfolio, start=start
    )
    _, _, t_index_returns = get_t(tickers=ticker_index, start=start)

    # ensure the same index
    starting_idx = max(t_portfolio_returns.index[0], t_index_returns.index[0])
    ending_idx = min(t_portfolio_returns.index[-1], t_index_returns.index[-1])
    t_index_returns = t_index_returns.loc[starting_idx:ending_idx]
    t_portfolio_returns = t_portfolio_returns.loc[starting_idx:ending_idx]

    # design sparse portfolio
    w_sparse = design_sparse(
        t_portfolio_returns, t_index_returns, l=1e-5, u=0.9, measure="ete"
    )

    # get dataframe with cumulative returns
    sparse_portfolio = dict()
    sparse_portfolio["sparse_" + ticker_index[0]] = (
        np.array(
            (
                1 + (t_portfolio_returns.to_numpy() * np.matrix(w_sparse).transpose())
            ).cumprod()
        )
        .flatten()
        .tolist()
    )
    sparse_portfolio[ticker_index[0]] = (
        (1 + t_index_returns.to_numpy()).cumprod().tolist()
    )
    sparse_portfolio = pd.DataFrame(sparse_portfolio, index=t_portfolio_returns.index)

    # plot sparse index portfolio vs index returns
    print(pd.DataFrame(w_sparse, index=t_portfolio_names).to_string())
    print(
        "CRMSE Tracking error: ",
        np.sqrt(
            np.sum(
                np.square(
                    sparse_portfolio["sparse_" + ticker_index[0]]
                    - sparse_portfolio[ticker_index[0]]
                )
            )
        ),
    )
    sparse_portfolio.plot.line(figsize=(12, 6))
    (1 + t_portfolio_returns).cumprod().plot.line(figsize=(12, 6))
