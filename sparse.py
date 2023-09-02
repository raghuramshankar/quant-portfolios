# %%
import numpy as np
import pandas as pd
import datetime as dt
from src.funcs import get_t, design_sparse

# if '__ipython__':
#     %load_ext autoreload
#     %autoreload 2

if __name__ == "__main__":
    # choose tickers
    tickers_portfolio = ["XRSG.L", "SPXP.L"]
    ticker_index = ["RSP"]

    # get ticker data
    start = dt.datetime.now() - dt.timedelta(days=365 * 4)
    _, _, t_portfolio_returns = get_t(tickers=tickers_portfolio, start=start)
    _, _, t_index_returns = get_t(tickers=ticker_index, start=start)

    # design sparse portfolio
    w_sparse = design_sparse(t_portfolio_returns, t_index_returns, u=0.9)

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
    print(pd.DataFrame(w_sparse, index=tickers_portfolio))
    sparse_portfolio.plot.line()
    (1 + t_portfolio_returns).cumprod().plot.line()
