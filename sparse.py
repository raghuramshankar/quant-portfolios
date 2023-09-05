# %%
import numpy as np
import pandas as pd
import datetime as dt
import itertools
import matplotlib.pyplot as plt
from src.funcs import get_t, design_sparse

if __name__ == "__main__":
    # choose tickers
    ticker_data = pd.read_json("ticker_data.json")
    tickers_portfolio = ticker_data.loc[:, "tickers"]
    ticker_index = ["RSP"]

    # get all combinations of n tickers
    num_tickers = 3
    tickers_comb = list(itertools.combinations(tickers_portfolio, num_tickers))
    c = ["ticker" + "_" + str(idx) for idx in range(num_tickers)]
    c.append("crmse")
    crmse_df = pd.DataFrame(columns=c)
    for comb in tickers_comb:
        try:
            # get ticker data
            start = dt.datetime(year=2020, month=1, day=1)
            t_portfolio_names, _, t_portfolio_returns = get_t(tickers=comb, start=start)
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
                        1
                        + (
                            t_portfolio_returns.to_numpy()
                            * np.matrix(w_sparse).transpose()
                        )
                    ).cumprod()
                )
                .flatten()
                .tolist()
            )
            sparse_portfolio[ticker_index[0]] = (
                (1 + t_index_returns.to_numpy()).cumprod().tolist()
            )
            sparse_portfolio = pd.DataFrame(
                sparse_portfolio, index=t_portfolio_returns.index
            )

            # get tracking error dictionary
            crmse_d = dict()
            for idx in range(num_tickers):
                crmse_d["ticker_" + str(idx)] = t_portfolio_names[idx]

            crmse_d["crmse"] = np.sqrt(
                np.sum(
                    np.square(
                        sparse_portfolio["sparse_" + ticker_index[0]]
                        - sparse_portfolio[ticker_index[0]]
                    )
                )
            )

            # add to crmse dataframe
            crmse_df = pd.concat(
                (crmse_df, pd.DataFrame(crmse_d, index=[0])), ignore_index=True
            )

            # print statistics
            print(pd.DataFrame(w_sparse, index=t_portfolio_names).to_string())

            # plot sparse index portfolio vs index returns
            sparse_portfolio.plot.line(figsize=(12, 6))

            plt.show()
        except:
            print("Skipping %s" % t_portfolio_names)
