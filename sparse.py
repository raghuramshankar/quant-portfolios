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
    tickers_portfolio = ticker_data.loc[0:5, "tickers"]
    ticker_index = ["RSP"]

    # get all combinations of n tickers
    num_tickers = 3
    tickers_comb = list(itertools.combinations(tickers_portfolio, num_tickers))
    c = ["ticker" + "_" + str(idx) for idx in range(num_tickers)]
    c = c + ["weight" + "_" + str(idx) for idx in range(num_tickers)]
    c = c + ["crmse"]
    crmse_df = pd.DataFrame(columns=c)

    # get all ticker data
    start_test = dt.datetime(year=2020, month=1, day=1)
    t_all_names, _, t_all_returns = get_t(tickers=tickers_portfolio, start=start_test)
    _, _, t_index_returns = get_t(tickers=ticker_index, start=start_test)

    for comb_idx, comb in enumerate(tickers_comb):
        print("Trying combination %d/%d: %s" % (comb_idx + 1, len(tickers_comb), comb))
        # get comb ticker data
        t_portfolio_returns = t_all_returns.loc[:, comb]
        t_portfolio_names, _, _ = get_t(tickers=comb, start=start_test)

        # ensure the same index
        starting_idx = max(t_portfolio_returns.index[0], t_index_returns.index[0])
        ending_idx = min(t_portfolio_returns.index[-1], t_index_returns.index[-1])
        t_index_returns = t_index_returns.loc[starting_idx:ending_idx]
        t_portfolio_returns = t_portfolio_returns.loc[starting_idx:ending_idx]

        # train only until for some of the data
        end_train = dt.datetime(year=2021, month=12, day=31)

        try:
            # design sparse portfolio
            w_sparse = design_sparse(
                t_portfolio_returns.loc[:end_train],
                t_index_returns.loc[:end_train],
                l=1e-9,
                u=0.5,
                measure="ete",
            )

            # get dataframe with cumulative returns for all the data
            sparse_portfolio = dict()
            sparse_portfolio["sparse_" + ticker_index[0] + ":" + str(comb)] = (
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

            # get crmse
            crmse = np.sqrt(
                np.sum(
                    np.square(
                        sparse_portfolio["sparse_" + ticker_index[0] + ":" + str(comb)]
                        - sparse_portfolio[ticker_index[0]]
                    )
                )
            )

            # add to results only if
            if crmse < 3.0:
                # get tracking error dictionary
                crmse_d = dict()
                for idx in range(num_tickers):
                    crmse_d["ticker_" + str(idx + 1)] = t_portfolio_names[idx]
                    crmse_d["weight_" + str(idx + 1)] = w_sparse[idx]

                crmse_d["crmse"] = crmse

                # add to crmse dataframe
                crmse_df = pd.concat(
                    (crmse_df, pd.DataFrame(crmse_d, index=[0])), ignore_index=True
                )

                # plot sparse index portfolio vs index returns
                sparse_portfolio.plot.line(figsize=(20, 6), ylabel="Return")
                plt.axvline(x=end_train)

                plt.show()
        except:
            pass

    # print final results
    print(crmse_df.to_string())
