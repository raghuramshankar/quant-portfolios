# %%
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from src.funcs import get_t, backtest_portfolio, build_sparse


if __name__ == "__main__":
    # get ticker names
    ticker_index = ["^SPXEW"]
    ticker_data = pd.read_json("ticker_data.json")
    tickers_portfolio = ticker_data.loc[0:, "tickers"]
    tickers_portfolio = [
        ticker for ticker in tickers_portfolio if ticker.startswith("V")
    ]
    num_tickers = 2

    # get all ticker data
    start_test = dt.datetime(year=2020, month=1, day=1)
    _, _, t_all_returns = get_t(tickers=tickers_portfolio, start=start_test)
    _, _, t_index_returns = get_t(tickers=ticker_index, start=start_test)

    # train only for some data
    end_train = dt.datetime(year=2022, month=12, day=31)

    # build sparse portfolio
    crmse_df = build_sparse(
        ticker_index=ticker_index,
        tickers_portfolio=tickers_portfolio,
        num_tickers=num_tickers,
        t_all_returns=t_all_returns,
        t_index_returns=t_index_returns,
        end_train=end_train,
    )

    # %%
    # print and plot results
    if not crmse_df.empty:
        result_df = crmse_df.iloc[crmse_df["crmse"].idxmin(), :]

        # get sparse portfolio tickers and weights
        tickers_results = [
            result_df[ticker]
            for ticker in result_df.index
            if ticker.startswith("ticker_")
        ]
        weights_results = [
            result_df[weight]
            for weight in result_df.index
            if weight.startswith("weight_")
        ]
        ters_results = ticker_data[ticker_data["tickers"].isin(tickers_results)][
            "ters"
        ].to_numpy()
        overall_ter = np.dot(
            ticker_data[ticker_data["tickers"].isin(tickers_results)]["ters"]
            .to_numpy()
            .reshape((1, -1)),
            np.matrix(weights_results).reshape((-1, 1)),
        ).tolist()[0][0]

        # get returns of results again
        returns_results = t_all_returns.loc[:, tickers_results]

        # ensure the same index
        starting_idx = max(returns_results.index[0], t_index_returns.index[0])
        ending_idx = min(returns_results.index[-1], t_index_returns.index[-1])
        t_index_returns = t_index_returns.loc[starting_idx:ending_idx]
        returns_results = returns_results.loc[starting_idx:ending_idx]

        # plot sparse index portfolio vs index returns
        _, ax = plt.subplots()
        _ = backtest_portfolio(
            t_portfolio_returns=returns_results,
            weights=weights_results,
            portfolio_name="sparse_" + ticker_index[0],
            PLOT=True,
            ax=ax,
        )
        _ = backtest_portfolio(
            t_portfolio_returns=t_index_returns,
            weights=[1.0],
            portfolio_name=ticker_index[0],
            PLOT=True,
            ax=ax,
        )
        plt.axvline(x=end_train)
        plt.show()

        # print final results
        print(result_df.to_string())
        print("Index = %s" % ticker_index)
        print(returns_results.tail())
        print("Overall TER = %f" % overall_ter)

    else:
        print("No good portfolios found")
