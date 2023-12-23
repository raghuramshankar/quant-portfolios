# %%
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from src.funcs import get_t, backtest_portfolio, build_sparse, plot_weights


def sparse(ticker_index, num_tickers):
    # get ticker names
    ticker_data = pd.read_json("ticker_data.json")
    tickers_portfolio = ticker_data.loc[0:, "tickers"]
    tickers_portfolio = [
        ticker for ticker in tickers_portfolio if ticker.startswith("V")
    ]

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

    # print and plot results
    if not crmse_df.empty:
        result_df = crmse_df.iloc[crmse_df["crmse"].idxmin(), :]

        # get sparse portfolio tickers and weights
        result_tickers = [
            result_df[ticker]
            for ticker in result_df.index
            if ticker.startswith("ticker_")
        ]
        result_weights = [
            result_df[weight]
            for weight in result_df.index
            if weight.startswith("weight_")
        ]
        weights_sparse = pd.Series(
            {result_tickers[i]: result_weights[i] for i in range(len(result_tickers))}
        )

        # get total expense ratio
        overall_ter = np.dot(
            ticker_data[ticker_data["tickers"].isin(result_tickers)]["ters"]
            .to_numpy()
            .reshape((1, -1)),
            np.matrix(result_weights).reshape((-1, 1)),
        ).tolist()[0][0]

        # get returns of results again
        returns_results = t_all_returns.loc[:, result_tickers]

        # ensure the same index
        starting_idx = max(returns_results.index[0], t_index_returns.index[0])
        ending_idx = min(returns_results.index[-1], t_index_returns.index[-1])
        t_index_returns = t_index_returns.loc[starting_idx:ending_idx]
        returns_results = returns_results.loc[starting_idx:ending_idx]

        # visualize weights
        _, ax = plt.subplots(figsize=(12, 6))
        plot_weights(
            weights=weights_sparse,
            title="Index = %s, Overall TER = %f" % (ticker_index[0], overall_ter),
            ax=ax,
        )

        # save pie plot
        plt.savefig(
            "outputs/sparse_weights__"
            + "_".join(ticker_index).replace(".", "_")
            + ".png"
        )

        # plot sparse index portfolio vs index returns
        _, ax = plt.subplots(figsize=(12, 6))
        _ = backtest_portfolio(
            t_portfolio_returns=returns_results,
            weights=result_weights,
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
        ax.axvline(x=end_train)

        # save backtest
        plt.savefig(
            "outputs/sparse_backtest__"
            + "_".join(ticker_index).replace(".", "_")
            + ".png"
        )

    else:
        print("No good portfolios found")


if __name__ == "__main__":
    ticker_index = ["^RUT"]
    num_tickers = 3
    sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    ticker_index = ["^SPXEW"]
    num_tickers = 3
    sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    # ticker_index = ["^NDXE"]
    # num_tickers = 3
    # sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    # ticker_index = ["INDA"]
    # num_tickers = 3
    # sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    ticker_index = ["URTH"]
    num_tickers = 3
    sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    ticker_index = ["^RUA"]
    num_tickers = 3
    sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    # ticker_index = ["^SP1500"]
    # num_tickers = 3
    # sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    ticker_index = ["^RUI"]
    num_tickers = 3
    sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    # ticker_index = ["IWSZ.L"]
    # num_tickers = 3
    # sparse(ticker_index=ticker_index, num_tickers=num_tickers)

    ticker_index = ["ARKK"]
    num_tickers = 3
    sparse(ticker_index=ticker_index, num_tickers=num_tickers)
