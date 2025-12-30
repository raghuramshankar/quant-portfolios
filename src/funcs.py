import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt


def get_t(
    tickers,
    start=dt.datetime.now() - dt.timedelta(days=365 * 10),
    end=dt.datetime.now(),
):
    """get ticker price data"""
    tickers = sorted(tickers)
    t_names = tickers
    t_prices = yf.download(tickers, start=start, end=end, progress=True)[
        "Close"
    ].dropna()

    # remove erroneous prices
    t_prices = t_prices[t_prices.pct_change() < 0.1].ffill()

    # get daily returns
    t_returns = (
        t_prices[t_prices.pct_change() < 0.1]
        .resample("D")
        .ffill()
        .pct_change()
        .dropna(axis=0)
    )

    # get cumulative returns
    t_cum_returns = (1 + t_returns).cumprod()

    return (t_names, t_prices, t_returns, t_cum_returns)


def get_t_fundamental(
    tickers,
):
    """get ticker all data"""
    t_data = pd.DataFrame({ticker: yf.Ticker(ticker).info for ticker in tickers})

    return t_data


def get_sp500_list():
    """get list of tickers in S&P500 index"""
    return pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0][
        "Symbol"
    ].tolist()


def plot_weights(weights, title, ax):
    """plot weights as pie chart"""
    weights.plot.pie(autopct="%1.1f%%", ax=ax, title=title)


def backtest_portfolio(t_returns, weights, portfolio_name, PLOT, ax=None):
    """backtest portfolio returns with fixed weights"""
    portfolio_cum_return = dict()
    portfolio_cum_return[portfolio_name] = (
        (1 + np.array(t_returns.to_numpy() * np.matrix(weights).reshape((-1, 1))))
        .flatten()
        .cumprod()
    )
    portfolio_cum_return = pd.DataFrame(portfolio_cum_return, index=t_returns.index)

    if PLOT:
        portfolio_cum_return.plot.line(
            ylabel="Cumulative Return", ax=ax, linewidth=2, figsize=(12, 9)
        )

    return portfolio_cum_return


def get_stats(t_prices: pd.Series):
    stats = pd.DataFrame()
    num_years = (t_prices.index[-1] - t_prices.index[0]).total_seconds() / (
        60 * 60 * 24 * 252
    )
    stats["Annualized Returns [%]"] = pd.Series(
        (
            np.power(
                t_prices.iloc[-1] / t_prices.iloc[0],
                (1 / num_years),
            )
            - 1
        )
        * 100,
    )
    stats["Annualized Volatility [%]"] = pd.Series(
        (t_prices.pct_change().std() * np.sqrt(252)) * 100
    )
    stats["Max Drawdown [%]"] = (
        (t_prices - t_prices.cummax()) / t_prices.cummax()
    ).min() * 100
    stats.index = [t_prices.name]

    return stats
