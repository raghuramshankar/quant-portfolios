# Risk Parity Portfolios
- Uses [Prof. Daniel.P.Palomar](https://github.com/dppalomar) and the [Convex Research](https://github.com/convexfi) group's quantitative tools to model stocks, construct and backtest risk parity/budgeting portfolios.

## Tasks/Tools:
- Selects list of tickers to model and construct a portfolio using factor based models (TBD)
- Fits Multivariate Student's t distribution to a list of stocks using [fitHeavyTail](https://github.com/convexfi/fitHeavyTail)
- Constructs a risk parity/budgeting portfolio using [riskParity.py](https://github.com/convexfi/riskparity.py)
- Backtests returns of constructed portfolios using [portfolioBacktest](https://github.com/dppalomar/portfolioBacktest)

## Requirements:
- Python
- [Rpy2](https://github.com/rpy2/rpy2)
- R compiler
- requirements.txt
