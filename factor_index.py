# %%
from src.funcs import get_t_fundamental, get_sp500_list

if "__ipython__":
    %load_ext autoreload
    %autoreload 2

if __name__ == "__main__":
    tickers = get_sp500_list()
    t_data = get_t_fundamental(tickers=tickers)