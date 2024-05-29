import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

sns.set_style("ticks")
sns.set_palette("Set2")


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def sma_strategy(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals["signal"] = 0.0

    # Create short simple moving average
    signals["short_mavg"] = (
        data["Adj Close"]
        .rolling(window=short_window, min_periods=1, center=False)
        .mean()
    )

    # Create long simple moving average
    signals["long_mavg"] = (
        data["Adj Close"]
        .rolling(window=long_window, min_periods=1, center=False)
        .mean()
    )

    # Create signals
    signals["signal"][short_window:] = np.where(
        signals["short_mavg"][short_window:] > signals["long_mavg"][short_window:],
        1.0,
        0.0,
    )

    # Generate trading orders
    signals["positions"] = signals["signal"].diff()

    return signals


def plot_strategy(data, signals, ticker):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the closing price
    ax.plot(data.index, data["Close"], label=ticker)

    # Plot the short and long moving averages
    ax.plot(
        signals.index,
        signals["short_mavg"],
        label="Short SMA",
        linestyle="--",
        alpha=0.5,
    )
    ax.plot(
        signals.index, signals["long_mavg"], label="Long SMA", linestyle="--", alpha=0.5
    )

    # Plot the buy signals
    ax.plot(
        signals.loc[signals.positions == 1.0].index,
        signals.short_mavg[signals.positions == 1.0],
        "^",
        markersize=10,
        color="g",
        label="Buy",
    )

    # Plot the sell signals
    ax.plot(
        signals.loc[signals.positions == -1.0].index,
        signals.short_mavg[signals.positions == -1.0],
        "v",
        markersize=10,
        color="r",
        label="Sell",
    )

    ax.set(
        title=f"{ticker} - SMA Crossover Strategy",
        xlabel="Date",
        ylabel="Close Price (BRL)",
    )
    ax.legend()
    sns.despine()
    plt.show()
