import helper as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
import matplotlib.dates as mdates

pairs = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']

ask = {}
x_axis_interval = 7

tech_indicators = False
for p in pairs:
    ask[p] = hp.minute_data_loading(p, 'ASK', filtered = False)
    if tech_indicators:
        ask[p], _ = hp.feature_calculation(ask[p])
        #select only a month of data
        start_idx = 10000
        month_size = 60*24*30
        ask[p] = ask[p].iloc[start_idx:start_idx+month_size]
        print(ask[p].iloc[start_idx])
        #plot the price and the technical indicators in subplots
        fig, axs = plt.subplots(4)
        axs[0].plot(ask[p]['Gmt time'], ask[p]['Close'], label = p)
        axs[0].plot(ask[p]['Gmt time'], ask[p]['ema3240'], label = 'ema 2days')
        axs[0].xaxis.set_major_locator(mdates.DayLocator(interval=x_axis_interval))
        axs[0].legend()
        axs[1].plot(ask[p]['Gmt time'], ask[p]['bb_pband3240'], label = 'bb_pband 2days')
        axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=x_axis_interval))
        axs[1].legend()
        axs[2].plot(ask[p]['Gmt time'], ask[p]['macd3240'], label = 'macd 2days')
        axs[2].xaxis.set_major_locator(mdates.DayLocator(interval=x_axis_interval))
        axs[2].legend()
        axs[3].plot(ask[p]['Gmt time'], ask[p]['rsi3240'], label = 'rsi 2days')
        axs[3].xaxis.set_major_locator(mdates.DayLocator(interval=x_axis_interval))
        axs[3].legend()
        plt.show()
    else:
        plt.plot(ask[p]['Gmt time'], ask[p]['Close'], label = p)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.legend()
        plt.show()

