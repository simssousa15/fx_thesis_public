import pickle
import os
import helper as hp
import numpy as np

tickers = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']

curr_dir = os.getcwd()

#load pickle
with open(curr_dir + f"/fin_score_2_0_hist.pkl", 'rb') as f:
    hist = pickle.load(f)

with open(curr_dir + f"/fin_score_2_0_fail.pkl", 'rb') as f:
    fail = pickle.load(f)

with open(curr_dir + f"/fin_score_2_0_total.pkl", 'rb') as f:
    total = pickle.load(f)

for p in tickers:
    print(f"Ticker: {p}")
    print(f"Failed rate: {fail[p] / total[p]}")
    print(f"ROI: {hist[p] / 10000}")
    print("\n")

length = len(hist['total'])
year_size = int(length / 4)


print("\n")
for i in range(4):
    print(f"Year {i+1}")
    print(f"ROI: {(hist['total'][(i+1)*year_size] - hist['total'][i * year_size]) /  hist['total'][i * year_size]}")

#plot hist['total']

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

hist['total'] = np.array(hist['total'])
hist['total'] = hist['total'] / hist['total'][0]
#load minute data
ask = {}
for p in tickers:
    ask[p] = hp.minute_data_loading(p, filtered=False)
    #pick the last length of data
    dates = ask[p]['Gmt time'][-length:]
    ask[p] = ask[p]['Close'].values
    ask[p] = ask[p][-length:]
    ask[p] = ask[p] / ask[p][0]

plt.plot(dates, hist['total'], label='Trading Capital', linewidth=0.5)
plt.plot(dates, ask['EURUSD'], label='EURUSD', linewidth=0.5)
plt.plot(dates, ask['AUDUSD'], label='AUDUSD', linewidth=0.5)
plt.plot(dates, ask['GBPUSD'], label='GBPUSD', linewidth=0.5)
plt.plot(dates, ask['USDJPY'], label='USDJPY', linewidth=0.5)
plt.plot(dates, ask['USDCHF'], label='USDCHF', linewidth=0.5)
plt.plot(dates, ask['USDCAD'], label='USDCAD', linewidth=0.5)
plt.plot(dates, ask['NZDUSD'], label='NZDUSD', linewidth=0.5)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xlabel('Time')
plt.ylabel('Relative Growth')
plt.legend()
plt.show()

