from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import helper as hp
import numpy as np
from matplotlib.collections import LineCollection

# 
tickers = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']

trades = {}
for p in tickers:
    #
    curr_dir = os.getcwd() + f"/storage/Pipelines/fin_xgb/unfiltered/{p}/train_size=16/rols=-1/week"

    with open(curr_dir + "/trades.pkl", 'rb') as f:
        trades = pickle.load(f)

    price = hp.minute_data_loading(p, filtered=False)
    price = price[['Gmt time', 'Close']]

    print("num rols: ", len(trades))
    for rol_id, rol in enumerate(trades):
        #check if there is any value != 0
        if not any(rol['week']):
            continue
        

        start, finish = hp.get_start_finish(rol, offset=60*24*7)
        rol_price = price[(price['Gmt time'] >= start) & (price['Gmt time'] <= finish)]
        dates = rol_price['Gmt time']
        rol_price = rol_price['Close'].values
        rol = np.append(rol['week'].values, np.array([0 for i in range(len(rol_price) - len(rol['week'].values))]))

        #calculate the profit
        profit = 0
        for i in range(len(rol)):
            if rol[i] != 0:
                profit += rol[i] * (rol_price[i + 60*24*7] - rol_price[i]) / rol_price[i]

        #rol to staked
        # Precompute the increment array
        increment = rol * 5 / 10000

        # Initialize the staked array
        staked = np.zeros(len(rol))

        # Apply the increments using vectorized operations
        for i in range(len(rol)):
            if increment[i] != 0:
                end = min(i + 60 * 24 * 7, len(staked))
                staked[i:end] += increment[i]

        # max absolutem value of staked 1
        staked = np.clip(staked, -1, 1)
        non_zero = np.count_nonzero(staked) / len(staked)

        # Example data
        x = np.linspace(0, len(rol_price), len(rol_price))
        y = rol_price
        intensity = staked  # This is the third value affecting the color intensity

        # Create segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize intensity for color mapping with a fixed range from -1 to 1
        norm = plt.Normalize(-1, 1)

        colors = [(1, 0.2, 0.2), (0.8, 0.8, 0.8), (0, 0.7, 0)] 
        positions = [0.0, 0.5, 1.0]
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))
        lc = LineCollection(segments, cmap=custom_cmap, norm=norm)

        # Set the values used for colormapping
        lc.set_array(intensity)
        lc.set_linewidth(2)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
        fig.suptitle(f"{p} Rol:{rol_id} Result: {profit:.2f}")

        ax1.add_collection(lc)
        ax1.margins(0.1)
        ax1.set_ylabel('Price')
        custom_ticks = [0, 10000, 20000, 30000, 40000, 50000]  # Replace with your actual tick positions
        custom_labels = [dates.iloc[i].strftime('%Y-%m-%d') for i in custom_ticks]  # Custom labels
        ax1.set_xticks(custom_ticks)
        ax1.set_xticklabels(custom_labels)
        #cbar = plt.colorbar(lc, ax=ax1, orientation='horizontal')
        #cbar.set_label('Intensity')

        ax2.plot(staked)
        ax2.margins(0.1)
        ax2.set_ylabel('staked frac')
        ax2.set_xticks(custom_ticks)
        ax2.set_xticklabels(custom_labels)
        plt.savefig(f'storage/Images/fin_xgb/{p}-{rol_id}.png')
        #plt.show()
        print("Saved: ", f"{p}-{rol_id}.png")
        print("Non zero staked: ", non_zero)
        plt.close(fig)