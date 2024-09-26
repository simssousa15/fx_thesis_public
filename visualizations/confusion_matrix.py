from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import helper as hp
import numpy as np
from matplotlib.collections import LineCollection

def goof_sign(x):
    if x < 0:
        return 0
    if x > 0:
        return 1
    return -10
# 
tickers = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']

trades = {}
cm = np.zeros((2, 2))
for p in tickers:
    
    pipeline = "fin_score_2_0"
    curr_dir = os.getcwd() + f"/storage/Pipelines/{pipeline}/unfiltered/{p}/train_size=16/rols=-1/week"

    with open(curr_dir + "/trades.pkl", 'rb') as f:
        trades = pickle.load(f)

    price = hp.minute_data_loading(p, filtered=False)
    price = price[['Gmt time', 'Close']]
    #price['week'] = (price['Close'].shift(60*24*7) - price['Close']) * 100/ price['Close']

    print("num rols: ", len(trades))
    for rol_id, rol in enumerate(trades):
        #check if there is any value != 0
        if not any(rol['week']):
            continue

        print("Rol: ", rol_id, end = "\r")
        start, finish = hp.get_start_finish(rol, offset=60*24*10)
        rol_price = price[(price['Gmt time'] >= start) & (price['Gmt time'] <= finish)]['Close'].values
        rol = np.append(rol['week'].values, np.array([0 for i in range(len(rol_price) - len(rol['week'].values))]))


        # confusion matrix
        for i in range(len(rol)):
            if rol[i] != 0:
                r = goof_sign(rol[i])
                change = goof_sign((rol_price[i + 60*24*7] - rol_price[i]) / rol_price[i])
                if change != -10:
                    cm[r][change] += 1

        #print(cm)

print("#################################")
print(pipeline)
# calculate accuracy
accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
print("Accuracy: ", accuracy)

#normalize
cm = cm / cm.sum(axis=1)[:, np.newaxis]
print("Final confusion matrix")
print(cm)