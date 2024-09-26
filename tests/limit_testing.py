import helper as hp
import xgboost as xgb
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

def merge_dicts(dict1, dict2):

    for key in dict2:
        if key in dict1:
            dict1[key] += dict2[key]
        else:
            dict1[key] = dict2[key]

    return dict1


#Important Note:
# Progression not linear as features are removed
if __name__ == "__main__":
    
    tf_dic = {"week": 60*24*7}
    tf_str = 'week'
    tf_int = 60*24*7

    pairs = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']
    data = {}
    rolled = {}
    
    #USING THE FINAL FEATURE SET
    for p in pairs:
        data[p] = hp.minute_data_loading(pair=p, filtered=False)
        data[p], feat_lab = hp.feature_calculation(data[p], final=True)       
        rolled[p] = hp.rolling_window(data[p], training_size = 16, testing_size = 1, rols = -1)
    
    print("DATA LOADED")
       
    edge_trade_total = {}
    limits = [2.5, 4.5, 1, 1.5, 2, 3, 3.5, 4, 5]
    for l in limits:
        edge_trade_total[l] = {}
        for p in pairs:
            edge_trade_total[l][p] = []
    
    for p in pairs:
        print(p)
        edge_trades_p = []
        for i, rol in enumerate(rolled[p]):
            print(f"ROL {i+1} / {len(rolled[p])}", end="\r")
            train = rol[0]
            test = rol[1]

            train_prepd = hp.targets(train, {tf_str:tf_int}, feat_lab)

            d = list(train_prepd.values())[0]
            
            dtrain = xgb.DMatrix(data = d[feat_lab], label=d[tf_str])
            params = {
                'device': 'cuda',
            }
            num_rounds = 10
            bst = xgb.train(params, dtrain, num_rounds)
            
            dtest = xgb.DMatrix(data=test[feat_lab])
            y_pred = bst.predict(dtest)
            
            for l in limits:
                edge_trades = hp.trading_strat2({tf_str: y_pred}, test['Gmt time'], {tf_str: [-l, l]})
                edge_trade_total[l][p].append(edge_trades)
                
    print("PREDICTIONS MADE")
    
    for l in limits:

        trades = edge_trade_total[l]
        last_lim = [-l, l]
        trade_total = []

        for p in pairs:
            trades_p = trades[p]
            if(trade_total == []):
                for rol in trades_p:
                    trade_total.append({p : rol})
            else:
                for i, rol in enumerate(trades_p):
                    trade_total[i][p] = rol

        #Assume all trades have the same number of rols
        hist = [10000]
        for rol in range(len(trade_total)):
            print(f"ROL {rol + 1} / {len(trade_total)}", end='\r')
            
            ask_i = {}
            trades_i = trade_total[rol]
            for p in pairs:
                t = trades_i[p]
                ask_p = data[p]
                start_date, finish_date = hp.get_start_finish(t, offset= 60*24*15)
                ask_i[p] = ask_p[(ask_p['Gmt time'] >= start_date) & (ask_p['Gmt time'] <= finish_date)]
                ask_i[p] = ask_i[p]['Close'].values

            @dataclass
            class setStruct:
                trade_sz: int
                lev: int
            settings = setStruct(trade_sz=5, lev=1)

            hist = hp.stop_lev_conc_equity([ask_i, ask_i], trades_i, {tf_str : tf_int}, hist, settings)
        
        
        roi = (hist[-1] - 10000) / 10000 * 100
        years = len(hist) / (60*24*365)
        anualized_roi = (hist[-1] / 10000) ** (1/years) - 1
        anualized_roi *= 100
        drawdown = hp.maxDrawDown(hist)
        calmar = anualized_roi / drawdown
        print("########## STATS ##########")
        print(f"Lim: {l}")
        print(f"ROI: {roi:.2f} %")
        print(f"Anualized ROI: {anualized_roi:.2f} %")
        print(f"MDD: {drawdown:.2f} %")
        print(f"Calmar: {calmar:.2f}")    
        