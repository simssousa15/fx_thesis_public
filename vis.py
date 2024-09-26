import pickle
from matplotlib import ticker
import matplotlib.pyplot as plt
import helper as hp
import argparse
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import time
import os

# to try
# from sklearn.gaussian_process import GaussianProcessRegressor
# SVR
# Vowpal Wabbit
# FTRL (Follow The Regularized Leader)
# linha inutil apagar

def check_time_difference(datetimes):
    #datetimes = [datetime.strptime(dt, "%d.%m.%Y %H:%M:%S.%f") for dt in datetimes]
    for i in range(len(datetimes) - 1):
        time_diff = datetimes[i+1] - datetimes[i]
        if time_diff > timedelta(minutes=1):
            return True
    return False

def print_stats(init_bank, eq, bh = False):
    print("######## Stats ########")
    print("Initial Bank: ", init_bank)
    print("Final Bank: ", round(eq[-1],2))
    print("Time: {:.2f} months".format(len(eq) / (60*24*30)))
    roi = (eq[-1] - init_bank) / init_bank * 100
    print("ROI: {:.2f} %".format(roi))
    years = len(eq) / (60*24*365)
    anualized_roi = (eq[-1] / init_bank) ** (1/years) - 1
    print("Annualized ROI: {:.2f} %".format(anualized_roi * 100))
    mdd = hp.maxDrawDown(eq)
    print(f"MDD: {mdd} %")
    print(f"Calmar: {anualized_roi/mdd*100}")
    #print(f"Min: {min(eq)}")
    if(bh != False):
        print(f"Buy&Hold")
        bh_roi = (bh[-1] - init_bank) / init_bank * 100
        print("ROI: {:.2f} %".format(bh_roi))
        anualized_roi2 = (bh[-1] / init_bank) ** (1/years) - 1
        print("Annualized ROI: {:.2f} %".format(anualized_roi2 * 100))
        print(f"MDD: {hp.maxDrawDown(bh)} %")
        print("-----------------------")
    return

def print_specs(p, settings):
    print("#### Trading Specs ####")
    print("Pair: ", p)
    print("Time Frame: ", settings.tf_str)
    print("Trade size: ", settings.trade_sz)
    print("Leverage: ", settings.lev)
    print("-----------------------")
    return

def load_data(p, settings):
    
    f_bool = ''
    if(settings.filtered == True):
        f_bool = 'filtered'
    else:
        f_bool = 'unfiltered'
    
    curr_dir = "/home/simao/Documents/IST/Code/financial" + f"/storage/Pipelines/{settings.pipeline}/{f_bool}/{p}"
    if "rolling" in settings.pipeline:
        curr_dir += "/train_size=" + str(settings.train_sz)

    curr_dir += '/rols=-1'

    with open(curr_dir + "/trades.pkl", 'rb') as f:
        trades = pickle.load(f)
    
    print(f'Trades {p} {settings.pipeline} train_size={settings.train_sz} {f_bool} loaded')
    
    #Just save close to spare ram memory
    ask = hp.minute_data_loading(p, filtered = settings.filtered)
    #bid = hp.minute_data_loading(p, 'BID', filtered = settings.filtered)
    bid = []
    
    return ask, bid, trades

def parallel_load_data(pairs, settings):
    ask = {}
    bid = {}
    trades = []

    def load_data_for_pair(pair):
        ask[pair], bid[pair], t = load_data(pair, settings)
        return t

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Load data for each pair concurrently
        futures = [executor.submit(load_data_for_pair, pair) for pair in pairs]

        # Retrieve the results as they become available
        for future in concurrent.futures.as_completed(futures):
            trades.append(future.result())

    return ask, bid, trades

def concurrent_trading(pairs, settings):
    #Alll the rols at the same time
    #Assume timeframe only one to begin with
    tf_dic = {'hour': 60, '3hour': 60*3, 'day': 60*24, '3day': 60*24*3, 'week': 60*24*7}
    tf = {settings.tf_str: tf_dic[settings.tf_str]}
    tf_str = list(tf.keys())[0]
    tf_int = tf[tf_str]

    """ ask, bid, trades = parallel_load_data(pairs, settings) """
    ask = {}
    bid = {}
    trades = []
    for p in pairs:
        ask[p] = hp.minute_data_loading(p, filtered = settings.filtered)
        #bid[p] = hp.minute_data_loading(p, 'BID', filtered = settings.filtered)
        trade_file = "/home/simao/Documents/IST/Code/financial" + f"/storage/Pipelines/{settings.pipeline}/unfiltered/{p}/train_size={settings.train_sz}/rols=-1/week/trades.pkl" 
        with open(trade_file, 'rb') as f:
            t = pickle.load(f)
        print(f"{settings.pipeline} {p} trades loaded")

        if (trades == []):
            for i in t:
                trades.append({p: i})
        else:
            for idx, val in enumerate(t):
                trades[idx][p] = val

    #Assume all trades have the same number of rols
    hist = {}
    total = {}
    fail = {}
    for p in pairs:
        total[p] = 0
        fail[p] = 0
        hist[p] = 0
    hist['total'] = [settings.init_bank]
    
    for rol in range(len(trades)):
        #print(f"ROL {rol + 1} / {len(trades)}", end='\r')
        
        ask_i = {}
        #bid_i = {}
        trades_i = trades[rol]
        for p in pairs:
            t = trades_i.get(p)
            if t is None:
                continue
            ask_p = ask[p]
            #bid_p = bid[p]
            start_date, finish_date = hp.get_start_finish(t, offset= (60*24*15))
            ask_i[p] = ask_p[(ask_p['Gmt time'] >= start_date) & (ask_p['Gmt time'] <= finish_date)]
            ask_i[p] = ask_i[p]['Close'].values
            #bid_i[p] = bid_p[(bid_p['Gmt time'] >= start_date) & (bid_p['Gmt time'] <= finish_date)]
            #bid_i[p] = bid_i[p]['Close'].values

        #hist, fail, total = hp.conc_equity([ask_i, ask_i], trades_i, tf, hist, settings, fail=fail, total=total)
        hist, fail, total = hp.stop_lev_conc_equity([ask_i, ask_i], trades_i, tf, hist, settings, fail, total)
    
    """ #get current directory from os
    curr_dir = os.getcwd()

    #save fail and total as pickle
    with open(curr_dir + f"/{settings.pipeline}_hist_no_lev.pkl", 'wb') as f:
        pickle.dump(hist, f)
    
    with open(curr_dir + f"/{settings.pipeline}_fail_no_lev.pkl", 'wb') as f:
        pickle.dump(fail, f)
    
    with open(curr_dir + f"/{settings.pipeline}_total_no_lev.pkl", 'wb') as f:
        pickle.dump(total, f)
     
    print("Saved") """

    return hist['total']

def concurrent_trading2(pairs, settings):
    #Alll the rols at the same time
    #Assume timeframe only one to begin with
    tf_dic = {'hour': 60, '3hour': 60*3, 'day': 60*24, '3day': 60*24*3, 'week': 60*24*7}
    tf = {settings.tf_str: tf_dic[settings.tf_str]}
    tf_str = list(tf.keys())[0]
    tf_int = tf[tf_str]

    """ ask, bid, trades = parallel_load_data(pairs, settings) """
    ask = {}
    bid = {}
    trades = []
    for p in pairs:
        ask[p] = hp.minute_data_loading(p, filtered = settings.filtered)
        #bid[p] = hp.minute_data_loading(p, 'BID', filtered = settings.filtered)
        trade_file = "/home/simao/Documents/IST/Code/financial" + f"/storage/Pipelines/{settings.pipeline}/filtered/{p}/train_size={settings.train_sz}/rols=-1/trades.pkl" 
        with open(trade_file, 'rb') as f:
            t = pickle.load(f)
        print(f"{settings.pipeline} {p} trades loaded")

        if (trades == []):
            for i in t:
                trades.append({p: i})
        else:
            for idx, val in enumerate(t):
                trades[idx][p] = val

    #Assume all trades have the same number of rols
    hist = [settings.init_bank]
    for rol in range(len(trades)):
        print(f"ROL {rol + 1} / {len(trades)}", end='\r')
        
        ask_i = {}
        #bid_i = {}
        volume_i = {}
        trades_i = trades[rol]
        #print(trades_i)
        for p in pairs:
            try:
                t = trades_i[p]
                ask_p = ask[p]
                #bid_p = bid[p]
                start_date, finish_date = hp.get_start_finish(t, offset= 60*24*14)
                ask_i[p] = ask_p[(ask_p['Gmt time'] >= start_date) & (ask_p['Gmt time'] <= finish_date)]
                volume_i[p] = ask_i[p]['Volume'].values
                ask_i[p] = ask_i[p]['Close'].values
                #bid_i[p] = bid_p[(bid_p['Gmt time'] >= start_date) & (bid_p['Gmt time'] <= finish_date)]
                #bid_i[p] = bid_i[p]['Close'].values
            except:
                print(p, rol)
            
            

        hist = hp.conc_equity_noFlats([ask_i, ask_i], volume_i, trades_i, tf, hist, settings)

    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--starting-bank', dest = 'init_bank', default = 10000, type = int, help='Starting value in account')
    parser.add_argument('-t', '--trade_size', dest='trade_sz', default = 5, type = float, help='Ammount per trade')
    parser.add_argument('-tf', '--time_frame', dest = 'tf_str', default = 'week', choices=['hour', '3hour', 'day', '3day', 'week'], help='Time frame for trades')
    parser.add_argument('-v', '--visualization', dest = 'vis', default = False, type=bool, help='Visualizations enabled/disabled')
    parser.add_argument('-l', '--leverage',  dest='lev', default = 1, type = float, help='Leverage to add to each transaction')
    parser.add_argument('-p', '--pair', default = 'full', choices= ['all', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'USDCAD', 'NZDUSD', 'full'])
    parser.add_argument('-pip', '--pipeline', default = 'fin_xgb')
    parser.add_argument('-ts', '--training_size', dest='train_sz', default = 16, type=int)
    parser.add_argument('-f', '--filtered', default = False, type = bool)


    settings = parser.parse_args()

    #missing 
    tickers = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']
    #tickers = ['EURUSD', 'AUDUSD', 'GBPUSD']
    tf_dic = {'hour': 60, '3hour': 60*3, 'day': 60*24, '3day': 60*24*3, 'week': 60*24*7}

    if settings.pair == 'all':
        settings.pair = tickers
    elif settings.pair == 'full':
        hist = concurrent_trading(tickers, settings)
        print_stats(settings.init_bank, hist)
        #save hist
        with open(f"/home/simao/Documents/IST/Code/financial/storage/Results/{settings.pipeline}-{len(tickers)}pairs_concorrent.pkl", 'wb') as f:
            pickle.dump(hist, f)
        
        return
    else:
        settings.pair = [settings.pair]

    
    for p in settings.pair:
        
        print_specs(p, settings)

        ask, bid, trades = load_data(p, settings)
        equity_curve = []
        bh_curve = []

        for idx in range(len(trades)):

            #print(f"ROL {idx + 1} / {len(trades)}", end='\r')
            trades_i = trades[idx][settings.tf_str]
            #print("Missing data status: ", check_time_difference(trades_i.index.tolist()))

            """ trades_idxs = np.isin(ask['Gmt time'].values.tolist(), trades_i.index.tolist())
            ask_i = ask[trades_idxs]['Close'].values
            bid_i = bid[trades_idxs]['Close'].values
            trades_i = trades_i.values """

            #add offset of a week (maximum timeframe) to complete all trades
            start_date, finish_date = hp.get_start_finish(trades_i, offset= 60*24*30)
            ask_i = ask[(ask['Gmt time'] >= start_date) & (ask['Gmt time'] <= finish_date)]
            ask_i = ask_i['Close'].values
            #bid_i = bid[(bid['Gmt time'] >= start_date) & (bid['Gmt time'] <= finish_date)]
            #bid_i = bid_i['Close'].values
            trades_i = trades_i.values


            #Trading
            start_trad = settings.init_bank if(idx == 0) else eq[-1]
            #start_bh = settings.init_bank if(idx == 0) else bh[-1]
            eq = hp.equity([ask_i, ask_i], trades_i, tf_dic[settings.tf_str], start_trad, settings.trade_sz, settings.lev)
            #bh = hp.buy_hold([ask_i, ask_i], start_bh, False if idx == 0 else True)
            equity_curve.append(eq)
            #bh_curve.append(bh)


            if(settings.vis):
                fig, axs = plt.subplots(3, 1, figsize=(12, 6))
                axs[0].plot(range(len(ask_i)),hp.space(ask_i, 1))
                axs[1].plot(hp.space(trades_i, 1))
                starting = eq[-1] if(idx != 0) else settings.init_bank
                axs[2].plot(hp.space(eq, 1))
                #axs[2].plot(hp.space(smoothed[idx], 1))
                plt.show()

        equity_curve = [i for eq in equity_curve for i in eq]
        bh_curve = [i for bh in bh_curve for i in bh]

        print_stats(settings.init_bank, equity_curve)


if __name__ == '__main__':
    main()

    


