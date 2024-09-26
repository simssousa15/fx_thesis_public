
import helper as hp
import os
import pickle
from dataclasses import dataclass
import numpy as np
import helper as hp

#Icrease staked value when multiple models agree
def conc_equity(price_data, trades, tf, hist, settings):

    ask = price_data[0]
    bid = price_data[1]
    pairs = list(ask.keys())
    tf_str = list(tf.keys())[0]
    tf_int = tf[tf_str]

    interest = 0.08
    cost_leverage = settings.trade_sz * (settings.lev-1) * tf_int * interest/(365*24*60)

    staked = {}
    for pair in pairs:
        staked[pair] = {'value': np.array([]),
                'dir': np.array([]),
                'age': np.array([])}

    i = 0
    # Assume trades have  all the same size
    num_trades = len(trades[pairs[0]])
    bank = hist[-1]

    while(i < num_trades or hp.checkNumTrades(staked) > 0):
        for p in pairs:
            ask_p = ask[p]
            bid_p = bid[p]
            trades_p = trades[p][tf_str].values
            stake_p = staked[p]
            if(i < num_trades):
                t = trades_p[i]
                if(t != 0):
                    if(bank - settings.trade_sz >= 0):
                        stake_p['value'] = np.append(stake_p['value'], abs(t) * settings.trade_sz * settings.lev * bid_p[i] / ask_p[i])
                        #stake_p['lev'] = np.append(stake_p['lev'], settings.lev)
                        stake_p['age'] = np.append(stake_p['age'], -1)
                        staked[p]['dir'] = np.append(stake_p['dir'], 1 if t > 0 else -1)
                        bank -= abs(t) * settings.trade_sz
                    """ else:
                        print(f'No funds for trade (idx {i}, pair {p})') """
            
            #update staked values
            t_var = (bid_p[i] - bid_p[i - 1]) / bid_p[i-1] if(i!=0) else 0
            stake_p['value'] *= 1 + t_var * stake_p['dir']
            stake_p['age'] += 1

            #move finished trades to bank
            if(stake_p['value'][stake_p['age'] >= tf_int].sum() != 0):
                #paying cost of leverage at the end
                bank += stake_p['value'][stake_p['age'] >= tf_int].sum() - (settings.lev-1)*settings.trade_sz - cost_leverage
                stake_p['value'] = stake_p['value'][stake_p['age'] < tf_int]
                stake_p['dir'] = stake_p['dir'][stake_p['age'] < tf_int]
                stake_p['age'] = stake_p['age'][stake_p['age'] < tf_int]
        
        hist_i = bank
        for p in pairs:
            hist_i += (staked[p]['value'] - settings.trade_sz * (settings.lev - 1)).sum()
        
        hist.append(hist_i)
        i += 1
            
    return hist

#Increase leverage when multiple models agree (instead of incresing staked value)
def lev_conc_equity(price_data, trades, tf, hist, settings):

    ask = price_data[0]
    bid = price_data[1]
    pairs = list(ask.keys())
    tf_str = list(tf.keys())[0]
    tf_int = tf[tf_str]

    interest = 0.08
    cost_leverage = settings.trade_sz * (settings.lev-1) * tf_int * interest/(365*24*60)

    staked = {}
    for pair in pairs:
        staked[pair] = {'value': np.array([]),
                'dir': np.array([]),
                'age': np.array([]),
                'lev': np.array([])}

    i = 0
    # Assume trades have  all the same size
    num_trades = len(trades[pairs[0]])
    bank = hist[-1]

    while(i < num_trades or hp.checkNumTrades(staked) > 0):
        for p in pairs:
            ask_p = ask[p]
            bid_p = bid[p]
            trades_p = trades[p][tf_str].values
            stake_p = staked[p]
            if(i < num_trades):
                t = trades_p[i]
                if(t != 0):
                    if(bank - settings.trade_sz >= 0):
                        lev = abs(t) * settings.lev
                        stake_p['value'] = np.append(stake_p['value'], settings.trade_sz * lev * bid_p[i] / ask_p[i])
                        stake_p['lev'] = np.append(stake_p['lev'], lev)
                        stake_p['age'] = np.append(stake_p['age'], -1)
                        staked[p]['dir'] = np.append(stake_p['dir'], 1 if t > 0 else -1)
                        bank -= settings.trade_sz
                    """ else:
                        print(f'No funds for trade (idx {i}, pair {p})') """
            
            #update staked values
            t_var = (bid_p[i] - bid_p[i - 1]) / bid_p[i-1] if(i!=0) else 0
            stake_p['value'] *= 1 + (t_var * stake_p['dir'])
            stake_p['age'] += 1

            #move finished trades to bank
            if(stake_p['value'][stake_p['age'] >= tf_int].sum() != 0):
                #paying cost of leverage at the end
                #do not count for cost of leverage at the moment
                bank += (stake_p['value'][stake_p['age'] >= tf_int] - (stake_p['lev'][stake_p['age'] >= tf_int]-1)*settings.trade_sz).sum()
                stake_p['value'] = stake_p['value'][stake_p['age'] < tf_int]
                stake_p['dir'] = stake_p['dir'][stake_p['age'] < tf_int]
                stake_p['lev'] = stake_p['lev'][stake_p['age'] < tf_int]
                stake_p['age'] = stake_p['age'][stake_p['age'] < tf_int]

        
        hist_i = bank
        for p in pairs:
            hist_i += (staked[p]['value'] - settings.trade_sz * (staked[p]['lev'] - 1)).sum()
        
        hist.append(hist_i)
        i += 1
            
    return hist

#adding stop loss to the increased leverage strategy
def stop_lev_conc_equity(price_data, trades, tf, hist, settings):

    ask = price_data[0]
    bid = price_data[1]
    pairs = list(ask.keys())
    tf_str = list(tf.keys())[0]
    tf_int = tf[tf_str]

    interest = 0.08
    cost_leverage = settings.trade_sz * (settings.lev-1) * tf_int * interest/(365*24*60)

    staked = {}
    for pair in pairs:
        staked[pair] = {'value': np.array([]),
                'dir': np.array([]),
                'age': np.array([]),
                'lev': np.array([])}

    i = 0
    # Assume trades have  all the same size
    num_trades = len(trades[pairs[0]])
    bank = hist[-1]

    while(i < num_trades or hp.checkNumTrades(staked) > 0):
        for p in pairs:
            ask_p = ask[p]
            bid_p = bid[p]
            trades_p = trades[p][tf_str].values
            stake_p = staked[p]
            if(i < num_trades):
                t = trades_p[i]
                if(t != 0):
                    if(bank - settings.trade_sz >= 0):
                        lev = abs(t) * settings.lev
                        stake_p['value'] = np.append(stake_p['value'], settings.trade_sz * lev * bid_p[i] / ask_p[i])
                        stake_p['lev'] = np.append(stake_p['lev'], lev)
                        stake_p['age'] = np.append(stake_p['age'], -1)
                        staked[p]['dir'] = np.append(stake_p['dir'], 1 if t > 0 else -1)
                        bank -= settings.trade_sz
                    """ else:
                        print(f'No funds for trade (idx {i}, pair {p})') """
            
            #update staked values
            t_var = (bid_p[i] - bid_p[i - 1]) / bid_p[i-1] if(i!=0) else 0
            stake_p['value'] *= 1 + (t_var * stake_p['dir'])
            stake_p['age'] += 1

            #remove trades that hit stop loss (worst trade result = 0)
            ok_condition = stake_p['value']  > ((stake_p['lev'] - 1) * settings.trade_sz)
            if(ok_condition.sum() != stake_p['value'].shape[0]):
                print("Stop Loss Hit: " + ok_condition.sum() + " trades")
                bank += (stake_p['value'][~ok_condition] - (stake_p['lev'][~ok_condition]-1)*settings.trade_sz).sum()
                stake_p['value'] = stake_p['value'][ok_condition]
                stake_p['dir'] = stake_p['dir'][ok_condition]
                stake_p['lev'] = stake_p['lev'][ok_condition]
                stake_p['age'] = stake_p['age'][ok_condition]

            #move finished trades to bank
            if(stake_p['value'][stake_p['age'] >= tf_int].sum() != 0):
                #paying cost of leverage at the end
                #do not count for cost of leverage at the moment
                bank += (stake_p['value'][stake_p['age'] >= tf_int] - (stake_p['lev'][stake_p['age'] >= tf_int]-1)*settings.trade_sz).sum()
                stake_p['value'] = stake_p['value'][stake_p['age'] < tf_int]
                stake_p['dir'] = stake_p['dir'][stake_p['age'] < tf_int]
                stake_p['lev'] = stake_p['lev'][stake_p['age'] < tf_int]
                stake_p['age'] = stake_p['age'][stake_p['age'] < tf_int]

        
        hist_i = bank
        for p in pairs:
            hist_i += (staked[p]['value'] - settings.trade_sz * (staked[p]['lev'] - 1)).sum()
        
        hist.append(hist_i)
        i += 1
            
    return hist


def concurrent_trading(t_p):

    tf = {'week': 60*24*7}
    ask = {}
    trades = []

    for p, t in t_p.items():
        ask[p] = hp.minute_data_loading(p, 'ASK')
        t = t_p[p]
        if (trades == []):
            for i in t:
                trades.append({p: i})
        else:
            for idx, val in enumerate(t):
                trades[idx][p] = val


    hist = [10000]
    for rol in range(len(trades)):
        print(f"ROL {rol + 1} / {len(trades)}", end='\r')
        
        ask_i = {}
        trades_i = trades[rol]
        for p in t_p.keys():
            t = trades_i[p]
            ask_p = ask[p]
            start_date, finish_date = hp.get_start_finish(t, offset= 60*24*15)
            ask_i[p] = ask_p[(ask_p['Gmt time'] >= start_date) & (ask_p['Gmt time'] <= finish_date)]
            ask_i[p] = ask_i[p]['Close'].values

        @dataclass
        class setStruct:
            trade_sz: int
            lev: int
        settings = setStruct(trade_sz=5, lev=1)
        
        hist = stop_lev_conc_equity([ask_i, ask_i], trades_i, tf, hist, settings)

    return hist

def print_stats(eq):
    init_bank = 10000
    print("######## Stats ########")
    print("Initial Bank: ", init_bank)
    print("Final Bank: ", round(eq[-1],2))
    print("Time: {:.2f} months".format(len(eq) / (60*24*30)))
    roi = (eq[-1] - init_bank) / init_bank * 100
    print("ROI: {:.2f} %".format(roi))
    years = len(eq) / (60*24*365)
    anualized_roi = (eq[-1] / init_bank) ** (1/years) - 1
    print("Annualized ROI: {:.2f} %".format(anualized_roi * 100))
    print(f"MDD: {hp.maxDrawDown(eq)} %")
    return


### TRADE AGGREGATION FUNCTIONS ###
def sum_legacy(trades_total):
    fin_t = {}
    for pair, val in trades_total.items():
        fin_t[pair] = []
        n_rols = len(val[next(iter(val))])
        for r_idx in range(n_rols):
            curr_rol= []
            for algo, rol in val.items():
                if(len(curr_rol) == 0):
                    curr_rol = rol[r_idx]
                else:
                    curr_rol = curr_rol + rol[r_idx]
            fin_t[pair].append(curr_rol)
    
    print("Trade Sum Complete")
    return fin_t



def sum(trades):
    summed = []
    f = True
    for m, t in trades.items():
        if(f==True):
            summed = t
            f = False
        else:
            for i, rol in enumerate(t):
                summed[i] += rol
    
    return summed


def weighted_sum(trades, w):
    summed = []
    f = True
    for m, t in trades.items():
        if(f==True):
            summed = t
            f = False
        else:
            for i, rol in enumerate(t):
                if(m == 'xgb' or m == 'lgbm'):
                    summed[i] += rol
                else:
                    summed[i] += w * rol
    
    return summed

def min_score(trades, score, w):
    
    summed = weighted_sum(trades, w)
    
    for i, rol in enumerate(summed):
        rol[abs(rol) < score] = 0

    
    return summed

def majority(trades):
    
    summed = weighted_sum(trades, 1)
    
    for i, rol in enumerate(summed):
        rol[abs(rol) < 3] = 0
        rol[rol >= 3] = 1
        rol[rol <= -3] = -1
    
    return summed


tickers = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']

trades_total = {}

for p in tickers:
    xgb_dir = os.getcwd() + f"/storage/Pipelines/fin_xgb/unfiltered/{p}/train_size=16/rols=-1/week/"
    lgbm_dir = os.getcwd() + f"/storage/Pipelines/fin_lgb/unfiltered/{p}/train_size=16/rols=-1/week/"
    cat_dir = os.getcwd() + f"/storage/Pipelines/fin_cat/unfiltered/{p}/train_size=16/rols=-1/week/"
    sing_dir = os.getcwd() + f"/storage/Pipelines/fin_sing/unfiltered/{p}/train_size=16/rols=-1/week/"

    models = {  
                "xgb": xgb_dir, 
                "lgbm": lgbm_dir,
                "cat": cat_dir, 
                "sing": sing_dir    
            }

    
    print(p)
    t = {}
    for n, dir in models.items():
        with open(os.path.join(dir, "trades.pkl"), 'rb') as f:
            t[n] =  pickle.load(f)
    
    
    #Trades Joined#
    joined = majority(t)
    #print(summed)

    output_dir = os.getcwd() + f"/storage/Pipelines/fin_majority2.0/unfiltered/{p}/train_size=16/rols=-1/week/"
    #create directory if does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "trades.pkl"), 'wb') as f:
        pickle.dump(joined, f)




















####################
# LEGACY RESULTS
####################


# lgbm 7%
# xgboost 5.5%
# catboost 4.11%
# rbfBoost 2.5%

# linSVM (useless)

# xgb + lgbm 5.69% (straigth sum)
# xgb + lgbm + catboost 4.39% (straigth sum)

# Lev XGB + LGBM + CAT 11.67 %
# Lev XGB + LGBM 10.50 %
# Lev2 XGB + LGBM + CAT 19.69%

# Lev 2 XGB + LGBM + CAT 22.73% (stop loss)
# Lev 2 XGB + LGBM + CAT + RBF  22.73% (stop loss)
