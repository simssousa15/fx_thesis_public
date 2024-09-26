import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from datetime import timedelta
from datetime import datetime

### Calculate Features ###
def feature_calculation (data, time_frames = 0, final = False):
    #create empty dataframe
    features = pd.DataFrame()

    #Time Frames (applied to minute data)
    if time_frames == 0:
        # 6 min, 18 min, 1 hr, 2 hr, 6 hr, 18 hr, 2 days, 7 days, 20 days
        time_frames = np.array([0.1, 0.33, 1, 2,  6, 18, 18*3, 18*3*3, 18*3*3*3]) * 60

    for tf in time_frames:
        #Bollinger Bands Indicator
        tf = int(tf)
        if(tf == 0): continue
        features[f'bb_pband{tf}'] = ta.bbands(data['Close'], length=tf, std=2)[f'BBP_{tf}_2.0']
        
        #RSI Indicator
        features[f'rsi{tf}'] = ta.rsi(data['Close'], length=tf)

        #MACD Indicator
        features[f'macd{tf}'] = ta.macd(data['Close'], fast=2*tf, slow=tf, signal=int(2* tf/3))[f'MACD_{tf}_{2*tf}_{int(2*tf/3)}']

        #EmaCross
        features[f'ema{tf}'] = ta.ema(data['Close'], length=tf)

    features = features.bfill()
    feature_labels = features.columns
    
    #1week 12month
    #final_features = ['ema6', 'macd60', 'ema60', 'macd120', 'ema120', 'rsi360', 'macd360', 'ema360','bb_pband1080', 'rsi1080', 'macd1080', 'ema1080', 'bb_pband3240', 'rsi3240','macd3240', 'ema3240', 'bb_pband9720', 'rsi9720', 'macd9720', 'ema9720','bb_pband29160', 'rsi29160', 'macd29160', 'ema29160']
    
    #3day 15month
    #final_features = ['macd1080', 'rsi3240', 'macd3240', 'ema3240', 'bb_pband9720','macd9720', 'macd29160', 'ema29160']

    #week 16month
    final_features = ['ema6', 'macd60', 'ema60', 'rsi120', 'macd120', 'ema120', 'rsi360',
       'macd360', 'ema360', 'bb_pband1080', 'rsi1080', 'macd1080', 'ema1080',
       'bb_pband3240', 'rsi3240', 'macd3240', 'ema3240', 'bb_pband9720',
       'rsi9720', 'macd9720', 'ema9720', 'bb_pband29160', 'rsi29160',
       'macd29160', 'ema29160']
    
    #interval
    #final_features = ['rsi60', 'macd60', 'ema60', 'macd120', 'ema120', 'rsi360', 'macd360', 'ema360', 'bb_pband1080', 'rsi1080', 'macd1080', 'ema1080', 'bb_pband3240', 'rsi3240', 'macd3240', 'ema3240', 'bb_pband9720', 'rsi9720', 'macd9720', 'ema9720', 'bb_pband29160', 'rsi29160', 'macd29160', 'ema29160']

    if(final == True):
        feature_labels = final_features
        features = features[final_features]
    
    print("Features Calculated")
    return pd.concat([data, features], axis=1), feature_labels

# Split data ready for rolling window application
# fixed 0.8 0.2 training test ratio (check older versions for changeable ratio)
def rolling_window(data, training_size = 12, testing_size = 1, rols = 5):

    if rols == -1:
        parts = np.floor((len(data)/(60*24*30) - training_size) / testing_size)
    else:
        parts = rols
    
    print("####### Rolling window stats #######")
    print("Total Data Size: {:.3f} yrs".format(len(data) / (60*24*365)))
    print("Train size: {:.1f} months".format(training_size))
    print("Test size: {:.1f} months".format(testing_size))
    print(f"Num Parts: {parts}")
    print("-----------------------------------")

    split_data = []
    test_sz_sec = int(testing_size*(60*24*30))
    train_sz_sec =  int(training_size*(60*24*30))
    starting_pos = len(data) - train_sz_sec - test_sz_sec - 1
    for i in range(int(parts)):
        
        start = int(starting_pos * i/ parts)
        tr_end = start + train_sz_sec
        end = tr_end + test_sz_sec

        spl = []
        spl.append(data[start:tr_end])
        spl.append(data[tr_end:end])
        split_data.append(spl)
    
    print("Data Rolled") 

    """ for i in range(int(parts)):
        aux = data[int(i*len(data)/parts):int(len(data)*(i+1)/parts)]
        
        spl = []
        for j in range(len(frac) - 1):
            f = frac[j]
            f1 = frac[j+1]
            spl.append(aux[int(f*len(aux)) : int(f1*len(aux))])
        split_data.append(spl)
    print("Data Rolled") """

    return split_data

#code in pipeline a bit repetitive but should be no problem
#dropping unnecessary data from smaller timeframes (should be no big deal)
def variance(data, tf_dic):
    variance = pd.DataFrame()
    for tf_str, tf_int in tf_dic.items():
        variance[tf_str] = (data['Close'].shift(-tf_int) - data['Close']) * 100 / data['Close']
    print("Variance Calculated")
    return pd.concat([data, variance], axis = 1).dropna()

#Label set
#train_prepd - DICT label:time_frame values:features+targets
def targets(train, tf_dic, feat_lab):

    train_prepd = {}
    target = pd.DataFrame()

    for tf_str, tf_int in tf_dic.items():
        target[tf_str] = (train['Close'].shift(tf_int) - train['Close']) * 100/ train['Close']
        train_prepd[tf_str] = pd.concat([train['Gmt time'], train[feat_lab], target[tf_str]], axis=1)
        train_prepd[tf_str] = train_prepd[tf_str].dropna()

    return train_prepd

def sign(num):
    return 1 if num > 0 else -1 if num < 0 else 0

#Looking into the future so unfair
def trading_strat(predicts, dates, percent = 0.1):
    
    trades = pd.DataFrame()

    for tf_str, pred in predicts.items():
        sorted = np.sort(pred)
        limits = [sorted[int(percent*len(sorted))], sorted[int((1-percent)*len(sorted))]]

        if(sign(limits[0]) < 0 and sign(limits[1]) > 0):   
            trades[tf_str] = np.where(pred < limits[0], -1, 0)
            trades[tf_str] = np.where(pred > limits[1], 1, trades[tf_str])
        elif(sign(limits[1]) < 0):
            trades[tf_str] = np.where(pred <= limits[0], -1, 0)
        elif(sign(limits[0]) > 0):
            trades[tf_str] = np.where(pred >= limits[1], 1, 0)
    
    if(len(dates) == len(trades)):
        trades['Gmt time'] = dates.values
        trades.set_index('Gmt time', inplace=True)
    else:
        raise Exception("Dates and trades should have matching lengths (helper.py - trading_strat)")

    
    return trades

#Simple before threshold
def trading_strat2(predicts, dates, edge):
    
    trades = pd.DataFrame()

    for tf_str, pred in predicts.items():
        limits = edge[tf_str]

        if(sign(limits[0]) < 0  and sign(limits[1]) > 0):   
            trades[tf_str] = np.where(pred < limits[0], -1, 0)
            trades[tf_str] = np.where(pred > limits[1], 1, trades[tf_str])
        elif(sign(limits[1]) < 0):
            trades[tf_str] = np.where(pred <= limits[0], -1, 0)
        elif(sign(limits[0]) > 0):
            trades[tf_str] = np.where(pred >= limits[1], 1, 0)
        elif(sign(limits[0]) == 0  and sign(limits[1]) == 0):
            trades[tf_str] = np.where(pred > 0, 1, -1)

    
    if(len(dates) == len(trades)):
        trades['Gmt time'] = dates.values
        trades.set_index('Gmt time', inplace=True)
    else:
        raise Exception(f"Dates and trades should have matching lengths (dates:{len(dates)} vs trades:{len(trades)}) (helper.py - trading_strat)")

    
    return trades

#Updatable threshold
def trading_strat3(predicts, dates, edge, frac = 0.001):
    
    trades = pd.DataFrame()

    for tf_str, pred in predicts.items():
        limits = edge[tf_str]
        
        trades_tf = [] 
        for i in range(len(pred)):
            #can be optimized
            if(i % 60*24*1 == 0 and i != 0):
                limits = get_edge_vals(pred[0:i], frac)
            
            if(sign(limits[0]) < 0 and sign(limits[1]) > 0):   
                trades_tf.append(-1 if pred[i] < limits[0] else 1 if pred[i] > limits[1] else 0) 
            elif(sign(limits[1]) < 0):
                trades_tf.append(-1 if pred[i] <= limits[0] else 0) 
            elif(sign(limits[0]) > 0):
                trades_tf.append(1 if pred[i] >= limits[1] else 0) 
        
        trades[tf_str] = trades_tf      
    
    #print("Final: ", limits)
    
    if(len(dates) == len(trades)):
        trades['Gmt time'] = dates.values
        trades.set_index('Gmt time', inplace=True)
    else:
        raise Exception("Dates and trades should have matching lengths (helper.py - trading_strat)")

    
    return trades

def space(array, spacing):

    if spacing == 1:
        return array
    new_array = np.array(array)
    idxs = [0]
    for i in range(len(new_array)):
        if(i % spacing == 0):
            idxs.append(i)
    idxs.append(len(new_array) - 1)

    return new_array[idxs]

#very simplified leverage with 8% interest
#always same trade_size always same leverage
def equity(price_data, trades, duration, starting_bank = 10000, trade_size = 1, leverage = 1):
    
    ask = price_data[0]
    bid = price_data[1]
    
    interest = 0.08
    cost_leverage = trade_size * (leverage-1) * duration * interest/(365*24*60)

    bank = starting_bank
    hist = [starting_bank]
    staked = {'value': np.array([]),
              'dir': np.array([]),
              'age': np.array([])}

    i = 0
    while(i < len(trades) or len(staked['value']) != 0):
        if(i < len(trades)):
            t = trades[i]
            if(t!=0):
                if(bank - trade_size >= 0):
                    staked['value'] = np.append(staked['value'], trade_size * leverage * bid[i] / ask[i])
                    staked['age'] = np.append(staked['age'], -1)
                    staked['dir'] = np.append(staked['dir'], t)
                    bank -= trade_size
                #else:
                    #print(f'No funds for trade (idx {i})')

        #update staked values
        t_var = (bid[i] - bid[i - 1]) / bid[i-1] if(i!=0) else 0
        staked['value'] *= 1 + t_var * staked['dir']
        staked['age'] += 1
        
        #move finished trades to bank
        if(staked['value'][staked['age'] >= duration].sum() != 0):
            #paying cost of leverage at the end
            bank += staked['value'][staked['age'] >= duration].sum() - (leverage-1)*trade_size - cost_leverage
            staked['value'] = staked['value'][staked['age'] < duration]
            staked['dir'] = staked['dir'][staked['age'] < duration]
            staked['age'] = staked['age'][staked['age'] < duration]

        hist.append(bank + (staked['value'] - trade_size * (leverage - 1)).sum())
        i += 1
        
    return hist


def checkNumTrades(staked):
    num = 0
    for pair in list(staked.keys()):
        num += len(staked[pair]['value'])
    
    return num

#modeled as security trading. Trading in forex might be more complex (ex. Holding EUR and making a trade in GBPUSD)
#data {'pair': [ask, bid, trades]}
#All rols at the same time
def conc_equity(price_data, trades, tf, hist, settings, fail = 0, total = 0):

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
                'init': np.array([])}

    i = 0
    # Assume trades have  all the same size
    num_trades = len(trades[pairs[0]])
    bank = hist['total'][-1]

    while(i < num_trades or checkNumTrades(staked) > 0):
        for p in pairs:
            ask_p = ask[p]
            bid_p = bid[p]
            trades_p = trades[p][tf_str].values
            stake_p = staked[p]
            if(i < num_trades):
                t = trades_p[i]
                total[p] += abs(t)
                if(t != 0):
                    if(bank - settings.trade_sz >= 0):
                        stake_p['init'] = np.append(stake_p['init'], abs(t))
                        stake_p['value'] = np.append(stake_p['value'], abs(t) * settings.trade_sz * settings.lev * bid_p[i] / ask_p[i])
                        #stake_p['lev'] = np.append(stake_p['lev'], settings.lev)
                        stake_p['age'] = np.append(stake_p['age'], -1)
                        staked[p]['dir'] = np.append(stake_p['dir'], 1 if t > 0 else -1)
                        bank -= abs(t) * settings.trade_sz
                    else:
                        fail[p] += abs(t)
                        #print(f'No funds for trade (idx {i}, pair {p})')
            
            #update staked values
            t_var = (bid_p[i] - bid_p[i - 1]) / bid_p[i-1] if(i!=0) else 0
            stake_p['value'] *= 1 + t_var * stake_p['dir']
            stake_p['age'] += 1

            #move finished trades to bank
            if(stake_p['value'][stake_p['age'] >= tf_int].sum() != 0):
                #paying cost of leverage at the end
                hist[p] += stake_p['value'][stake_p['age'] >= tf_int].sum()
                hist[p] -= stake_p['init'][stake_p['age'] >= tf_int].sum() * settings.trade_sz
                bank += stake_p['value'][stake_p['age'] >= tf_int].sum() - (settings.lev-1)*settings.trade_sz - cost_leverage
                stake_p['value'] = stake_p['value'][stake_p['age'] < tf_int]
                stake_p['dir'] = stake_p['dir'][stake_p['age'] < tf_int]
                stake_p['init'] = stake_p['init'][stake_p['age'] < tf_int]
                stake_p['age'] = stake_p['age'][stake_p['age'] < tf_int]
        
        hist_i = bank
        for p in pairs:
            hist_i += (staked[p]['value'] - settings.trade_sz * (settings.lev - 1)).sum()
        
        hist['total'].append(hist_i)
        i += 1
            
    return hist, fail, total


def conc_equity_noFlats(price_data, volume, trades, tf, hist, settings):

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
    t_i_p = {p: 0 for p in pairs}
    # Assume trades have  all the same size
    num_trades = len(trades[pairs[0]])
    bank = hist[-1]
    t_i = 0
    
    while(t_i < num_trades or checkNumTrades(staked) > 0):
        #first iteration is fucked but it should be ok
        for p in pairs:
            ask_p = ask[p]
            bid_p = bid[p]
            volume_p = volume[p]
            t_i = t_i_p[p]
            if (volume_p[i-1] != 0):
                trades_p = trades[p][tf_str].values
                stake_p = staked[p]
                if(i < num_trades):
                    t = trades_p[t_i]
                    if(t != 0):
                        if(bank - settings.trade_sz >= 0):
                            stake_p['value'] = np.append(stake_p['value'], settings.trade_sz * settings.lev * bid_p[i] / ask_p[i])
                            stake_p['age'] = np.append(stake_p['age'], -1)
                            staked[p]['dir'] = np.append(stake_p['dir'], t)
                            bank -= settings.trade_sz
                        #else:
                            #print(f'No funds for trade (idx {i}, pair {p})')
                
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
                
                t_i_p[p] += 1
                
        hist_i = bank
        for p in pairs:
            hist_i += (staked[p]['value'] - settings.trade_sz * (settings.lev - 1)).sum()
        
        hist.append(hist_i)
        i += 1
            
            
    return hist


def stop_lev_conc_equity(price_data, trades, tf, hist, settings, fail=0, total=0):

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
    bank = hist['total'][-1]

    while(i < num_trades or checkNumTrades(staked) > 0):
        for p in pairs:
            ask_p = ask[p]
            bid_p = bid[p]
            trades_p = trades[p][tf_str].values
            stake_p = staked[p]
            if(i < num_trades):
                t = trades_p[i]
                total[p] += abs(t)
                if(t != 0):
                    if(bank - settings.trade_sz >= 0):
                        lev = abs(t) * settings.lev
                        stake_p['value'] = np.append(stake_p['value'], settings.trade_sz * lev * bid_p[i] / ask_p[i])
                        stake_p['lev'] = np.append(stake_p['lev'], lev)
                        stake_p['age'] = np.append(stake_p['age'], -1)
                        staked[p]['dir'] = np.append(stake_p['dir'], 1 if t > 0 else -1)
                        bank -= settings.trade_sz
                    else:
                        fail[p] += abs(t)
                        #print(f'No funds for trade (idx {i}, pair {p})')
            
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
                hist[p] += (stake_p['value'][stake_p['age'] >= tf_int] - (stake_p['lev'][stake_p['age'] >= tf_int]-1)*settings.trade_sz).sum()
                hist[p] -= len(stake_p['value'][stake_p['age'] >= tf_int]) * settings.trade_sz
                bank += (stake_p['value'][stake_p['age'] >= tf_int] - (stake_p['lev'][stake_p['age'] >= tf_int]-1)*settings.trade_sz).sum()
                stake_p['value'] = stake_p['value'][stake_p['age'] < tf_int]
                stake_p['dir'] = stake_p['dir'][stake_p['age'] < tf_int]
                stake_p['lev'] = stake_p['lev'][stake_p['age'] < tf_int]
                stake_p['age'] = stake_p['age'][stake_p['age'] < tf_int]

        
        hist_i = bank
        for p in pairs:
            hist_i += (staked[p]['value'] - settings.trade_sz * (staked[p]['lev'] - 1)).sum()
        
        hist['total'].append(hist_i)
        i += 1
            
    return hist, fail, total


def buy_hold(price_data, starting, first = False):
    ask = price_data[0]
    bid = price_data[1]

    hist = [starting]
    if(first):
        hist[0] *= bid[0] / ask[0]
    
    for i in range(1, len(bid)):
        var = bid[i] / bid[i-1]
        hist.append(hist[i-1] * var)

    return hist


def maxDrawDown(equity_curve): 
    
    eq = pd.Series(np.array(equity_curve).flatten())
    cumulative_max = eq.cummax()
    drawdown = round((cumulative_max - eq) * 100/ cumulative_max, 2)
    max_drawdown = drawdown.max()

    return max_drawdown

def date_parser(date):
    return datetime.strptime(date, "%d.%m.%Y %H:%M:%S.%f")

def minute_data_loading(pair, price = 'BID', filtered = False, interval = 1):

    if(filtered):
        dir = 'Filtered_Flats'
    else:
        dir = 'Unfiltered'

    minute1 = pd.read_csv(f'storage/Datasets/{dir}/{pair}/{pair}_Candlestick_1_M_{price}_01.01.2019-01.01.2022.csv',parse_dates=['Gmt time'], date_format="%d.%m.%Y %H:%M:%S.%f")
    minute2 = pd.read_csv(f'storage/Datasets/{dir}/{pair}/{pair}_Candlestick_1_M_{price}_01.01.2022-01.03.2024.csv',parse_dates=['Gmt time'], date_format="%d.%m.%Y %H:%M:%S.%f")
    minute = pd.concat([minute1, minute2])
    minute = minute.reset_index(drop=True)
    minute = minute.drop_duplicates(subset='Gmt time', keep='first')
    minute = minute.iloc[::interval, :]

    print(f'{pair} {price} {dir} Data Loaded')

    return minute

def get_edge_vals(array, frac = 0.01):
    sorted = np.sort(array)
    limits = [sorted[int(frac*len(sorted))], sorted[int((1-frac)*len(sorted))]]

    return limits


def get_edge_idx(array, frac = 0.01):

    abs_array = np.abs(array)
    #Sort in descending order
    sorted_abs_values = np.sort(abs_array)[::-1]
    #Find edge limit
    edge_value = sorted_abs_values[int(len(sorted_abs_values) * frac)]

    idxs = np.where((array >= edge_value) | (array <= -edge_value))[0]
    
    return idxs

def get_edge(y_true, y_pred,frac = 0.01):
    pred = np.array(y_pred)
    true = np.array(y_true)

    idxs = get_edge_idx(pred, frac)
    pred = pred[idxs]
    true = true[idxs]
    #print(f"EDGE: {len(pred)} / {len(y_pred)}")

    return true, pred

#Old trade score with the lookout for the month
def trade_score(true, pred, frac = 0.01):
    true, pred = get_edge(true, pred, frac)
    score = 0
    for i in range(len(true)):
        score += true[i] * sign(pred[i])
    score /= len(true)

    return score

def trade_score2(trades, data, tf_str = 'week', tf_int = 60*24*7):
    flat_trades = trades[0]
    for i in range(1, len(trades)):
        flat_trades = pd.concat([flat_trades, trades[i]])

    #print(flat_trades)
    full_data = pd.merge(data, flat_trades, on='Gmt time', how='left')
    full_data[f'1{tf_str}'] = full_data['Close'].shift(-tf_int)
    full_data['score'] = (full_data[f'1{tf_str}'] - full_data['Close']) / full_data['Close'] * full_data[tf_str]

    #find negative values in score
    """ total_trades = np.where(flat_trades[tf_str].values != 0)[0]
    wrong_trades = np.where(full_data['score'].values < 0)[0]
    print(f"Wrong trades: {len(wrong_trades)} / {len(total_trades)}") """

    score = full_data['score'].sum()
    
    return score

def get_start_finish (data, offset = 0):
    #datetimes = [datetime.strptime(dt, "%d.%m.%Y %H:%M:%S.%f") for dt in data.index]
    sorted_dt = sorted(data.index)

    return [sorted_dt[0], sorted_dt[-1] + timedelta(minutes=offset)]


if __name__ == "__main__":
    minute1 = pd.read_csv('/storage/Datasets/EUR|USD/EURUSD_Candlestick_1_M_BID_01.01.2019-01.01.2022.csv')
    rolled = rolling_window(minute1)

    print(f"({len(rolled)},{len(rolled[0])})")
    for i in range(len(rolled)):
        ratio = len(rolled[i][0]) / (len(rolled[i][2]) + len(rolled[i][1]) + len(rolled[i][0]))
        print(f"{i} - {int(ratio * 100)}%")