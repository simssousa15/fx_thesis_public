import helper as hp
import pickle
from dataclasses import dataclass
import concurrent.futures



def print_stats(init_bank, eq, bh = False):
    print("######## Stats ########")
    #print("Initial Bank: ", init_bank)
    print("Final Bank: ", round(eq[-1],2))
    #print("Time: {:.2f} months".format(len(eq) / (60*24*30)))
    roi = (eq[-1] - init_bank) / init_bank * 100
    print("ROI: {:.2f} %".format(roi))
    years = len(eq) / (60*24*365)
    anualized_roi = (eq[-1] / init_bank) ** (1/years) - 1
    print("Annualized ROI: {:.2f} %".format(anualized_roi * 100))
    print(f"MDD: {hp.maxDrawDown(eq)} %")
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

def concurrent_trading(ask, pairs, tf_str, tf_int, train_size):
    #Alll the rols at the same time
    #Assume timeframe only one to begin with
    tf = {tf_str: tf_int}

    trades = []
    for p in pairs:
        trade_file = "/home/simao/Documents/IST/Code/financial" + f"/storage/Pipelines/tf_and_tradeSize3/unfiltered/{p}/train_size={train_size}/rols=-1/{tf_str}/trades.pkl" 
        with open(trade_file, 'rb') as f:
            t = pickle.load(f)
        #print(f"tf_and_tradeSize {p} trades loaded")

        if (trades == []):
            for i in t:
                trades.append({p: i})
        else:
            for idx, val in enumerate(t):
                trades[idx][p] = val

    #Assume all trades have the same number of rols
    hist = [10000]
    for rol in range(len(trades)):
        #print(f"ROL {rol + 1} / {len(trades)}", end='\r')
        
        ask_i = {}
        trades_i = trades[rol]
        for p in pairs:
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

        hist = hp.stop_lev_conc_equity([ask_i, ask_i], trades_i, tf, hist, settings)

    return hist


pairs = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']
    

sizes = [16]
tf_dic = {'3hour': 60 * 3, '2day': 60 * 24, 'week': 60 * 24 * 7}

ask = {}

for p in pairs:
    ask[p] = hp.minute_data_loading(p, 'ASK', filtered = False)

def process_hist(s, tf_str, tf_int):
    hist = concurrent_trading(ask, pairs, tf_str, tf_int, s)
    print("######## Stats ########")
    print(f"TrainSize: {s}, TF: {tf_str}")
    eq = hist
    init_bank = 10000
    print("Final Bank: ", round(eq[-1],2))
    roi = (eq[-1] - init_bank) / init_bank * 100
    print("ROI: {:.2f} %".format(roi))
    years = len(eq) / (60*24*365)
    anualized_roi = (eq[-1] / init_bank) ** (1/years) - 1
    print("Annualized ROI: {:.2f} %".format(anualized_roi * 100))
    print(f"MDD: {hp.maxDrawDown(eq)} %")


# Use ThreadPoolExecutor to parallelize the processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for s in sizes:
        for tf_str, tf_int in tf_dic.items():
            # Schedule the process_hist function to be executed for each combination of s and tf_str
            future = executor.submit(process_hist, s, tf_str, tf_int)