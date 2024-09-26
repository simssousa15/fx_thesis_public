import helper as hp
import xgboost as xgb
import os


def tester_pipeline(data, t_sz, pair, rols):

    tf_dic = {'week': 60*24*7}
    tf_str = 'week'
    tf_int = 60*24*7

    data, feat_lab = hp.feature_calculation(data)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    #high importance in last_lim
    
    lims = [1, 1.5, 2, 2.5, 3]
    for l in lims:
        last_lim = [-l, l]
        print(last_lim)
        for i, rol in enumerate(rolled):
            #print("------------")
            #print(f"ROL {i + 1} / {len(rolled)}", end = "\r")
            train = rol[0]
            test = rol[1]
            train_prepd = hp.targets(train, tf_dic, feat_lab)
            predicts = {}
            

            d = train_prepd[tf_str]

            ############ TESTING SECTION ############
            #_train = d[:int(len(train) * 0.9)]
            #_val = d[int(len(train) * 0.9):]
                
            ###XGBOOST###
            #start = time.time()
            #print("XGBoost")
            dtrain = xgb.DMatrix(data = d[feat_lab], label = d[tf_str])
            params = {
                'device': 'cuda'
            }
            num_rounds = 10
            bst = xgb.train(params, dtrain, num_rounds )
            dtest = xgb.DMatrix(data=test[feat_lab])
            y_pred = bst.predict(dtest)
            predicts[tf_str] = y_pred
            edge_trades_xgb = hp.trading_strat2({tf_str : y_pred}, test['Gmt time'], {tf_str : last_lim})
            #print(f"Time: {time.time() - start:.2f}s")
            #print("Trade Percent: ", (edge_trades_xgb[tf].astype(bool).sum() / len(edge_trades_xgb)) * 100)
            ##############
                

        hp.equity()

    return


if __name__ == "__main__":
    pairs = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']
    
    print("Runnig pipeline.py")
    
    sizes = [16]
    for p in pairs:
        minute = hp.minute_data_loading(p, filtered=False)
        for s in sizes:
            tester_pipeline(minute, s, p, -1)
