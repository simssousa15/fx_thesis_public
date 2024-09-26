import helper as hp
import xgboost as xgb
import os
import pandas as pd
import numpy as np

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

    
    tf_dic = {'hour': 60, '3hour': 60*3, 'day': 60*24, '3day': 60*24*3, 'week': 60*24*7}
    pairs = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']

    for p in pairs:
        temp_data_file = f'storage/{p}_temp_data.parquet'
        feat_file = f'storage/{p}_feat_lab.npy'
    
        if os.path.exists(temp_data_file) and os.path.exists(feat_file):
            data = pd.read_parquet(temp_data_file)
            og_feat_lab = np.load(feat_file, allow_pickle=True)
            print(temp_data_file + 'and' + feat_file + 'loaded')
        else:
            data = hp.minute_data_loading(pair=p, filtered=False)
            data, og_feat_lab = hp.feature_calculation(data)
            data = hp.variance(data, tf_dic)
            data.to_parquet(temp_data_file, compression='snappy')
            np.save(feat_file, og_feat_lab, allow_pickle=True)
            print("Data saved to " + temp_data_file + 'and' + feat_file)

        rolled = hp.rolling_window(data, training_size = 12, testing_size = 1, rols = -1)

        
        idx = list(tf_dic.keys())
        cols = ['Initial TradeScore', 'Best TradeScore', 'Feature Set']
        stats = pd.DataFrame(index=idx, columns=cols)
        
        tf_str = 'week'
        tf_int = 60*24*7
        
        feat_lab = og_feat_lab.copy()
        benchmark = -np.inf
        best_feat = []
        
        print(f"#######\n{p} {tf_str}\n#######")
        while(len(feat_lab) > 0):
            w_feat = {}
            r = 0
            edge_trades_total = []
            last_lim = [-2.5, 2.5]
            for i, rol in enumerate(rolled):
                print(f"ROL {i+1} / {len(rolled)}", end="\r")
                train = rol[0]
                train = train.drop(columns = list(tf_dic.keys()))
                test = rol[1]

                train_prepd = hp.targets(train, {tf_str:tf_int}, feat_lab)

                d = list(train_prepd.values())[0]
                
                dtrain = xgb.DMatrix(data = d[feat_lab], label=d[tf_str])
                params = {
                    'device': 'cuda'
                }
                num_rounds = 10
                bst = xgb.train(params, dtrain, num_rounds)
                feat_imp = bst.get_fscore()
                merge_dicts(w_feat, feat_imp)
                
                dtest = xgb.DMatrix(data=test[feat_lab])
                y_pred = bst.predict(dtest)
                
                edge_trades = hp.trading_strat2({"week": y_pred}, test['Gmt time'], {"week": last_lim})
                edge_trades_total.append(edge_trades)
            
            score = hp.trade_score2(edge_trades_total, data)
            print(f"Trade Score: {score}")

            if(benchmark == -np.inf):
                benchmark = r
                stats.at[tf_str, 'Initial Trade Score'] = r
                best_feat = feat_lab
            elif(benchmark > r):
                benchmark = r
                best_feat = feat_lab
            
            sorted_w_feat = dict(sorted(w_feat.items(), key=lambda x: x[1]))
            feat_del = list(sorted_w_feat.keys())[0]
            mask = ~np.isin(feat_lab, feat_del)
            feat_lab = feat_lab[mask]
            print(f"Feature {feat_del} removed")
        
        stats.at[tf_str, 'Best Trade Score'] = benchmark
        stats.at[tf_str, 'Feature Set'] = best_feat
    
    stats.to_csv('TradeScore_feature_eliminator.csv')