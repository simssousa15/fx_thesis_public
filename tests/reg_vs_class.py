import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import helper as hp
from sklearn.metrics import confusion_matrix

def three_class(data, threshold):

    transformed = pd.Series(
    [1 if x > threshold else (-1 if x < -threshold else 0) for x in data]
    )

    return transformed

def rolling_reg_vs_class(data, t_sz, pair, rols):

    curr_dir = os.getcwd() +f"/storage/Pipelines/rolling_joint[xgb|lgbm|cat]/unfiltered/{pair}/train_size={t_sz}/rols={rols}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
        os.makedirs(curr_dir + "/sing")
    
    tf_dic = {'week': 60*24*7}
    
    data, feat_lab = hp.feature_calculation(data)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    data['1week'] = data['Close'].shift(-60*7*24)
    data['1week_var'] = (data['1week'] - data['Close']) / data['Close']

    threshold = 2.5
    # confusion matrices
    f_reg = 0
    f_clas = 0

    for i, rol in enumerate(rolled):
        #print("------------")
        print(f"ROL {i + 1} / {len(rolled)}", end="\r")
        train = rol[0]
        test = rol[1]

        train_prepd = hp.targets(train, tf_dic, feat_lab)
        
        tf = 'week'
        d = train_prepd[tf]
        
        
        test_var = pd.merge(test, data, on='Gmt time', how='left')
        test_var = test_var['1week_var']
        
        #### REGRESSION ####
        y_true = np.array([1 if x > 0 else -1 for x in test_var.values])

        dtrain = xgb.DMatrix(data = d[feat_lab], label = d[tf])
        params = {
            'device': 'cuda'
        }
        num_rounds = 10
        bst = xgb.train(params, dtrain, num_rounds )
        dtest = xgb.DMatrix(data=test[feat_lab])
        y_pred = bst.predict(dtest)
        y_pred = np.array([1 if x > threshold else (-1 if x < -threshold else 0) for x in y_pred])

        if(len(y_pred) != len(y_true)):
            print(f"ERROR CLASS: y_pred and y_true have different lengths ( {len(y_pred)} vs {len(y_true)} )")
            return

        indices = np.where(y_pred != 0)[0]
        y_pred = y_pred[indices]
        y_true = y_true[indices]

        #get accuracy
        if(len(y_true) != 0):
            cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
            if(f_reg == 0):
                f_reg = 1
                reg = cm
            else:
                reg += cm
        
        ##############

        #### 3 CLASS classification ####
        y_true = np.array([2 if x > 0 else 0 for x in test_var.values])
        lab = pd.Series([2 if x > threshold else (0 if x < -threshold else 1) for x in d[tf]])
        
        dtrain = xgb.DMatrix(data = d[feat_lab], label = lab)
        params = {
            'device': 'cuda',
            'objective': 'multi:softmax',
            'num_class': 3
        }
        num_rounds = 10
        bst = xgb.train(params, dtrain, num_rounds )
        dtest = xgb.DMatrix(data=test[feat_lab])
        y_pred = bst.predict(dtest)

        if(len(y_pred) != len(y_true)):
            print(f"ERROR CLASS: y_pred and y_true have different lengths ( {len(y_pred)} vs {len(y_true)} )")
            return
        
        indices = np.where(y_pred != 1)[0]
        y_pred = y_pred[indices]
        y_true = y_true[indices]

        #get accuracy
        if(len(y_true) != 0):
            cm = confusion_matrix(y_true, y_pred, labels=[0, 2])
            if(f_clas == 0):
                f_clas = 1
                clas = cm
            else:
                clas += cm
        ##############

    print(f"REGRESSION: {reg}")
    print(f"3 CLASS: {clas}")

    return


if __name__ == "__main__":

    pairs = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']    
    print("Runnig reg_vs_class.py")
    
    sizes = [12]
    for p in pairs:
        minute = hp.minute_data_loading(p, filtered=False)
        for s in sizes:
            rolling_reg_vs_class(minute, s, p, -1)

