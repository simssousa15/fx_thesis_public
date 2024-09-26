from tabnanny import verbose
from tracemalloc import start
from turtle import st
import pandas as pd
import helper as hp
import xgboost as xgb
import os
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import concurrent.futures
from functools import partial
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)

""" def edge_MSE(y_true, y_pred,frac = 0.01):
    edge_true, edge_preds = hp.get_edge(y_true, y_pred, frac)
    return MSE(edge_true, edge_preds)

###############################################################################
### MSE FUNCTIONS ###
'''Train using Python implementation of Mean Squared Error.'''
def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient for Mean Squared Error.'''
    
    y = dtrain.get_label()
    #print(predt)
    grad = (predt - y)

    #higher for edge cases
    grad[hp.get_edge_idx(predt)] *= 20

    return grad

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for Mean Squared Error.'''
    return np.ones_like(predt)

def mse(predt: np.ndarray,
        dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Mean Squared Error objective.'''
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

def compute_mse(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    '''Mean Squared Error metric.'''
    y = dtrain.get_label()
    elements = np.power(predt - y, 2)
    elements = elements[hp.get_edge_idx(predt)]
    return 'PyMSE', float(np.sqrt(np.sum(elements) / len(y))) """
###############################################################################


def example_pipeline(data, pair):

    curr_dir = os.getcwd() +f"/storage/Pipelines/example_pipline/{pair}"
    try:
        os.makedirs(curr_dir)
    except OSError:
        pass
    
    tf_dic = {'hour': 60, '3hour': 60*3, 'day': 60*24, '3day': 60*24*3, 'week': 60*24*7}
    #tf_dic = {'hour': 60, '3hour': 60*3}

    data, feat_lab = hp.feature_calculation(data)
    data = hp.variance(data, tf_dic)
    rolled = hp.rolling_window(data)
    

    tests = []
    trades_total = []
    preds_total = []
    for i, rol in enumerate(rolled):
        print("------------")
        print(f"ROL {i + 1} / {len(rolled)}")
        train = rol[0]
        train = train.drop(columns = list(tf_dic.keys()))
        test = rol[1]
        tests.append(test)

        train_prepd = hp.targets(train, tf_dic, feat_lab)
        print("Data Preped")
        predicts = {}
        for tf, d in train_prepd.items():

            dtrain = xgb.DMatrix(data = d[feat_lab], label=d[tf])
            params = {
                'device': 'cuda'
            }
            num_rounds = 10
            bst = xgb.train(params, dtrain, num_rounds)

            dtest = xgb.DMatrix(data=test[feat_lab])
            y_pred = bst.predict(dtest)
            
            predicts[tf] = y_pred
            print(f"{tf} model -> MAPE = {round(MAPE(test[tf], y_pred), 2):e} / R2 = {round(R2(test[tf], y_pred),2)}")
        
        preds_total.append(predicts)
        #set up trading strategy
        trades = hp.trading_strat(predicts, percent=0.01)
        trades_total.append(trades)
    
    print("-------------")

    with open(curr_dir + "/tests.pkl", 'wb') as f:
        pickle.dump(tests, f)
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(trades_total, f)
    with open(curr_dir + "/preds.pkl", 'wb') as f:
        pickle.dump(preds_total, f)
    
    
    
    return

def feature_pipeline(data, feat_csv, pair):

    eliminator = pd.read_csv(feat_csv)
    eliminator.set_index(eliminator.columns[0], inplace=True) 
    eliminator.index.name = 'tf_str'

    feat_el = eliminator['Feature Set']

    new_feat = {}
    for idx in feat_el.index:
        new_feat[idx] = [x.strip("'") for x in feat_el[idx].strip("[]").split()]
    print(new_feat)

    curr_dir = os.getcwd() + f"/storage/Pipelines/feature_pipline/{pair}"
    try:
        os.makedirs(curr_dir)
    except OSError:
        pass
    
    tf_dic = {'hour': 60, '3hour': 60*3, 'day': 60*24, '3day': 60*24*3, 'week': 60*24*7}
    #tf_dic = {'hour': 60, '3hour': 60*3}

    
    data, feat_lab = hp.feature_calculation(data)
    data = hp.variance(data, tf_dic)
    rolled = hp.rolling_window(data, rols=-1)
    

    tests = []
    trades_total = []
    preds_total = []
    for i, rol in enumerate(rolled):
        print("------------")
        print(f"ROL {i + 1} / {len(rolled)}")
        train = rol[0]
        train = train.drop(columns = list(tf_dic.keys()))
        test = rol[1]
        tests.append(test)

        train_prepd = hp.targets(train, tf_dic, feat_lab)
        print("Data Preped")
        predicts = {}
        for tf, d in train_prepd.items():

            dtrain = xgb.DMatrix(data = d[new_feat[tf]], label=d[tf])
            params = {
                'device': 'cuda'
            }
            num_rounds = 10
            bst = xgb.train(params, dtrain, num_rounds)

            dtest = xgb.DMatrix(data=test[new_feat[tf]])
            y_pred = bst.predict(dtest)
            
            predicts[tf] = y_pred
            print(f"{tf} model -> MAPE = {round(MAPE(test[tf], y_pred), 2):e} / R2 = {round(R2(test[tf], y_pred),2)}")
        
        preds_total.append(predicts)
        #set up trading strategy
        trades = hp.trading_strat(predicts,  test['Gmt time'], percent=0.01)
        trades_total.append(trades)
    
    print("-------------")

    """ with open(curr_dir + "/tests.pkl", 'wb') as f:
        pickle.dump(tests, f) """
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(trades_total, f)
    """ with open(curr_dir + "/preds.pkl", 'wb') as f:
        pickle.dump(preds_total, f) """
    
    return

def CS2_rolling_pipeline(data, t_sz, pair, rols):

    curr_dir = os.getcwd() +f"/storage/Pipelines/rolling2_pipline/unfiltered/{pair}/train_size={t_sz}/rols={rols}"
    try:
        os.makedirs(curr_dir)
    except OSError:
        pass
    
    
    #tf_dic = {'3day': 60*24*3, 'week': 60*24*7, '3week': 60*24*7*3, 'month': 60*24*30}
    tf_dic = {'week': 60*24*7}
    
    data, feat_lab = hp.feature_calculation(data)
    data = hp.variance(data, tf_dic)
    rolled = hp.rolling_window(data, t_sz, rols = rols)


    preds_trades_total = []
    median_trades_total = []
    edge_trades_total = []

    total_true_edge = []
    total_preds_edge = []
    total_median = []
    total_preds = []
    total_true = []

    #Median Comparison
    mae_preds = 0
    mae_median = 0
    mae_edge = 0
    trad_score_preds = 0
    trad_score_median = 0

    for i, rol in enumerate(rolled):
        #print("------------")
        #print(f"ROL {i + 1} / {len(rolled)}", end = "\r")
        train = rol[0]
        train = train.drop(columns = list(tf_dic.keys()))
        test = rol[1]
        
        train_prepd = hp.targets(train, tf_dic, feat_lab)
        #print("Data Preped") 
        predicts = {}
        median_predicts = {}
        for tf, d in train_prepd.items():

            dtrain = xgb.DMatrix(data = d[feat_lab], label=d[tf])
            params = {
                'device': 'cuda',
            }
            num_rounds = 10
            bst = xgb.train(params, dtrain, num_rounds)

            #bst = xgb.train(params, dtrain, num_rounds)
            dtest = xgb.DMatrix(data=test[feat_lab])
            y_pred = bst.predict(dtest)
            

            #Median comparisson
            median = np.median(d[tf].values)
            median_arr = np.full(len(y_pred), median)

            predicts[tf] = y_pred
            median_predicts[tf] = median_arr

            """ out = f"{tf} model:\n"
            out += f"MAPE = {MAPE(test[tf], y_pred):2e} \n"
            out += f"R2 = {R2(test[tf], y_pred):.2f} \n"

            out += f"Edge MAE: {MAE(edge_true, edge_preds):.2f}\n"
            out += f"Mean Edge: pred: {np.mean(np.abs(edge_preds)):.2f} "
            out += f"| real: {np.mean(np.abs(edge_true)):.2f}" """
        
            #Median comparison
            mae_preds += MAE(y_pred, test[tf].values)
            mae_median += MAE(median_arr, test[tf].values)
            edge_preds, edge_true = hp.get_edge(y_pred, test[tf].values, frac = 0.01)
            mae_edge += MAE(edge_true, edge_preds)
            trad_score_preds += hp.trade_score(test[tf].values, y_pred, frac = 0.01)
            trad_score_median += hp.trade_score(test[tf].values, median_arr, frac = 0.5)

        #set up trading strategy
        edge_trades = hp.trading_strat(predicts, test['Gmt time'], percent = 0.01)
        preds_trades = hp.trading_strat(predicts, test['Gmt time'], percent = 0.5)
        median_trades = hp.trading_strat(median_predicts, test['Gmt time'], percent=0.5)

        preds_trades_total.append(preds_trades)
        edge_trades_total.append(edge_trades)
        median_trades_total.append(median_trades)

        edge_true, edge_preds = hp.get_edge(test[tf].values, y_pred, frac = 0.01)
        total_true_edge.append(edge_true)
        total_preds_edge.append(edge_preds)
        total_median.append(median_arr)
        total_preds.append(y_pred)
        total_true.append(test[tf].values)


    mae_preds /= (len(tf_dic) * len(rolled))
    mae_median /= (len(tf_dic) * len(rolled))
    mae_edge /= (len(tf_dic) * len(rolled))
    trad_score_preds /= (len(tf_dic) * len(rolled))
    trad_score_median /= (len(tf_dic) * len(rolled))

    
    print("MAE PRED: {:.2f}".format(mae_preds))
    total_preds = np.array([i for j in total_preds for i in j])
    total_true = np.array([i for j in total_true for i in j])
    class_preds = np.where(total_preds > 0, 1, -1)
    class_true = np.where(total_true > 0, 1, -1)
    cm = confusion_matrix(class_true, class_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    print("MAE EDGE PRED: {:.2f}".format(mae_edge))
    total_preds_edge = np.array([i for j in total_preds_edge for i in j])
    total_true_edge = np.array([i for j in total_true_edge for i in j])
    class_preds = np.where(total_preds_edge > 0, 1, -1)
    class_true = np.where(total_true_edge > 0, 1, -1)
    cm = confusion_matrix(class_true, class_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    print("MAE MEDIAN: {:.2f}".format(mae_median))
    total_median = np.array([i for j in total_median for i in j])
    class_preds = np.where(total_median > 0, 1, -1)
    class_true = np.where(total_true > 0, 1, -1)
    cm = confusion_matrix(class_true, class_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    print("Trade Score (preds): {:.2f}".format(trad_score_preds))
    print("Trade Score (median): {:.2f}".format(trad_score_median))
    
    #Plotting (total_true_edge sorted)
    """ indices = np.argsort(total_true_edge)
    true_plt = total_true_edge[indices]
    preds_plt = total_preds_edge[indices]

    x = [i for i in range(len(indices))]
    plt.plot(x, true_plt, label='Real')
    plt.plot(x, preds_plt, label='Prediction', marker='.', linestyle='None', markersize=3)
    plt.ylabel('Variance')
    plt.title('Edge preditions visualizer')
    plt.legend()
    plt.show() """
    

    #Confusion Matrix
    """ class_preds = np.where(total_preds_edge > 0, 1, -1)
    class_true = np.where(total_true_edge > 0, 1, -1)

    cm = confusion_matrix(class_true, class_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print(cm) """

        
    
    print("-------------")

    with open(os.getcwd() + f"/storage/caseStudy2/{p}/edge_trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)
    
    with open(os.getcwd() + f"/storage/caseStudy2/{p}/preds_trades.pkl", 'wb') as f:
        pickle.dump(preds_trades_total, f)
    
    with open(os.getcwd() + f"/storage/caseStudy2/{p}/median_trades.pkl", 'wb') as f:
        pickle.dump(median_trades_total, f)
       
    return

def rolling_pipeline(data, t_sz, pair, rols):

    curr_dir = os.getcwd() +f"/storage/Pipelines/rolling2/unfiltered/{pair}/train_size={t_sz}/rols={rols}"
    try:
        os.makedirs(curr_dir)
    except OSError:
        pass
    
    
    #tf_dic = {'3day': 60*24*3, 'week': 60*24*7, '3week': 60*24*7*3, 'month': 60*24*30}
    tf_dic = {'week': 60*24*7}
    
    data, feat_lab = hp.feature_calculation(data)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    last_lim = [-2.5, 2.5]
    #high importance in last_lim
    
    for i, rol in enumerate(rolled):
        #print("------------")
        print(f"ROL {i + 1} / {len(rolled)}", end = "\r")
        train = rol[0]
        test = rol[1]
        train_prepd = hp.targets(train, tf_dic, feat_lab)
        predicts = {}
        edge_rol = {}
        
        #print(test)
        for tf, d in train_prepd.items():
            dtrain = xgb.DMatrix(data = d[feat_lab], label=d[tf])
            params = {
                'device': 'cuda',
            }
            num_rounds = 10
            bst = xgb.train(params, dtrain, num_rounds)

            #bst = xgb.train(params, dtrain, num_rounds)
            dtest = xgb.DMatrix(data=test[feat_lab])
            y_pred = bst.predict(dtest)
            predicts[tf] = y_pred
            
            
            ################### !!! ####################
             #LAST LIM NOT PREPARED FOR MORE THAN 1 TF#
            ################### !!! #######v############
            #edge_rol[tf] = hp.get_edge_vals(bst.predict(dtrain))
            edge_rol[tf] = last_lim
            
        
        edge_trades = hp.trading_strat2(predicts, test['Gmt time'], edge_rol)
        edge_trades_total.append(edge_trades)
        
        
    #score
    strt = time.time()
    score = hp.trade_score2(edge_trades_total, data)    
    print(f"Trade Score: {score}")
    duration = time.time() - strt
    print(f"Time taken: {duration:.3f} seconds")
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)

    return

def rolling_joint_pipeline(data, t_sz, pair, rols):

    curr_dir = os.getcwd() +f"/storage/Pipelines/rolling2_singTree/unfiltered/{pair}/train_size={t_sz}/rols={rols}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
        os.makedirs(curr_dir + "/sing")

    #Load Trades from LGBM and XGB
    """ with open(os.getcwd() + f"/storage/Pipelines/rolling_joint[xgb|lgbm|cat]/unfiltered/{pair}/train_size={t_sz}/rols={rols}/trades.pkl", 'rb') as f:
        loaded_trades = pickle.load(f)
    print("Trades Loaded") """
    
    #tf_dic = {'3day': 60*24*3, 'week': 60*24*7, '3week': 60*24*7*3, 'month': 60*24*30}
    tf_dic = {'week': 60*24*7}
    
    data, feat_lab = hp.feature_calculation(data, final=False)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []
    xgb_trades = []
    lgbm_trades = []
    cat_trades = []

    #still some issues with the last lim
    last_lim = [-2.5, 2.5]
    #high importance in last_lim
    
    """ if(len(loaded_trades) != len(rolled)):
        print("Loaded trades and rolled data length mismatch")
        return """
    
    for i, rol in enumerate(rolled):
        #print("------------")
        print(f"ROL {i + 1} / {len(rolled)}")
        train = rol[0]
        test = rol[1]
        train_prepd = hp.targets(train, tf_dic, feat_lab)
        predicts = {}
        edge_rol = {}
        
        #print(test)
        for tf, d in train_prepd.items():
            
            ############ TESTING SECTION ############
            #_train = d[:int(len(train) * 0.9)]
            #_val = d[int(len(train) * 0.9):]
            
            ###XGBOOST###
            """ start = time.time()
            print("XGBoost")
            dtrain = xgb.DMatrix(data = d[feat_lab], label = d[tf])
            params = {
                'device': 'cuda'
            }
            num_rounds = 10
            bst = xgb.train(params, dtrain, num_rounds )
            dtest = xgb.DMatrix(data=test[feat_lab])
            y_pred = bst.predict(dtest)
            predicts[tf] = y_pred
            edge_trades_xgb = hp.trading_strat2({"week" : y_pred}, test['Gmt time'], {"week" : last_lim})
            print(f"Time: {time.time() - start:.2f}s")
            print("Trade Percent: ", (edge_trades_xgb['week'].astype(bool).sum() / len(edge_trades_xgb)) * 100)
            score_xg = hp.trade_score2([edge_trades_xgb], data)
            print(f"Score: {score_xg:.2f}") """
            ##############
            
            ###LigthGBM###
            """ params = {
                'device': 'gpu',
                'verbose': -1
            }
            start = time.time()
            print("LigthGBM")
            train_data = lgb.Dataset(d[feat_lab], label=d[tf], params=params)
            bst = lgb.train(params, train_data, 100)
            y_pred = bst.predict(test[feat_lab], num_iteration=bst.best_iteration)
            predicts[tf] = y_pred
            edge_trades_lgb = hp.trading_strat2({"week" : y_pred}, test['Gmt time'], {"week" : last_lim})
            print(f"Time: {time.time() - start:.2f}s") """
            #print("Trade Percent: ", (edge_trades_lgb['week'].astype(bool).sum() / len(edge_trades_lgb)) * 100)
            #score_lgb = hp.trade_score2([edge_trades_lgb], data)
            #print(f"Score: {score_rfg:.2f}")
            ##################
            
            ###CatBoostRegressor###
            """ start = time.time()
            print("CatBoostRegressor")
            model = CatBoostRegressor(task_type="CPU", thread_count=-1, verbose=False)
            model.fit(d[feat_lab], d[tf])
            y_pred = model.predict(test[feat_lab])
            edge_trades_cat = hp.trading_strat2({"week" : y_pred}, test['Gmt time'], {"week" : last_lim})
            print(f"Time: {time.time() - start:.2f}s") """
            #print("Trade Percent: ", (edge_trades_cat['week'].astype(bool).sum() / len(edge_trades_cat)) * 100)
            #score_cat = hp.trade_score2([edge_trades_cat], data)

            ###SingTreeRegressor###
            start = time.time()
            model = DecisionTreeRegressor()
            model.fit(d[feat_lab], d[tf])
            y_pred = model.predict(test[feat_lab])
            edge_trades_sing = hp.trading_strat2({"week" : y_pred}, test['Gmt time'], {"week" : last_lim})
            print(f"Time: {time.time() - start:.2f}s")
            ####################################################################################################
            
        
        #edge_trades = edge_trades_xgb + edge_trades_lgb + edge_trades_cat
        edge_trades = edge_trades_sing
        edge_trades_total.append(edge_trades)

        """ xgb_trades.append(edge_trades_xgb)
        lgbm_trades.append(edge_trades_lgb)
        cat_trades.append(edge_trades_cat)
        linSVM_trades.append(edge_trades_linSVM) """
        #rbfSVM_trades.append(edge_trades_rbfSVM)
        #kbb_trades.append(edge_trades_knn)
        
        
    #score
    """ strt = time.time()
    score = hp.trade_score2(edge_trades_total, data)    
    print(f"Trade Score: {score}")
    duration = time.time() - strt
    print(f"Time taken: {duration:.3f} seconds") """
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)

    with open(curr_dir + "/sing/xgboost.pkl", "wb") as f:
        pickle.dump(xgb_trades, f)
    
    with open(curr_dir + "/sing/lgbm.pkl", "wb") as f:
        pickle.dump(lgbm_trades, f)
    
    with open(curr_dir + "/sing/catboost.pkl", "wb") as f:
        pickle.dump(cat_trades, f)
    

    return


def tester_pipeline(data, t_sz, pair, rols):

    curr_dir = os.getcwd() +f"/storage/Pipelines/rolling_joint[xgb|lgbm|cat]/unfiltered/{pair}/train_size={t_sz}/rols={rols}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
        os.makedirs(curr_dir + "/sing")

    tf_dic = {'week': 60*24*7}
    
    data, feat_lab = hp.feature_calculation(data)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    #high importance in last_lim
    
    lims = [0.75, 1.5, 2.5, 5]
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
            
            tf = 'week'
            d = train_prepd[tf]

            ############ TESTING SECTION ############
            #_train = d[:int(len(train) * 0.9)]
            #_val = d[int(len(train) * 0.9):]
                
            ###XGBOOST###
            #start = time.time()
            #print("XGBoost")
            dtrain = xgb.DMatrix(data = d[feat_lab], label = d[tf])
            params = {
                'device': 'cuda'
            }
            num_rounds = 10
            bst = xgb.train(params, dtrain, num_rounds )
            dtest = xgb.DMatrix(data=test[feat_lab])
            y_pred = bst.predict(dtest)
            predicts[tf] = y_pred
            edge_trades_xgb = hp.trading_strat2({tf : y_pred}, test['Gmt time'], {tf : last_lim})
            #print(f"Time: {time.time() - start:.2f}s")
            #print("Trade Percent: ", (edge_trades_xgb[tf].astype(bool).sum() / len(edge_trades_xgb)) * 100)
            ##############
                
            
            edge_trades = edge_trades_xgb
            edge_trades_total.append(edge_trades)
        
        score_xg = hp.trade_score2(edge_trades_total, data)
        print(f"Score: {score_xg:.2f}")
    


    return

def final_pipeline(data, t_sz, pair, rols):

    curr_dir = os.getcwd() +f"/storage/Pipelines/rolling_caseStudy[xgb]/unfiltered/{pair}/train_size={t_sz}/rols={rols}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
        os.makedirs(curr_dir + "/sing")

    tf_dic = {'3day': 60*24*3}
    tf_str = '3day'
    tf_int = 60*24*3

    data, feat_lab = hp.feature_calculation(data)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    #high importance in last_lim

    last_lim = [-0.5, 0.5]
    
    for i, rol in enumerate(rolled):
        #print("------------")
        print(f"ROL {i + 1} / {len(rolled)}", end = "\r")
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
            
        
        edge_trades = edge_trades_xgb
        edge_trades_total.append(edge_trades)
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)
    
    return

def parallel_pipeline(data, t_sz, pair, rols):
    curr_dir = os.getcwd() +f"/storage/Pipelines/rolling2_singTree/unfiltered/{pair}/train_size={t_sz}/rols={rols}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
        os.makedirs(curr_dir + "/sing")

    #Load Trades from LGBM and XGB
    """ with open(os.getcwd() + f"/storage/Pipelines/rolling_joint[xgb|lgbm|cat]/unfiltered/{pair}/train_size={t_sz}/rols={rols}/trades.pkl", 'rb') as f:
        loaded_trades = pickle.load(f)
    print("Trades Loaded") """
    
    #tf_dic = {'3day': 60*24*3, 'week': 60*24*7, '3week': 60*24*7*3, 'month': 60*24*30}
    tf_dic = {'week': 60*24*7}
    tf_str = 'week'
    tf_int = 60*24*7

    data, feat_lab = hp.feature_calculation(data, final=False)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    last_lim = [-2.5, 2.5]
    #high importance in last_lim
    
    """ if(len(loaded_trades) != len(rolled)):
        print("Loaded trades and rolled data length mismatch")
        return """
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map keeps results in order
        func = partial(p_rol, feat_lab=feat_lab, last_lim=last_lim)
        edge_trades_total = list(executor.map(func, rolled))
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)

def p_rol(rol, feat_lab, last_lim):
    #print("------------")
    tf_dic = {'week': 60*24*7}
    tf_str = 'week'
    tf_int = 60*24*7

    train = rol[0]
    test = rol[1]
    train_prepd = hp.targets(train, tf_dic, feat_lab)
    
    d = train_prepd[tf_str]
        
    ###SingTreeRegressor###
    start = time.time()
    model = DecisionTreeRegressor()
    model.fit(d[feat_lab], d[tf_str])
    y_pred = model.predict(test[feat_lab])
    edge_trades_sing = hp.trading_strat2({tf_str : y_pred}, test['Gmt time'], {tf_str : last_lim})
    print(f"Time: {time.time() - start:.2f}s")
    ##############################################################################################
        
    
    #edge_trades = edge_trades_xgb + edge_trades_lgb + edge_trades_cat
    return edge_trades_sing



def xgb_pipeline(data, t_sz, pair, rols, tf_str, tf_int, threshold):
    
    curr_dir = os.getcwd() +f"/storage/Pipelines/fin_xgb/unfiltered/{pair}/train_size={t_sz}/rols={rols}/{tf_str}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)

    tf_dic = { tf_str : tf_int }

    data, feat_lab = hp.feature_calculation(data, final=True)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    #high importance in last_lim

    last_lim = [-threshold, threshold]
    
    for i, rol in enumerate(rolled):
        #print("------------")
        print(f"ROL {i + 1} / {len(rolled)}", end = "\r")
        train = rol[0]
        test = rol[1]
        train_prepd = hp.targets(train, tf_dic, feat_lab)
        predicts = {}
        
        d = train_prepd[tf_str]

        ############ TESTING SECTION ############
        #_train = d[:int(len(train) * 0.9)]
        #_val = d[int(len(train) * 0.9):]
            
        ###XGBOOST###
        start = time.time()
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
        print(f"Time: {time.time() - start:.2f}s")
        #print("Trade Percent: ", (edge_trades_xgb[tf].astype(bool).sum() / len(edge_trades_xgb)) * 100)
        ##############
            
        
        edge_trades = edge_trades_xgb
        edge_trades_total.append(edge_trades)
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)
    
    return

def lgbm_pipeline(data, t_sz, pair, rols, tf_str, tf_int, threshold):
    
    curr_dir = os.getcwd() +f"/storage/Pipelines/fin_lgb/unfiltered/{pair}/train_size={t_sz}/rols={rols}/{tf_str}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)

    tf_dic = { tf_str : tf_int }

    data, feat_lab = hp.feature_calculation(data, final=True)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    #high importance in last_lim

    last_lim = [-threshold, threshold]
    
    for i, rol in enumerate(rolled):
        #print("------------")
        #print(f"ROL {i + 1} / {len(rolled)}")
        train = rol[0]
        test = rol[1]
        train_prepd = hp.targets(train, tf_dic, feat_lab)
        predicts = {}
        
        d = train_prepd[tf_str]
            
        ###LGBOOST###
        params = {
            'device': 'gpu',
            'verbose': -1
        }
        start = time.time()
        #print("LigthGBM")
        train_data = lgb.Dataset(d[feat_lab], label=d[tf_str], params=params)
        bst = lgb.train(params, train_data, 100)
        y_pred = bst.predict(test[feat_lab], num_iteration=bst.best_iteration)
        predicts[tf_str] = y_pred
        edge_trades_lgb = hp.trading_strat2({"week" : y_pred}, test['Gmt time'], {"week" : last_lim})
        print(f"Time: {time.time() - start:.2f}s")
        #print("Trade Percent: ", (edge_trades_lgb['week'].astype(bool).sum() / len(edge_trades_lgb)) * 100)
        #score_lgb = hp.trade_score2([edge_trades_lgb], data)
        #print(f"Score: {score_rfg:.2f}")
            
        
        edge_trades = edge_trades_lgb
        edge_trades_total.append(edge_trades)
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)
    
    return

def cat_pipeline(data, t_sz, pair, rols, tf_str, tf_int, threshold):
    
    curr_dir = os.getcwd() +f"/storage/Pipelines/fin_cat/unfiltered/{pair}/train_size={t_sz}/rols={rols}/{tf_str}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)

    tf_dic = { tf_str : tf_int }

    data, feat_lab = hp.feature_calculation(data, final=True)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    #high importance in last_lim

    last_lim = [-threshold, threshold]
    
    for i, rol in enumerate(rolled):
        #print("------------")
        print(f"ROL {i + 1} / {len(rolled)}", end = "\r")
        train = rol[0]
        test = rol[1]
        train_prepd = hp.targets(train, tf_dic, feat_lab)
        predicts = {}
        
        d = train_prepd[tf_str]

        ###CATBOOST###
        start = time.time()
        #print("catboost")
        catboost_model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=False)
        catboost_model.fit(d[feat_lab], d[tf_str])
        y_pred = catboost_model.predict(test[feat_lab])
        predicts[tf_str] = y_pred
        edge_trades_lgb = hp.trading_strat2({"week" : y_pred}, test['Gmt time'], {"week" : last_lim})
        print(f"Time: {time.time() - start:.2f}s")
        
        edge_trades = edge_trades_lgb
        edge_trades_total.append(edge_trades)
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)
    
    return

def singTree_pipeline(data, t_sz, pair, rols, tf_str, tf_int, threshold):
    
    curr_dir = os.getcwd() +f"/storage/Pipelines/fin_sing/unfiltered/{pair}/train_size={t_sz}/rols={rols}/{tf_str}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)

    tf_dic = { tf_str : tf_int }

    data, feat_lab = hp.feature_calculation(data, final=True)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    #high importance in last_lim

    last_lim = [-threshold, threshold]
    
    for i, rol in enumerate(rolled):
        #print("------------")
        #print(f"ROL {i + 1} / {len(rolled)}", end = "\r")
        train = rol[0]
        test = rol[1]
        train_prepd = hp.targets(train, tf_dic, feat_lab)
        
        d = train_prepd[tf_str]
            
        ### Sing Tree ###
        start = time.time()
        model = DecisionTreeRegressor()
        model.fit(d[feat_lab], d[tf_str])
        y_pred = model.predict(test[feat_lab])
        edge_trades_sing = hp.trading_strat2({"week" : y_pred}, test['Gmt time'], {"week" : last_lim})
        print(f"Time: {time.time() - start:.2f}s")
        ###################
        
        edge_trades = edge_trades_sing
        edge_trades_total.append(edge_trades)
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)
    
    return

def rbfSVM_pipeline(data, t_sz, pair, rols, tf_str, tf_int, threshold):
    
    curr_dir = os.getcwd() +f"/storage/Pipelines/fin_rbfSVM2/unfiltered/{pair}/train_size={t_sz}/rols={rols}/{tf_str}"
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)

    tf_dic = { tf_str : tf_int }

    data, feat_lab = hp.feature_calculation(data, final=True)
    rolled = hp.rolling_window(data, t_sz, testing_size=1, rols = rols)

    #model outputs
    edge_trades_total = []

    #still some issues with the last lim
    #high importance in last_lim

    last_lim = [-threshold, threshold]
    
    for i, rol in enumerate(rolled):
        #print("------------")
        #print(f"ROL {i + 1} / {len(rolled)}", end = "\r")
        train = rol[0]
        test = rol[1]
        train_prepd = hp.targets(train, tf_dic, feat_lab)
        predicts = {}
        
        d = train_prepd[tf_str]

        ###SVMBoost###
        start = time.time()
        #print("rbfSVM")

        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1, max_iter=250)
        svr_rbf.fit(d[feat_lab], d[tf_str])
        y_pred = svr_rbf.predict(test[feat_lab])
        predicts[tf_str] = y_pred
        edge_trades_svm = hp.trading_strat2({"week" : y_pred}, test['Gmt time'], {"week" : last_lim})
        print(f"Time: {time.time() - start:.2f}s")
        #print("Trade Percent: ", (edge_trades_lgb['week'].astype(bool).sum() / len(edge_trades_lgb)) * 100)
        #score_lgb = hp.trade_score2([edge_trades_lgb], data)
        #print(f"Score: {score_rfg:.2f}")
            
        edge_trades = edge_trades_svm
        edge_trades_total.append(edge_trades)
    
    with open(curr_dir + "/trades.pkl", 'wb') as f:
        pickle.dump(edge_trades_total, f)
    
    return


if __name__ == "__main__":
    #'EURUSD',
    pairs = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']
    print("Runnig pipeline.py")
    
    sizes = [16]
    tf_dic = {'week': 60*24*7 }

    
    for p in tqdm(pairs):
        minute = hp.minute_data_loading(p, filtered=False)
        for tf_str, tf_int in tf_dic.items():
            for s in sizes:
                print("xgb")
                xgb_pipeline(minute, s, p, -1, tf_str, tf_int, 2.5)
                """
                print("lgbm")
                lgbm_pipeline(minute, s, p, -1, tf_str, tf_int, 2.5)
                print("catBoost")
                cat_pipeline(minute, s, p, -1, tf_str, tf_int, 2.5)
                print("singTree")
                singTree_pipeline(minute, s, p, -1, tf_str, tf_int, 2.5)
                print("rbfSVM")
                rbfSVM_pipeline(minute, s, p, -1, tf_str, tf_int, 2.5)
                """