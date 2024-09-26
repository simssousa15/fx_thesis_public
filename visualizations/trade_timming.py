import os
import pickle
import matplotlib.pyplot as plt

tickers = ['EURUSD', 'AUDUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD']


nothing = 0
    
xgb = 0
lgbm = 0
cat = 0
sing = 0

xgb_sing = 0
lgbm_sing = 0
cat_sing = 0
xgb_lgbm = 0
xgb_cat = 0
lgbm_cat = 0

xgb_lgbm_cat = 0
xgb_lgbm_sing = 0
xgb_cat_sing = 0
lgbm_cat_sing = 0

xgb_lgbm_cat_sing = 0


for p in tickers:
    xgb_dir = os.getcwd() + f"/storage/Pipelines/fin_xgb/unfiltered/{p}/train_size=16/rols=-1/week/"
    lgbm_dir = os.getcwd() + f"/storage/Pipelines/fin_lgbm/unfiltered/{p}/train_size=16/rols=-1/week/"
    cat_dir = os.getcwd() + f"/storage/Pipelines/fin_cat/unfiltered/{p}/train_size=16/rols=-1/week/"
    sing_dir = os.getcwd() + f"/storage/Pipelines/fin_sing/unfiltered/{p}/train_size=16/rols=-1/week/"

    models = {  "XGBoost": xgb_dir, 
                "LightGBM": lgbm_dir,
                "CatBoost": cat_dir, 
                "Decision Tree": sing_dir    
            }

    
    print(p)
    t = {}
    for n, dir in models.items():
        with open(os.path.join(dir, "trades.pkl"), 'rb') as f:
            aux =  pickle.load(f)
            t[n] = []
            for rol in aux:
                t[n].extend(list(rol['week'].values))
    
    #JI CODE
    #check pecentage of equal values between 2 models
    """ for n1, t1 in t.items():
        for n2, t2 in t.items():
            if n1 != n2:
                a = 0
                b = 0
                equal = 0
                for i in range(len(t1)):
                    if t1[i] != 0:
                        a += 1
                        if(t1[i] == t2[i]):
                            equal += 1
                    if t2[i] != 0:
                        b += 1

                #calculate Jaccard index
                jac = equal/(a+b-equal)
                print(f"{n1} & {n2}: {jac}") """

    for i in range(len(t['XGBoost'])):
        if t['XGBoost'][i] == 0 and t['LightGBM'][i] == 0 and t['CatBoost'][i] == 0 and t['Decision Tree'][i] == 0:
            nothing += 1

        elif t['XGBoost'][i] != 1 and t['LightGBM'][i] == 0 and t['CatBoost'][i] == 0 and t['Decision Tree'][i] == 0:
            xgb += 1
        elif t['XGBoost'][i] == 0 and t['LightGBM'][i] != 1 and t['CatBoost'][i] == 0 and t['Decision Tree'][i] == 0:
            lgbm += 1
        elif t['XGBoost'][i] == 0 and t['LightGBM'][i] == 0 and t['CatBoost'][i] != 1 and t['Decision Tree'][i] == 0:
            cat += 1
        elif t['XGBoost'][i] == 0 and t['LightGBM'][i] == 0 and t['CatBoost'][i] == 0 and t['Decision Tree'][i] != 1:
            sing += 1

        elif t['XGBoost'][i] != 1 and t['LightGBM'][i] != 0 and t['CatBoost'][i] == 0 and t['Decision Tree'][i] == 1:
            xgb_sing += 1
        elif t['XGBoost'][i] != 0 and t['LightGBM'][i] == 1 and t['CatBoost'][i] != 0 and t['Decision Tree'][i] == 1:
            lgbm_sing += 1
        elif t['XGBoost'][i] == 0 and t['LightGBM'][i] != 0 and t['CatBoost'][i] != 1 and t['Decision Tree'][i] == 1:
            cat_sing += 1
        elif t['XGBoost'][i] != 1 and t['LightGBM'][i] != 1 and t['CatBoost'][i] != 0 and t['Decision Tree'][i] == 0:
            xgb_lgbm += 1
        elif t['XGBoost'][i] != 1 and t['LightGBM'][i] == 0 and t['CatBoost'][i] == 1 and t['Decision Tree'][i] != 0:
            xgb_cat += 1
        elif t['XGBoost'][i] == 0 and t['LightGBM'][i] != 1 and t['CatBoost'][i] == 1 and t['Decision Tree'][i] != 0:
            lgbm_cat += 1

        
        elif t['XGBoost'][i] != 1 and t['LightGBM'][i] != 1 and t['CatBoost'][i] != 1 and t['Decision Tree'][i] == 0:
            xgb_lgbm_cat += 1
        elif t['XGBoost'][i] != 1 and t['LightGBM'][i] != 1 and t['CatBoost'][i] == 0 and t['Decision Tree'][i] != 1:
            xgb_lgbm_sing += 1
        elif t['XGBoost'][i] != 1 and t['LightGBM'][i] == 0 and t['CatBoost'][i] != 1 and t['Decision Tree'][i] != 1:
            xgb_cat_sing += 1
        elif t['XGBoost'][i] == 0 and t['LightGBM'][i] != 1 and t['CatBoost'][i] != 1 and t['Decision Tree'][i] != 1:
            lgbm_cat_sing += 1
        
        elif t['XGBoost'][i] != 1 and t['LightGBM'][i] != 1 and t['CatBoost'][i] != 1 and t['Decision Tree'][i] != 1:
            xgb_lgbm_cat_sing += 1
    

print(f"Nothing: {nothing}")

print(f"XGBoost: {xgb}")
print(f"LightGBM: {lgbm}")
print(f"CatBoost: {cat}")
print(f"Decision Tree: {sing}")

print(f"XGBoost & LightGBM: {xgb_lgbm}")
print(f"XGBoost & CatBoost: {xgb_cat}")
print(f"LightGBM & CatBoost: {lgbm_cat}")
print(f"XGBoost & Decision Tree: {xgb_sing}")
print(f"LightGBM & Decision Tree: {lgbm_sing}")
print(f"CatBoost & Decision Tree: {cat_sing}")

print(f"XGBoost & LightGBM & CatBoost: {xgb_lgbm_cat}")
print(f"XGBoost & LightGBM & Decision Tree: {xgb_lgbm_sing}")
print(f"XGBoost & CatBoost & Decision Tree: {xgb_cat_sing}")
print(f"LightGBM & CatBoost & Decision Tree: {lgbm_cat_sing}")

print(f"XGBoost & LightGBM & CatBoost & Decision Tree: {xgb_lgbm_cat_sing}")

            

    