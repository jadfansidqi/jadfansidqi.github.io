import numpy as np, pandas as pd, json, time, warnings, pickle
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import loguniform, uniform, randint, wilcoxon
import xgboost as xgb, lightgbm as lgb
import empirical as E

df = pd.read_csv('dataset_all.csv')
nall = len(df)
df = df[~df.overflow].reset_index(drop=True)
print('valid', len(df), 'of', nall)
df['tc_kirpich'] = E.kirpich(df); df['tc_kerby'] = E.kerby_manning(df); df['tc_kw'] = E.kinematic_sum(df)

NUM = ['n_floor','h_floor','A_roof','aspect','S_roof','n_roof','K_green','a_dp','D_dp',
       'b_g','d_g','S_g','L_p','S_p','D_p','R24']
def feats(d, hybrid=False):
    X = d[NUM].copy()
    for k in range(4): X[f'roof_{k}'] = (d.roof_type == k).astype(float)
    if hybrid: X['log_tc_kw'] = np.log(d.tc_kw)
    return X
LOGF = ['A_roof','S_roof','n_roof','a_dp','S_g','L_p','S_p','R24','h_floor','n_floor','aspect','b_g','d_g','D_dp','D_p']
def mlr_X(X):
    Z = X.copy()
    for c in LOGF: Z[c] = np.log(Z[c])
    Z['K_green'] = np.log1p(Z['K_green'])
    return Z.drop(columns=['roof_0'])

def metrics(o, p):
    o, p = np.asarray(o), np.asarray(p)
    nse = 1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
    r = np.corrcoef(o, p)[0, 1]
    kge = 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)
    return dict(R2=r**2, NSE=nse, KGE=kge, RMSE=np.sqrt(np.mean((o-p)**2)), MAE=np.mean(np.abs(o-p)),
                MAPE=np.mean(np.abs(o-p)/o)*100, PBIAS=100*np.sum(p-o)/np.sum(o),
                W10=np.mean(np.abs(p-o)/o <= 0.10)*100)

def models(seed=0):
    return {
     'MLR': (LinearRegression(), None),
     'SVR': (make_pipeline(StandardScaler(), SVR()), {'svr__C': loguniform(1e-1, 1e3), 'svr__gamma': loguniform(1e-3, 1), 'svr__epsilon': loguniform(1e-3, 1e-1)}),
     'ANN': (make_pipeline(StandardScaler(), MLPRegressor(max_iter=3000, early_stopping=True, random_state=seed)),
             {'mlpregressor__hidden_layer_sizes': [(64,64), (128,64), (128,128,64), (256,128,64)],
              'mlpregressor__alpha': loguniform(1e-6, 1e-2), 'mlpregressor__learning_rate_init': loguniform(3e-4, 3e-3)}),
     'RF': (RandomForestRegressor(n_jobs=4, random_state=seed),
            {'n_estimators': randint(200, 600), 'max_features': uniform(0.3, 0.7), 'min_samples_leaf': randint(1, 5)}),
     'XGBoost': (xgb.XGBRegressor(n_jobs=4, random_state=seed, tree_method='hist'),
                 {'n_estimators': randint(300, 1500), 'learning_rate': loguniform(0.01, 0.2), 'max_depth': randint(3, 9),
                  'subsample': uniform(0.6, 0.4), 'colsample_bytree': uniform(0.5, 0.5), 'min_child_weight': randint(1, 10),
                  'reg_lambda': loguniform(1e-3, 10)}),
     'LightGBM': (lgb.LGBMRegressor(n_jobs=4, random_state=seed, verbose=-1),
                  {'n_estimators': randint(300, 1500), 'learning_rate': loguniform(0.01, 0.2), 'num_leaves': randint(15, 127),
                   'subsample': uniform(0.6, 0.4), 'subsample_freq': [1], 'colsample_bytree': uniform(0.5, 0.5),
                   'min_child_samples': randint(5, 40), 'reg_lambda': loguniform(1e-3, 10)}),
    }

def fit(name, est, grid, X, y, n_iter=25, seed=0):
    if name == 'MLR':
        est.fit(mlr_X(X), y); return est, {}
    rs = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=KFold(5, shuffle=True, random_state=seed),
                            scoring='neg_root_mean_squared_error', n_jobs=1 if name in ('RF','XGBoost','LightGBM') else 4,
                            random_state=seed)
    rs.fit(X, y); return rs.best_estimator_, rs.best_params_

def pred(name, m, X):
    return np.exp(m.predict(mlr_X(X) if name == 'MLR' else X))

if __name__ == '__main__':
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df.roof_type)
    out = {'n_valid': len(df), 'n_all': nall, 'n_train': len(tr), 'n_test': len(te)}
    res = {}; preds = {'tc': te.tc.values / 60}; best = {}; cvres = {}; fitted = {}
    for k, col in [('Kirpich', 'tc_kirpich'), ('Kerby–Manning', 'tc_kerby'), ('KW component sum', 'tc_kw')]:
        res[k] = metrics(te.tc/60, te[col]/60); preds[k] = te[col].values/60
    for hybrid in [False, True]:
        Xtr, Xte = feats(tr, hybrid), feats(te, hybrid); ytr = np.log(tr.tc)
        for name, (est, grid) in models().items():
            t0 = time.time()
            m, bp = fit(name, est, grid, Xtr, ytr)
            key = name + (' (hybrid)' if hybrid else '')
            p = pred(name, m, Xte)/60
            res[key] = metrics(te.tc/60, p); res[key]['fit_s'] = time.time()-t0
            preds[key] = p; best[key] = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in bp.items()}
            fitted[key] = m
            print(key, {k: round(v, 4) for k, v in res[key].items()}, flush=True)
    # per roof type for best models
    rt = {}
    for key in ['KW component sum', 'XGBoost', 'LightGBM', 'XGBoost (hybrid)', 'LightGBM (hybrid)', 'MLR']:
        rt[key] = {int(k): metrics(te.tc.values[te.roof_type.values == k]/60, preds[key][te.roof_type.values == k]) for k in range(4)}
    out.update(test=res, by_roof=rt, best_params=best)
    pd.DataFrame(preds).assign(roof_type=te.roof_type.values, n_floor=te.n_floor.values, R24=te.R24.values,
        tc_node=te.t_node.values/60).to_csv('test_preds.csv', index=False)
    # Wilcoxon on absolute errors: best ML vs KW
    e1 = np.abs(preds['KW component sum']-preds['tc']); 
    for key in ['XGBoost', 'LightGBM', 'XGBoost (hybrid)', 'LightGBM (hybrid)']:
        e2 = np.abs(preds[key]-preds['tc']); w = wilcoxon(e1, e2)
        out.setdefault('wilcoxon_vs_KW', {})[key] = dict(stat=float(w.statistic), p=float(w.pvalue))
    # MLR coefficients (explicit power law)
    m = fitted['MLR']; cols = mlr_X(feats(tr)).columns
    out['mlr_coef'] = dict(intercept=float(m.intercept_), **{c: float(v) for c, v in zip(cols, m.coef_)})
    json.dump(out, open('ml_results.json', 'w'), indent=1, default=float)
    pickle.dump({'fitted': fitted, 'tr_idx': tr.index.values, 'te_idx': te.index.values}, open('models.pkl', 'wb'))
    df.to_csv('dataset_valid.csv', index=False)
