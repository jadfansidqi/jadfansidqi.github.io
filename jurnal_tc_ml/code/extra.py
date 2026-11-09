"""Out-of-distribution tests, learning curve, SHAP, peak-flow propagation."""
import numpy as np, pandas as pd, json, pickle, shap, warnings
warnings.filterwarnings('ignore')
import ml, sim
df = pd.read_csv('dataset_valid.csv')
R = json.load(open('ml_results.json'))
BEST = R['best_params']
def mk(name, hybrid=False):
    est, _ = ml.models()[name]
    est.set_params(**BEST[name + (' (hybrid)' if hybrid else '')])
    return est
out = {}
# OOD extrapolation
splits = {'height (train <=15 floors, test >=16)': (df.n_floor <= 15, df.n_floor >= 16),
          'return period (train T<=10, test T>=20)': (df['T'] <= 10, df['T'] >= 20),
          'roof size (train A<=2500, test A>2500)': (df.A_roof <= 2500, df.A_roof > 2500)}
for s, (a, b) in splits.items():
    tr, te = df[a], df[b]; out[s] = {'n_train': int(a.sum()), 'n_test': int(b.sum())}
    out[s]['KW component sum'] = ml.metrics(te.tc/60, te.tc_kw/60)
    for name in ['MLR', 'XGBoost', 'LightGBM']:
        for hyb in [False, True]:
            if name == 'MLR' and hyb: continue
            est = mk(name, hyb) if name != 'MLR' else ml.models()['MLR'][0]
            X = ml.feats(tr, hyb); 
            est.fit(ml.mlr_X(X) if name == 'MLR' else X, np.log(tr.tc))
            p = ml.pred(name, est, ml.feats(te, hyb))/60
            out[s][name + (' (hybrid)' if hyb else '')] = ml.metrics(te.tc/60, p)
    print(s, {k: round(v['NSE'], 3) for k, v in out[s].items() if isinstance(v, dict)}, flush=True)
# learning curve
P = pickle.load(open('models.pkl', 'rb'))
tr = df.loc[P['tr_idx']]; te = df.loc[P['te_idx']]
lc = {}
rng = np.random.default_rng(0)
for n in [100, 250, 500, 1000, 2000, len(tr)]:
    lc[n] = {}
    for name, hyb in [('LightGBM', False), ('LightGBM', True), ('XGBoost', False), ('MLR', False), ('ANN', False)]:
        vals = []
        for rep in range(3 if n < len(tr) else 1):
            sub = tr.sample(n, random_state=rep) if n < len(tr) else tr
            est = mk(name, hyb) if name != 'MLR' else ml.models()['MLR'][0]
            X = ml.feats(sub, hyb); est.fit(ml.mlr_X(X) if name == 'MLR' else X, np.log(sub.tc))
            p = ml.pred(name, est, ml.feats(te, hyb))/60
            vals.append(ml.metrics(te.tc/60, p)['MAPE'])
        lc[n][name + (' (hybrid)' if hyb else '')] = float(np.mean(vals))
    print('lc', n, lc[n], flush=True)
out['learning_curve_MAPE'] = lc
# SHAP on best pure-ML tree model
key = 'LightGBM' if R['test']['LightGBM']['RMSE'] <= R['test']['XGBoost']['RMSE'] else 'XGBoost'
m = P['fitted'][key]; Xte = ml.feats(te)
sv = shap.TreeExplainer(m).shap_values(Xte)
np.save('shap_values.npy', sv); Xte.to_csv('shap_X.csv', index=False)
imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns).sort_values(ascending=False)
out['shap_model'] = key; out['shap_mean_abs'] = imp.to_dict()
# peak discharge propagation: Qp ∝ i(max(tc,5 min)); also without the 5-min floor
pr = pd.read_csv('test_preds.csv')
qp = {}
for floor in [300.0, 60.0]:
    sim.TD_MIN = floor
    iref = sim.mononobe(pr.R24.values, pr.tc.values*60)
    qp[floor] = {}
    for k in ['Kirpich', 'Kerby–Manning', 'KW component sum', 'MLR', 'XGBoost', 'LightGBM', 'LightGBM (hybrid)', 'XGBoost (hybrid)']:
        e = sim.mononobe(pr.R24.values, pr[k].values*60)/iref-1
        qp[floor][k] = dict(mean_abs_pct=float(np.mean(np.abs(e))*100), p95_abs_pct=float(np.percentile(np.abs(e), 95)*100),
                            bias_pct=float(np.mean(e)*100))
sim.TD_MIN = 300.0
out['Qp_error'] = qp
out['frac_tc_below_5min'] = float(np.mean(df.tc < 300))
out['frac_tc_below_5min_by_roof'] = {int(k): float(np.mean(g.tc < 300)) for k, g in df.groupby('roof_type')}
json.dump(out, open('extra_results.json', 'w'), indent=1, default=float)
print(json.dumps(qp, indent=1))
