"""Check fixed-point residual per sample and repair non-converged samples by bisection."""
import numpy as np, pandas as pd, sim, json
df = pd.read_csv('dataset_all.csv')
d = {c: df[c].values for c in df.columns}
def G(dsub, tc):
    i = sim.mononobe(dsub['R24'], tc)/1000/3600
    t95, Hmax, y_eq, te, Qin = sim.node_t95(dsub, i)
    Qd = 0.95*Qin
    tdp, Vt = sim.downpipe_lag(Qd, dsub['D_dp'], dsub['H_b'])
    tcp, sur, Vp = sim.pipe_lag(dsub['N_dp']*Qd, dsub['D_p'], dsub['n_p'], dsub['S_p'], dsub['L_p'])
    return t95+tdp+tcp, dict(t_node=t95, t_roof=te, t_dp=tdp, t_cp=tcp, Hmax=Hmax, y_eq=y_eq, Q_dp=Qd, surcharge=sur, Vt=Vt, i_mmh=i*3.6e6)
tcn, aux = G(d, df.tc.values)
res = np.abs(tcn-df.tc.values)/df.tc.values
bad = np.nonzero(~(res < 2e-3))[0]
print('residual median', np.nanmedian(res), 'n bad', len(bad))
if len(bad):
    sub = {k: v[bad] for k, v in d.items()}
    lo = np.full(len(bad), np.log(10.0)); hi = np.full(len(bad), np.log(7200.0))
    for it in range(18):
        mid = 0.5*(lo+hi); t, _ = G(sub, np.exp(mid))
        f = np.log(t) - mid           # decreasing in mid
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    tcb = np.exp(0.5*(lo+hi)); t, a2 = G(sub, tcb)
    df.loc[bad, 'tc'] = t
    for k, v in a2.items(): df.loc[bad, k] = v
    tcn[bad] = t; 
    for k in aux: aux[k][bad] = a2[k]
# final consistency of auxiliary outputs
for k, v in aux.items(): df[k] = v
df['tc'] = tcn
df['overflow'] = (df.Hmax > df.d_g) | (df.y_eq > df.d_g) | df.tc.isna()
tc2, _ = G(d | {}, df.tc.values) if False else (None, None)
df.to_csv('dataset_all.csv', index=False)
json.dump({'n_bisected': int(len(bad)), 'median_residual_before': float(np.nanmedian(res)), 'overflow_frac': float(df.overflow.mean())}, open('polish_meta.json', 'w'))
print('ok', df.overflow.mean())
