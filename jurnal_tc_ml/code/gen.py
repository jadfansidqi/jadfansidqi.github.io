import numpy as np, pandas as pd, time, sim, json
N = 6000
d = sim.sample(N, seed=2026)
t0 = time.time(); o = sim.simulate(d); el = time.time()-t0
df = pd.DataFrame({k: v for k, v in d.items()})
for k in ['tc','i_mmh','t_node','t_roof','t_dp','t_cp','Hmax','y_eq','Q_dp','surcharge','Vt']:
    df[k] = o[k]
df['overflow'] = (df.Hmax > df.d_g) | (df.y_eq > df.d_g) | df.tc.isna()
df.to_csv('dataset_all.csv', index=False)
json.dump({'N': N, 'elapsed_s': el, 'iters': len(o['conv']), 'conv': list(map(float, o['conv'])),
           'overflow_frac': float(df.overflow.mean())}, open('gen_meta.json', 'w'), indent=1)
print('done', el, df.overflow.mean())
# grid-convergence check
sub = {k: v[:300] for k, v in d.items()}
res = {}
for Nx in [10, 20, 30, 60, 120]:
    i = sim.mononobe(sub['R24'], df.tc.values[:300])/1000/3600
    t95 = sim.node_t95(sub, i, Nx=Nx)[0]
    res[Nx] = t95
ref = res[120]
gc = {Nx: float(np.nanmedian(np.abs(res[Nx]-ref)/ref)*100) for Nx in res}
gc95 = {Nx: float(np.nanpercentile(np.abs(res[Nx]-ref)/ref*100, 95)) for Nx in res}
json.dump({'median_abs_pct': gc, 'p95_abs_pct': gc95}, open('grid_conv.json', 'w'), indent=1)
print(gc, gc95)
