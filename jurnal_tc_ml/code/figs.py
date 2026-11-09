import numpy as np, pandas as pd, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
import sim
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.linewidth': 0.8,
                     'savefig.dpi': 600, 'figure.dpi': 150})
df = pd.read_csv('dataset_valid.csv'); pr = pd.read_csv('test_preds.csv')
R = json.load(open('ml_results.json')); X = json.load(open('extra_results.json'))
RT = {k: v[0] for k, v in sim.ROOF.items()}
COL = ['#3B6EA8', '#D08C2E', '#8C4A8C', '#4C9A5B']
MK = ['o', 's', '^', 'D']

# Fig 1: schematic of the drainage cascade
fig, ax = plt.subplots(figsize=(6.8, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
ax.add_patch(Rectangle((1.2, 0.6), 2.4, 4.2, fc='#EEEEEE', ec='k', lw=0.8))
for yy in np.arange(1.2, 4.8, 0.6): ax.plot([1.2, 3.6], [yy, yy], color='#BBBBBB', lw=0.5)
ax.add_patch(Polygon([[1.0, 4.8], [2.4, 5.5], [3.8, 4.8]], closed=True, fc='#CFE0F1', ec='k', lw=0.8))
for xx in np.linspace(1.3, 3.5, 7): ax.annotate('', (xx-0.15, 5.6), (xx, 5.95), arrowprops=dict(arrowstyle='->', color='#3B6EA8', lw=0.7))
ax.text(2.4, 5.98, 'i(t$_c$) = R$_{24}$/24 · (24/t$_c$)$^{2/3}$', ha='center', fontsize=8)
ax.add_patch(Rectangle((3.6, 4.6), 0.35, 0.22, fc='#999999', ec='k', lw=0.6))
ax.plot([3.9, 3.9], [4.6, 0.6], color='#3B6EA8', lw=2.2)
ax.plot([3.9, 8.6], [0.45, 0.25], color='#3B6EA8', lw=3)
ax.add_patch(Rectangle((8.6, 0.0), 0.5, 0.6, fc='#777777', ec='k', lw=0.6))
ax.text(9.15, 0.75, 'Campus\ndrain', fontsize=7, ha='center')
lab = [('(1) Roof plane: kinematic wave\n    q = (√S/n) h$^{5/3}$, length L$_r$', 5.2, 5.2),
       ('(2) Green-roof substrate (if any):\n    linear reservoir, K', 5.2, 4.35),
       ('(3) Gutter: 1-D kinematic wave with\n    lateral inflow q$_L$(t), length L$_g$', 5.2, 3.5),
       ('(4) Outlet node: level pool,\n    weir/orifice control (D$_{dp}$)', 5.2, 2.65),
       ('(5) Downpipe: H$_b$ = N$_f$·h$_f$ + 1,\n    terminal velocity V$_t$', 5.2, 1.8),
       ('(6) Collector pipe: kinematic\n    celerity, L$_p$, S$_p$, D$_p$', 5.2, 0.95)]
for t, x, y in lab: ax.text(x, y, t, fontsize=7.5, va='center')
for t, x, y in [('(1)', 1.75, 5.0), ('(3)', 3.75, 5.0), ('(4)', 4.15, 4.45), ('(5)', 4.15, 2.6), ('(6)', 6.2, 0.55)]:
    ax.text(x, y, t, fontsize=8, color='#B03A2E', fontweight='bold')
ax.text(2.4, 2.6, 'N$_f$ floors\n× h$_f$', ha='center', fontsize=8)
fig.tight_layout(); fig.savefig('Fig1_schematic.png'); plt.close()

# Fig 2: dataset characteristics
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
bins = np.logspace(np.log10(df.tc.min()/60), np.log10(df.tc.max()/60), 40)
for k in range(4):
    axs[0].hist(df.tc[df.roof_type == k]/60, bins=bins, histtype='step', lw=1.2, color=COL[k], label=RT[k])
axs[0].set_xscale('log'); axs[0].set_xlabel('Reference t$_c$ (min)'); axs[0].set_ylabel('Count'); axs[0].legend(fontsize=6.5, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.62))
axs[0].axvline(5, color='k', ls=':', lw=0.8); axs[0].text(5.3, axs[0].get_ylim()[1]*0.93, '5 min', fontsize=7)
hard = df[df.roof_type < 3]
fb = pd.cut(hard.n_floor, [1, 5, 10, 15, 20, 25, 30])
sh = hard.groupby(fb, observed=True).apply(lambda g: pd.Series({
    'Roof+gutter+outlet': (g.t_node/g.tc).mean(), 'Downpipe': (g.t_dp/g.tc).mean(), 'Collector': (g.t_cp/g.tc).mean()}))
bot = np.zeros(len(sh)); xs = np.arange(len(sh))
for c, cc in zip(sh.columns, ['#9DB9D8', '#3B6EA8', '#1F3B5C']):
    axs[1].bar(xs, sh[c]*100, bottom=bot, color=cc, label=c, width=0.75); bot += sh[c].values*100
axs[1].set_xticks(xs); axs[1].set_xticklabels(['2–5', '6–10', '11–15', '16–20', '21–25', '26–30'], rotation=45, fontsize=7)
axs[1].set_xlabel('Number of floors'); axs[1].set_ylabel('Share of t$_c$ (%)'); axs[1].legend(fontsize=6, frameon=False, loc='lower left')
axs[1].set_title('Impervious roofs', fontsize=8)
for k in range(4):
    g = df[df.roof_type == k].groupby('n_floor').tc.median()/60
    axs[2].plot(g.index, g.values, marker=MK[k], ms=2.5, lw=0.9, color=COL[k], label=RT[k])
axs[2].set_yscale('log'); axs[2].set_yticks([1, 2, 5, 10, 20, 40]); axs[2].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); axs[2].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); axs[2].set_xlabel('Number of floors'); axs[2].set_ylabel('Median t$_c$ (min)')
for a, l in zip(axs, 'abc'): a.text(-0.2, 1.04, f'({l})', transform=a.transAxes, fontweight='bold')
fig.tight_layout(); fig.savefig('Fig2_dataset.png'); plt.close()

# Fig 3: predicted vs reference
keys = ['Kirpich', 'Kerby–Manning', 'KW component sum', 'MLR', 'LightGBM', 'LightGBM (hybrid)']
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
lo, hi = 0.3, 100
for a, k in zip(axs.flat, keys):
    for r in range(4):
        m = pr.roof_type == r
        a.scatter(pr.tc[m], pr[k][m], s=3, alpha=0.5, color=COL[r], marker=MK[r], lw=0, label=RT[r])
    a.plot([lo, hi], [lo, hi], 'k-', lw=0.7); a.plot([lo, hi], [lo*1.1, hi*1.1], 'k:', lw=0.5); a.plot([lo, hi], [lo/1.1, hi/1.1], 'k:', lw=0.5)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(lo*0.3 if k == 'Kirpich' else lo, hi)
    s = R['test'][k]; a.set_title(k, fontsize=8.5)
    a.text(0.04, 0.96, f"NSE = {s['NSE']:.3f}\nMAPE = {s['MAPE']:.1f}%", transform=a.transAxes, va='top', fontsize=7)
    a.set_xlabel('Reference t$_c$ (min)', fontsize=8); a.set_ylabel('Estimated t$_c$ (min)', fontsize=8)
axs[0, 0].legend(fontsize=6, frameon=False, loc='lower right', markerscale=2)
fig.tight_layout(); fig.savefig('Fig3_scatter.png'); plt.close()

# Fig 4: SHAP
import shap
sv = np.load('shap_values.npy'); Xs = pd.read_csv('shap_X.csv')
NAMES = {'n_floor': 'N$_f$ (floors)', 'h_floor': 'h$_f$', 'A_roof': 'A$_{roof}$', 'aspect': 'Aspect ratio', 'S_roof': 'S$_{roof}$',
         'n_roof': 'n$_{roof}$', 'K_green': 'K (green)', 'a_dp': 'a$_{dp}$', 'D_dp': 'D$_{dp}$', 'b_g': 'b$_g$', 'd_g': 'd$_g$', 'S_g': 'S$_g$',
         'L_p': 'L$_p$', 'S_p': 'S$_p$', 'D_p': 'D$_p$', 'R24': 'R$_{24}$', 'roof_0': 'Roof: flat concrete', 'roof_1': 'Roof: metal',
         'roof_2': 'Roof: clay tile', 'roof_3': 'Roof: green'}
plt.figure(figsize=(6.8, 4.2))
shap.summary_plot(sv, Xs.rename(columns=NAMES), max_display=14, show=False, plot_size=None, cmap='viridis')
ax = plt.gca(); ax.set_xlabel('SHAP value (impact on ln t$_c$)', fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig('Fig4_shap.png'); plt.close()

# Fig 5: SHAP dependence n_floor & learning curve
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
j = list(Xs.columns).index('n_floor'); hardm = Xs.roof_3 < 0.5
axs[0].scatter(Xs.n_floor[hardm], sv[hardm, j], s=3, c='#3B6EA8', alpha=0.5, lw=0, label='Impervious')
axs[0].scatter(Xs.n_floor[~hardm], sv[~hardm, j], s=3, c='#4C9A5B', alpha=0.5, lw=0, label='Green')
axs[0].set_xlabel('Number of floors N$_f$'); axs[0].set_ylabel('SHAP of N$_f$'); axs[0].legend(fontsize=6.5, frameon=False, markerscale=3)
jj = list(Xs.columns).index('a_dp')
sc = axs[1].scatter(Xs.a_dp, sv[:, jj], s=3, c=Xs.D_dp*1000, cmap='viridis', lw=0)
cb = plt.colorbar(sc, ax=axs[1]); cb.set_label('D$_{dp}$ (mm)', fontsize=7); cb.ax.tick_params(labelsize=6)
axs[1].set_xlabel('Roof area per downpipe a$_{dp}$ (m²)'); axs[1].set_ylabel('SHAP of a$_{dp}$')
lc = X['learning_curve_MAPE']; ns = sorted(map(int, lc))
for k, c, m in [('MLR', '#999999', 'o'), ('ANN', '#D08C2E', 's'), ('XGBoost', '#8C4A8C', '^'), ('LightGBM', '#3B6EA8', 'D'), ('LightGBM (hybrid)', '#1F3B5C', 'v')]:
    axs[2].plot(ns, [lc[str(n)][k] for n in ns], marker=m, ms=3, lw=0.9, color=c, label=k)
axs[2].set_xscale('log'); axs[2].set_xlabel('Training samples'); axs[2].set_ylabel('Test MAPE (%)'); axs[2].legend(fontsize=6, frameon=False)
for a, l in zip(axs, 'abc'): a.text(-0.22, 1.04, f'({l})', transform=a.transAxes, fontweight='bold')
fig.tight_layout(); fig.savefig('Fig5_dependence_lc.png'); plt.close()
print('figs ok')
