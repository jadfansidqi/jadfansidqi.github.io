import numpy as np, sim
def kirpich(df):
    L = df.L_r + df.L_g + df.H_b + df.L_p
    drop = df.L_r*df.S_roof + df.L_g*df.S_g + df.H_b + df.L_p*df.S_p
    return 0.0195*L**0.77*(drop/L)**-0.385*60            # s
def _vfull_rect(df):
    A = df.b_g*df.d_g; R = A/(df.b_g+2*df.d_g)
    return R**(2/3)*np.sqrt(df.S_g)/df.n_g
def _vfull_pipe(df):
    return (df.D_p/4)**(2/3)*np.sqrt(df.S_p)/df.n_p
def kerby_manning(df):
    nd = df.roof_type.map({k: v[3] for k, v in sim.ROOF.items()})
    t0 = 1.44*(df.L_r*nd/np.sqrt(df.S_roof))**0.467*60
    return t0 + df.L_g/_vfull_rect(df) + df.H_b/3.0 + df.L_p/_vfull_pipe(df)
def kinematic_sum(df, n_iter=40):
    """Ragan-Duru kinematic-wave roof time + equilibrium Manning travel times,
    iterated with the Mononobe IDF (no lateral-inflow dynamics, no outlet
    storage, no substrate detention)."""
    tc = np.full(len(df), 300.0)
    a = np.sqrt(df.S_roof.values)/df.n_roof.values
    for _ in range(n_iter):
        i = sim.mononobe(df.R24.values, tc)/1000/3600
        te = (df.L_r.values/(a*i**(2/3)))**0.6
        Qg = i*df.L_r.values*df.L_g.values
        Ag = sim.rect_A_of_Q(Qg, df.b_g.values, df.n_g.values, df.S_g.values)
        tg = df.L_g.values/(Qg/Ag)
        Qd = 2*Qg
        tdp, _ = sim.downpipe_lag(Qd, df.D_dp.values, df.H_b.values)
        Q = df.N_dp.values*Qd
        _, _, Vp = sim.pipe_lag(Q, df.D_p.values, df.n_p.values, df.S_p.values, df.L_p.values)
        tnew = te + tg + tdp + df.L_p.values/Vp
        if np.max(np.abs(tnew-tc)/tc) < 1e-6: tc = tnew; break
        tc = np.sqrt(tc*tnew)
    return tc
