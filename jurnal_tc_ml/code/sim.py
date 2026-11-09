"""Physically based reference simulator for time of concentration (tc) of
roof runoff in multi-storey buildings: roof plane (analytic kinematic wave)
-> [green-roof substrate linear reservoir] -> eaves/box gutter (numerical
kinematic wave with lateral inflow) -> roof outlet node (level-pool, weir/
orifice control) -> vertical downpipe (Wyly-Eaton terminal velocity)
-> horizontal collector pipe (kinematic celerity). Rainfall intensity is
coupled to tc through the Mononobe IDF relation (fixed-point iteration)."""
import numpy as np
g = 9.81
M = 5/3
# Log-Pearson III design daily rainfall, Lab. Hidrologi UB station 2005-2024
R24_T = {2: 99.839, 5: 128.565, 10: 150.135, 20: 172.777, 25: 180.381, 50: 205.177}

TD_MIN = 300.0   # minimum design storm duration (5 min) for the IDF relation
def mononobe(R24, t_s):
    th = np.maximum(t_s, TD_MIN) / 3600.0
    return R24/24.0 * (24.0/th)**(2/3)          # mm/h

def lhs(n, d, rng):
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u

ROOF = {  # type: (S range, Manning n range, Kerby nd, green K range [s])
    0: ('Flat concrete', (0.01, 0.03), (0.012, 0.016), 0.02, None),
    1: ('Metal sheet', (0.09, 0.36), (0.010, 0.013), 0.02, None),
    2: ('Clay tile', (0.36, 0.70), (0.016, 0.024), 0.10, None),
    3: ('Extensive green', (0.01, 0.05), (0.15, 0.35), 0.40, (120.0, 600.0)),
}

def sample(n, seed=1):
    rng = np.random.default_rng(seed)
    u = lhs(n, 16, rng)
    d = {}
    d['roof_type'] = rng.choice(4, n, p=[0.40, 0.30, 0.15, 0.15])
    d['n_floor'] = np.floor(2 + u[:, 0]*29).astype(int)          # 2..30
    d['h_floor'] = 3.2 + u[:, 1]*1.3
    d['A_roof'] = np.exp(np.log(300) + u[:, 2]*np.log(6000/300))  # m2
    d['aspect'] = 1 + u[:, 3]*3
    S = np.empty(n); nr = np.empty(n); K = np.zeros(n)
    for k, (_, sr, nn, _, kr) in ROOF.items():
        m = d['roof_type'] == k
        S[m] = sr[0] + u[m, 4]*(sr[1]-sr[0])
        nr[m] = nn[0] + u[m, 5]*(nn[1]-nn[0])
        if kr: K[m] = kr[0] + u[m, 6]*(kr[1]-kr[0])
    d['S_roof'], d['n_roof'], d['K_green'] = S, nr, K
    d['a_dp'] = 30 + u[:, 7]*120                 # roof area served per downpipe, m2
    d['D_dp'] = np.array([0.100, 0.125, 0.150])[np.minimum((u[:, 8]*3).astype(int), 2)]
    d['b_g'] = 0.15 + u[:, 9]*0.25
    d['d_g'] = 0.15 + u[:, 10]*0.15
    d['S_g'] = 0.002 + u[:, 11]*0.008
    d['L_p'] = 5 + u[:, 12]*95
    d['S_p'] = 0.005 + u[:, 13]*0.015
    d['u_Dp'] = 0.7 + u[:, 14]*0.6          # collector sizing factor (0.7-1.3 of design flow)
    Ts = np.array(list(R24_T))
    d['T'] = Ts[np.minimum((u[:, 15]*len(Ts)).astype(int), len(Ts)-1)]
    d['R24'] = np.array([R24_T[t] for t in d['T']])
    d['n_g'] = np.full(n, 0.013); d['n_p'] = np.full(n, 0.012)
    return derive(d)

def derive(d):
    B = np.sqrt(d['A_roof']*d['aspect']); W = np.sqrt(d['A_roof']/d['aspect'])
    ndp = np.maximum(2, 2*np.ceil(d['A_roof']/d['a_dp']/2))
    d['N_dp'] = ndp
    d['L_r'] = W/2*np.sqrt(1+d['S_roof']**2)       # overland flow length on roof plane
    d['L_g'] = B/ndp                                # gutter run to each outlet
    d['H_b'] = d['n_floor']*d['h_floor'] + 1.0      # downpipe drop height
    if 'D_p' not in d:   # collector sized for the 5-min design flow times a sizing factor
        Qdes = d['A_roof']*mononobe(d['R24'], TD_MIN)/1000/3600*d['u_Dp']
        Dl = np.array([0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00])
        Qf = np.array([[circ(2*np.pi, np.array([D]), 0.012, s)[1][0] for D in Dl] for s in d['S_p']])
        ok = Qf >= Qdes[:, None]
        d['D_p'] = np.where(ok.any(1), Dl[np.argmax(ok, 1)], Dl[-1])
    return d

def rect_Q(A, b, n, S):
    y = A/b
    return A*(A/(b+2*y))**(2/3)*np.sqrt(S)/n

def rect_A_of_Q(Q, b, n, S):
    lo = np.zeros_like(Q); hi = np.full_like(Q, 1.0)*b*2.0
    for _ in range(60):
        mid = 0.5*(lo+hi); f = rect_Q(mid, b, n, S) > Q
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    return 0.5*(lo+hi)

def circ(theta, D, n, S):
    A = D**2/8*(theta-np.sin(theta)); P = D*theta/2
    return A, A*(A/P)**(2/3)*np.sqrt(S)/n

def pipe_lag(Q, D, n, S, L):
    Afull = np.pi*D**2/4
    Qfull = circ(np.full_like(D, 2*np.pi), D, n, S)[1]
    lo = np.full_like(Q, 1e-3); hi = np.full_like(Q, 5.28)    # 5.28 rad ~ Qmax
    for _ in range(60):
        mid = 0.5*(lo+hi); f = circ(mid, D, n, S)[1] > Q
        hi = np.where(f, mid, hi); lo = np.where(f, lo, mid)
    th = 0.5*(lo+hi)
    A1, Q1 = circ(th*0.999, D, n, S); A2, Q2 = circ(th*1.001, D, n, S)
    c = (Q2-Q1)/(A2-A1)
    V = Q/circ(th, D, n, S)[0]
    sur = Q > 0.95*Qfull
    lag = np.where(sur, L*Afull/Q, L/c)
    return lag, sur, np.where(sur, Q/Afull, V)

def downpipe_lag(Qd, D, H):
    Vt = 10.07*((Qd*1000)/(D*1000))**0.4
    Lt = 0.1706*Vt**2
    return np.where(H <= Lt, np.sqrt(2*H/g), 2*Lt/Vt + (H-Lt)/Vt), Vt

def outlet_Q(H, D):
    H = np.maximum(H, 0)
    return np.minimum(1.6*np.pi*D*H**1.5, 0.6*np.pi*D**2/4*np.sqrt(2*g*H))

def node_t95(d, i, Nx=30, max_steps=400000):
    """Simulate roof->gutter->outlet node under constant intensity i (m/s).
    Returns time (s) at which outlet discharge reaches 95% of equilibrium,
    max node head, max gutter depth."""
    n = len(i)
    a = np.sqrt(d['S_roof'])/d['n_roof']
    Lr, Lg, b, ng, Sg, D, K = d['L_r'], d['L_g'], d['b_g'], d['n_g'], d['S_g'], d['D_dp'], d['K_green']
    q_eq = i*Lr
    te = (Lr/(a*i**(M-1)))**(1/M)
    Qg_eq = q_eq*Lg
    A_eq = rect_A_of_Q(Qg_eq, b, ng, Sg)
    c_eq = (rect_Q(A_eq*1.01, b, ng, Sg)-rect_Q(A_eq*0.99, b, ng, Sg))/(0.02*A_eq)
    dx = Lg/Nx
    dt = 0.8*dx/c_eq
    As = b*Lg                       # backwater ponding area serving the outlet
    Qin_eq = 2*Qg_eq
    A = np.zeros((n, Nx)); Hs = np.zeros(n); qg = np.zeros(n)
    t = np.zeros(n); t95 = np.full(n, np.nan); Hmax = np.zeros(n)
    Qprev = np.zeros(n); tprev = np.zeros(n)
    green = K > 0
    active = np.ones(n, bool)
    step = 0
    while active.any() and step < max_steps:
        idx = np.nonzero(active)[0]
        tt = t[idx]; dti = dt[idx]
        tm = tt + 0.5*dti
        qr = np.where(tm < te[idx], a[idx]*(i[idx]*tm)**M, q_eq[idx])
        # green-roof substrate detention (linear reservoir, exact update)
        Kg = np.where(green[idx], K[idx], 1.0)
        qgi = np.where(green[idx], qr + (qg[idx]-qr)*np.exp(-dti/Kg), qr)
        qg[idx] = qgi
        Ai = A[idx]
        Q = rect_Q(Ai, b[idx, None], ng[idx, None], Sg[idx, None])
        Qup = np.concatenate([np.zeros((len(idx), 1)), Q[:, :-1]], 1)
        Ai = np.maximum(Ai + dti[:, None]*(qgi[:, None] - (Q-Qup)/dx[idx, None]), 0)
        A[idx] = Ai
        Qin = 2*rect_Q(Ai[:, -1], b[idx], ng[idx], Sg[idx])
        # implicit level-pool update of outlet node
        H = Hs[idx].copy(); Dd = D[idx]; Asi = As[idx]
        for _ in range(4):
            Qo = outlet_Q(H, Dd); dQ = (outlet_Q(H+1e-5, Dd)-Qo)/1e-5
            F = Asi*(H-Hs[idx])/dti - Qin + Qo
            H = np.maximum(H - F/(Asi/dti + dQ), 0)
        Hs[idx] = H
        Qo = outlet_Q(H, Dd)
        Hmax[idx] = np.maximum(Hmax[idx], H)
        t[idx] = tt + dti
        hit = Qo >= 0.95*Qin_eq[idx]
        # linear interpolation of crossing time
        fr = (0.95*Qin_eq[idx]-Qprev[idx])/np.maximum(Qo-Qprev[idx], 1e-15)
        tc_hit = tprev[idx] + np.clip(fr, 0, 1)*(t[idx]-tprev[idx])
        t95[idx[hit]] = tc_hit[hit]
        Qprev[idx] = Qo; tprev[idx] = t[idx]
        active[idx[hit]] = False
        step += 1
    y_eq = A_eq/b
    return t95, Hmax, y_eq, te, Qin_eq

def simulate(d, n_iter=15, verbose=True):
    n = len(d['R24'])
    x = np.full(n, np.log(600.0)); xp = fp = None
    hist = []
    for k in range(n_iter):
        tc = np.exp(x)
        i = mononobe(d['R24'], tc)/1000/3600
        t95, Hmax, y_eq, te, Qin_eq = node_t95(d, i)
        Qd = 0.95*Qin_eq
        tdp, Vt = downpipe_lag(Qd, d['D_dp'], d['H_b'])
        tcp, sur, Vp = pipe_lag(d['N_dp']*Qd, d['D_p'], d['n_p'], d['S_p'], d['L_p'])
        tc_new = t95 + tdp + tcp
        rel = np.nanmax(np.abs(tc_new-tc)/tc)
        hist.append(rel)
        if verbose: print(f'iter {k+1}: max rel change {rel:.5f}', flush=True)
        if rel < 1e-3:
            tc = tc_new; break
        f = np.log(tc_new) - x
        if fp is None:
            xn = x + 0.5*f
        else:
            den = f - fp
            sec = x - f*(x-xp)/np.where(np.abs(den) > 1e-12, den, np.nan)
            xn = np.where(np.isfinite(sec), sec, x + 0.5*f)
            xn = np.clip(xn, x-1.0, x+1.0)
        xp, fp, x = x, f, xn
        tc = tc_new
    out = dict(tc=tc, i_mmh=i*1000*3600, t_node=t95, t_roof=te, t_dp=tdp, t_cp=tcp,
               Hmax=Hmax, y_eq=y_eq, Q_dp=Qd, surcharge=sur, Vt=Vt, conv=np.array(hist))
    return out
