"""
Coupled (beta_e, theta) rotation-controlled BVP solver -- TIP FORCE version.

Loading: transverse force P at the free tip of a cantilever
    =>  M(s1) = P * (L - s1),   dM/ds1 = -P
(undeformed moment-arm approximation; stated explicitly as an assumption).

Unknown parameter: P  (found by solve_bvp so that theta(L) = thL).
Rotation control on the TIP rotation thL lets us track past the limit
point of P, exactly as the constant-M code tracked past the peak of M.

Boundary conditions (6 total: 4th-order be-ODE + 1st-order Theta + 1 param):
  LEFT (s1=0, the clamped root):
      be(0) = END_LEFT value
      'tight' -> be,1(0) = 0     |  'loose' -> dR/dbe2 = 0
  RIGHT (s1=L, the loaded tip), configurable:
      'free'  -> two natural BCs: dR/dbe2 = 0  AND  dR/dbe1 - d/ds(dR/dbe2) = 0
                 (no cross-section fixture at the tip; M(L)=0 there)
      (value,'tight'/'loose') -> same clamp semantics as before
  Theta(0) = 0,  Theta(L) = thL.

Requires coupled_force_sym.pkl (built once by build_coupled_force.py).
Run:  python3 run_case_force.py  ->  saves <OUT_NAME>.pkl
The output is plot_results.py-compatible: r['M'] stores the ROOT moment
P*L (the quantity that peaks), and r['P'] stores the tip force itself.
"""
import sympy as sp
import numpy as np
import pickle, time
from scipy.integrate import solve_bvp

# ======================================================================
# CONFIG
# ======================================================================
L = 200.0                      # tape length [mm]
TH_SCALE= L/200.0
END_LEFT  = (np.pi/2 , 'tight')    # clamped root: (be value, 'tight'/'loose')
END_RIGHT = (np.pi/2, 'tight')            # 'free'  or  (value, 'tight'/'loose')

TH_LIST = np.concatenate([np.linspace(0.002, 0.06, 8),
                          np.linspace(0.07, 0.35, 24)])*TH_SCALE   # tip rotation sweep [rad]

OUT_NAME = 'case_force'
TIME_BUDGET_SEC = 240
# ======================================================================

t0 = time.time()
d = pickle.load(open('coupled_force_sym.pkl', 'rb'))
b0s, b1s, b2s, b3s, be, be1, be2, M, Ms, M1s = d['syms']

f_c4  = sp.lambdify((b0s, b1s, b2s, b3s, Ms, M1s), d['coeff4'], modules='numpy', cse=True)
f_rem = sp.lambdify((b0s, b1s, b2s, b3s, Ms, M1s), d['rem'],    modules='numpy', cse=True)
f_TH  = sp.lambdify((be, be1, be2, M),             d['TH'],     modules='numpy', cse=True)
f_nbc = sp.lambdify((be, be1, be2, M),             d['dR_db2'], modules='numpy', cse=True)
f_q3  = sp.lambdify((b0s, b1s, b2s, b3s, Ms, M1s), d['Q3'],     modules='numpy', cse=True)
print("lambdified", time.time() - t0, flush=True)

beta0_natural = 0.95   # must match build_coupled_force.py

def rhs(s, y, p):
    P = p[0]
    Mv = P * (L - s)          # prescribed moment distribution
    M1 = -P                   # dM/ds1
    y4 = -f_rem(y[0], y[1], y[2], y[3], Mv, M1) / f_c4(y[0], y[1], y[2], y[3], Mv, M1)
    return np.vstack([y[1], y[2], y[3], y4, f_TH(y[0], y[1], y[2], Mv)])

def make_bc(thL, endL, endR):
    valL, typeL = endL
    def bc(ya, yb, p):
        P = p[0]
        ML, MR = P * L, 0.0          # M at root and at tip
        c0 = ya[0] - valL
        c1 = ya[1] if typeL == 'tight' else f_nbc(ya[0], ya[1], ya[2], ML)
        if endR == 'free':
            c2 = f_nbc(yb[0], yb[1], yb[2], MR)
            c3 = f_q3(yb[0], yb[1], yb[2], yb[3], MR, -P)
        else:
            valR, typeR = endR
            c2 = yb[0] - valR
            c3 = yb[1] if typeR == 'tight' else f_nbc(yb[0], yb[1], yb[2], MR)
        c4 = ya[4]              # Theta(0) = 0
        c5 = yb[4] - thL        # Theta(L) = thL  (tip-rotation control)
        return np.array([c0, c1, c2, c3, c4, c5])
    return bc

x = np.linspace(0, L, int(161*TH_SCALE))
y = np.zeros((5, x.size))
y[0] = beta0_natural + (END_LEFT[0] - beta0_natural) * np.exp(-x / 12.)
if END_RIGHT != 'free':
    y[0] += (END_RIGHT[0] - beta0_natural) * np.exp(-(L - x) / 12.)
p = np.array([1e-4])

results = []
deadline = time.time() + TIME_BUDGET_SEC
for thL in TH_LIST:
    if time.time() > deadline:
        print("time budget reached, stopping sweep"); break
    sol = solve_bvp(rhs, make_bc(thL, END_LEFT, END_RIGHT), x, y, p=p,
                    max_nodes=60000, tol=1e-5)
    if not sol.success:
        print(f"thL={thL:.4f}  FAILED ({sol.message}) -> likely near a limit point")
        break
    x, y, p = sol.x, sol.y, sol.p
    ss = np.linspace(0, L, 2000)
    prof = sol.sol(ss)
    imin = int(np.argmin(prof[0]))
    P = float(p[0])
    results.append(dict(thL=thL, P=P, M=P * L,           # M = root moment (peaks)
                        s_fold=float(ss[imin]), be_fold=float(prof[0][imin]),
                        s=ss, be=prof[0].copy(), be1=prof[1].copy(),
                        Mdist=(P * (L - ss)).copy()))
    print(f"thL={thL:7.4f}  P={P:9.4f} N   M_root={P*L:9.2f} N*mm   "
          f"min be={np.degrees(prof[0][imin]):6.2f} deg @ {ss[imin]:6.1f} mm", flush=True)

meta = dict(L=L, END_LEFT=END_LEFT, END_RIGHT=END_RIGHT, loading='tip_force')
pickle.dump(dict(meta=meta, results=results), open(f'{OUT_NAME}.pkl', 'wb'))
print(f"\nsaved {OUT_NAME}.pkl  ({len(results)} converged points)   "
      f"total time {time.time() - t0:.1f}s")
