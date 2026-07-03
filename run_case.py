"""
Coupled (beta_e, theta) rotation-controlled BVP solver.
EDIT THE "CONFIG" BLOCK BELOW to change L and boundary conditions.
Requires coupled_sym.pkl (built once by build_coupled.py).

Run:  python3 run_case.py
  -> saves results to <OUT_NAME>.pkl  (read this with plot_results.py)
"""
import sympy as sp
import numpy as np
import pickle, time
from scipy.integrate import solve_bvp

# ======================================================================
# CONFIG -- edit this block to explore different geometries / clamps
# ======================================================================
L = 200.0                      # tape length for this run [mm]

# Each end: (value, type). type = 'tight' -> beta_e,1 = 0 also enforced
#                          type = 'loose' -> natural BC (dR/dbe'' = 0) instead
END_LEFT  = (0.95, 'tight')   # e.g. clamp flattened to 90deg, rigid
END_RIGHT = (0.95,    'tight')   # e.g. clamp held at natural beta0, rigid

TH_LIST = np.concatenate([np.linspace(0.005, 0.12, 8),
                           np.linspace(0.135, 0.5, 20)])   # rotation sweep [rad]

OUT_NAME = 'case_custom'       # output file -> case_custom.pkl
TIME_BUDGET_SEC = 240
# ======================================================================

t0 = time.time()
d = pickle.load(open('coupled_sym.pkl', 'rb'))
b0s, b1s, b2s, b3s, be, be1, be2, M = d['syms']

f_c4  = sp.lambdify((b0s,b1s,b2s,b3s,M), d['coeff4'],  modules='numpy', cse=True)
f_rem = sp.lambdify((b0s,b1s,b2s,b3s,M), d['rem'],     modules='numpy', cse=True)
f_TH  = sp.lambdify((be,be1,be2,M),      d['TH'],      modules='numpy', cse=True)
f_nbc = sp.lambdify((be,be1,be2,M),      d['dR_db2'],  modules='numpy', cse=True)
print("lambdified", time.time()-t0, flush=True)

beta0_natural = 0.95   # must match the value baked into build_coupled.py

def rhs(s, y, p):
    Mv = p[0]
    y4 = -f_rem(y[0],y[1],y[2],y[3],Mv) / f_c4(y[0],y[1],y[2],y[3],Mv)
    return np.vstack([y[1], y[2], y[3], y4, f_TH(y[0],y[1],y[2],Mv)])

def make_bc(thL, endL, endR):
    valL, typeL = endL
    valR, typeR = endR
    def bc(ya, yb, p):
        Mv = p[0]
        c0 = ya[0] - valL
        c1 = ya[1] if typeL == 'tight' else f_nbc(ya[0], ya[1], ya[2], Mv)
        c2 = yb[0] - valR
        c3 = yb[1] if typeR == 'tight' else f_nbc(yb[0], yb[1], yb[2], Mv)
        c4 = ya[4]              # Theta(0) = 0 (reference)
        c5 = yb[4] - thL        # Theta(L) = thL  (rotation control)
        return np.array([c0, c1, c2, c3, c4, c5])
    return bc

x = np.linspace(0, L, 161)
y = np.zeros((5, x.size))
y[0] = beta0_natural \
     + (END_LEFT[0]  - beta0_natural)*np.exp(-x/12.) \
     + (END_RIGHT[0] - beta0_natural)*np.exp(-(L-x)/12.)
p = np.array([0.0])

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
    results.append(dict(thL=thL, M=float(p[0]), s_fold=float(ss[imin]),
                         be_fold=float(prof[0][imin]), s=ss, be=prof[0].copy(),
                         be1=prof[1].copy()))
    print(f"thL={thL:7.4f}  M={p[0]:9.2f} N*mm   min be={np.degrees(prof[0][imin]):6.2f} deg @ {ss[imin]:6.1f} mm",
          flush=True)

meta = dict(L=L, END_LEFT=END_LEFT, END_RIGHT=END_RIGHT)
pickle.dump(dict(meta=meta, results=results), open(f'{OUT_NAME}.pkl', 'wb'))
print(f"\nsaved {OUT_NAME}.pkl  ({len(results)} converged points)   total time {time.time()-t0:.1f}s")
