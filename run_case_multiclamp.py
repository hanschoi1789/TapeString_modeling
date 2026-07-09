"""
Multi-clamp tape spring BVP -- STACKED single-BVP formulation.

WHY THIS EXISTS
---------------
The previous multiclamp solver marched segment-by-segment at FIXED P and
wrapped an outer secant on P to hit theta(L)=thL. Near the limit point the
map thL(P) becomes near-vertical / multivalued, so the outer secant stalls
or diverges -- refining the mesh or the thL grid cannot fix this because it
is a parameterization problem, not a resolution problem.

The single-clamp solver never had this issue: there P is an UNKNOWN
PARAMETER of one solve_bvp call, so Newton solves (solution, P) jointly and
walks straight past the limit point (we observed M_root rise to 5601 and
then FALL to 5470 before stopping).

This script gives the multi-clamp problem the same structure: all segments
are stacked into ONE solve_bvp system on xi in [0,1]:
    state  Y = [be_i, be_i', be_i'', be_i''', Theta_i  for each segment i]
    (primes = d/ds1; d/dxi = D_i * d/ds1 with D_i the segment length)
    params p = [P]
    BCs: root clamp (3) + tip clamp (2) + 5 per interior clamp
         (be=val on both sides, be'=0 both sides, Theta continuity)
         + Theta_last(1) = thL   ->  5*n_seg + 1 total.
Interior plate point-weights and distributed self-weight enter through
M(s1) exactly as before (M''=w needs the M2s-aware coupled_force_sym.pkl).

All clamps are 'tight' (be and be' prescribed). 'loose' would need the
natural-BC variant per clamp; not implemented here.

Run:  python3 run_case_multiclamp_stacked.py   ->  case_multiclamp.pkl
Output schema identical to the old script (plot_multiclamp.py compatible).
"""
import sympy as sp
import numpy as np
import pickle, time
from scipy.integrate import solve_bvp

# ======================================================================
# CONFIG
# ======================================================================
L = 1000.0

N_TAPE = 1
G = 9.81
TAPE_G_PER_70CM = 20.0
W_SELF = (TAPE_G_PER_70CM * 1e-3 * G) / 700.0
PLATE_MASS_G = 80.0
W_PLATE = PLATE_MASS_G * 1e-3 * G

CLAMPS = [
    (0.0,    np.pi/2, 'tight'),
    #(100.0,  np.pi/2, 'tight'),
    #(250.0,  np.pi/2, 'tight'),
    #(400.0,  np.pi/2, 'tight'),
    #(650.0,  np.pi/2, 'tight'),
    (L,      np.pi/2, 'tight'),
]

TH_LIST = np.concatenate([np.linspace(0.01, 0.20, 8),
                          np.linspace(0.22, 0.60, 30)])

OUT_NAME = 'case_multiclamp'
TIME_BUDGET_SEC = 1200
BE_FRAC_LIMIT_THEORY = 0.76   # Martin(2020) uniform-solution limit-point criterion
N_BISECT_MAX = 6              # bisections toward last success before giving up
TH_GAP_MIN = 1e-3             # stop bisecting once the thL bracket is this small
# ======================================================================

t0 = time.time()

def log(msg):
    print(msg, flush=True)

d = pickle.load(open('coupled_force_sym.pkl', 'rb'))
b0s, b1s, b2s, b3s, be, be1, be2, M, Ms, M1s, M2s = d['syms']
f_c4  = sp.lambdify((b0s, b1s, b2s, b3s, Ms, M1s, M2s), d['coeff4'], modules='numpy', cse=True)
f_rem = sp.lambdify((b0s, b1s, b2s, b3s, Ms, M1s, M2s), d['rem'],    modules='numpy', cse=True)
f_TH  = sp.lambdify((be, be1, be2, M),                  d['TH'],     modules='numpy', cse=True)
log(f"lambdified symbolic model  ({time.time()-t0:.1f}s)")

beta0_natural = 0.95
DECAY_LEN_GUESS = 12.0

positions = [c[0] for c in CLAMPS]
values    = [c[1] for c in CLAMPS]
types     = [c[2] for c in CLAMPS]
assert all(t == 'tight' for t in types), "stacked solver: tight clamps only"
assert positions[0] == 0.0 and abs(positions[-1] - L) < 1e-9
n_seg = len(positions) - 1
D = np.array([positions[i+1] - positions[i] for i in range(n_seg)])

_plate_pos = np.array([p for p in positions if p > 1e-12])

def M_of(s, P):
    s = np.asarray(s, dtype=float)
    base = (P / N_TAPE) * (L - s) + 0.5 * W_SELF * (L - s) ** 2
    for pj in _plate_pos:
        base = base + (W_PLATE / N_TAPE) * np.clip(pj - s, 0.0, None)
    return base

def M1_of(s, P):
    s = np.asarray(s, dtype=float)
    d1 = -(P / N_TAPE) - W_SELF * (L - s)
    for pj in _plate_pos:
        d1 = d1 - (W_PLATE / N_TAPE) * (s < pj)
    return d1

# ---------------------------------------------------------------------
def rhs(xi, Y, p):
    P = p[0]
    out = np.empty_like(Y)
    for i in range(n_seg):
        y = Y[5*i:5*i+5]
        s = positions[i] + xi * D[i]
        Mv = M_of(s, P)
        M1 = M1_of(s, P)
        y4 = -f_rem(y[0], y[1], y[2], y[3], Mv, M1, W_SELF) \
             / f_c4(y[0], y[1], y[2], y[3], Mv, M1, W_SELF)
        out[5*i+0] = D[i] * y[1]
        out[5*i+1] = D[i] * y[2]
        out[5*i+2] = D[i] * y[3]
        out[5*i+3] = D[i] * y4
        out[5*i+4] = D[i] * f_TH(y[0], y[1], y[2], Mv)
    return out

def make_bc(thL):
    def bc(ya, yb, p):
        c = []
        # root clamp (segment 0, xi=0)
        c.append(ya[0] - values[0])
        c.append(ya[1])
        c.append(ya[4])                       # Theta(0) = 0
        # interior clamps j = 1 .. n_seg-1
        for j in range(1, n_seg):
            iL, iR = 5*(j-1), 5*j
            c.append(yb[iL]   - values[j])    # left side be = val
            c.append(yb[iL+1])                # left side be' = 0
            c.append(ya[iR]   - values[j])    # right side be = val
            c.append(ya[iR+1])                # right side be' = 0
            c.append(ya[iR+4] - yb[iL+4])     # Theta continuity
        # tip clamp (last segment, xi=1)
        iT = 5*(n_seg-1)
        c.append(yb[iT]   - values[-1])
        c.append(yb[iT+1])
        # rotation control
        c.append(yb[iT+4] - thL)
        return np.array(c)
    return bc

# ---------------------------------------------------------------------
# initial guess: Chebyshev-clustered xi grid (dense near both segment ends
# where the clamp boundary layers live)
tt = np.linspace(0, 1, 141)
xi0 = 0.5 * (1 - np.cos(np.pi * tt))
Y0 = np.zeros((5 * n_seg, xi0.size))
for i in range(n_seg):
    s = positions[i] + xi0 * D[i]
    Y0[5*i] = beta0_natural \
        + (values[i]   - beta0_natural) * np.exp(-(s - positions[i])   / DECAY_LEN_GUESS) \
        + (values[i+1] - beta0_natural) * np.exp(-(positions[i+1] - s) / DECAY_LEN_GUESS)
p0 = np.array([1e-4])

def assemble_profile(sol, n=4000):
    ss = np.linspace(0, L, n)
    be_all = np.empty_like(ss); be1_all = np.empty_like(ss)
    for i in range(n_seg):
        m = (ss >= positions[i] - 1e-9) & (ss <= positions[i+1] + 1e-9)
        xi_m = (ss[m] - positions[i]) / D[i]
        prof = sol.sol(np.clip(xi_m, 0, 1))
        be_all[m]  = prof[5*i]
        be1_all[m] = prof[5*i+1]
    return ss, be_all, be1_all

# ---------------------------------------------------------------------
results = []
deadline = time.time() + TIME_BUDGET_SEC
log(f"\nSTACKED multi-clamp sweep: L={L:.0f}, {len(CLAMPS)} clamps at {positions}")
log(f"segments={n_seg}, state dim={5*n_seg}, weights: N_TAPE={N_TAPE}, "
    f"w_self={W_SELF:.4e} N/mm, W_plate={W_PLATE:.4f} N\n")

x, Y, p = xi0, Y0, p0
peak_seen = False
th_prev_ok = None
bisect_left = N_BISECT_MAX
queue = list(TH_LIST)
k = 0

while queue:
    if time.time() > deadline:
        log("time budget reached, stopping"); break
    thL = queue.pop(0)
    if th_prev_ok is not None and thL <= th_prev_ok + 1e-12:
        continue   # stale target left behind by an earlier bisection
    tstep = time.time()
    sol = solve_bvp(rhs, make_bc(thL), x, Y, p=p, max_nodes=25000, tol=1e-5)
    dt = time.time() - tstep

    if sol.success:
        x, Y, p = sol.x, sol.y, sol.p
        P = float(p[0])
        ss, be_prof, be1_prof = assemble_profile(sol)
        imin = int(np.argmin(be_prof))
        Mroot = float(M_of(0.0, P))

        is_desc = len(results) > 0 and Mroot < results[-1]['M'] - 1e-9
        tag = "[<<<]" if is_desc else "[OK] "
        log(f"{tag} step {k:2d}  thL={thL:7.4f}  P={P:9.4f} N  M_root={Mroot:10.2f} N*mm  "
            f"min be={np.degrees(be_prof[imin]):6.2f}deg @ {ss[imin]:6.1f}mm  "
            f"nodes={sol.x.size:5d}  {dt:5.1f}s")

        results.append(dict(thL=float(thL), P=P, M=Mroot,
                            s_fold=float(ss[imin]), be_fold=float(be_prof[imin]),
                            s=ss, be=be_prof.copy(), be1=be1_prof.copy(),
                            Mdist=M_of(ss, P).copy()))
        k += 1
        th_prev_ok = thL
        bisect_left = N_BISECT_MAX   # reset the bisection budget after any success

        if is_desc:
            peak_seen = True
            log("\n      >>> dM/dthL changed sign (M decreased): LIMIT POINT found, stopping <<<")
            break
        continue

    # ---- step FAILED: bisect toward the last success instead of just stopping ----
    log(f"[fail] thL={thL:.4f}  ({sol.message})  {dt:.1f}s")
    if th_prev_ok is None:
        log("   first step failed -> numerical setup problem, stopping")
        break
    gap = thL - th_prev_ok
    if bisect_left > 0 and gap > TH_GAP_MIN:
        mid = th_prev_ok + 0.5 * gap
        bisect_left -= 1
        log(f"   bisecting: retrying at thL={mid:.4f} (bracket {gap:.4f}, "
            f"{bisect_left} bisections left)")
        queue.insert(0, mid)
        continue
    # bisection exhausted: this is as close as we can numerically get.
    # If be_fold is already well under the natural angle, the physics says
    # we ARE past (or essentially at) the limit point -- just resolution-limited.
    be_frac_last = results[-1]['be_fold'] / beta0_natural if results else 1.0
    log(f"   bisection exhausted (be/natural={be_frac_last*100:.0f}%) -- "
        f"{'treating last point as the practical limit' if be_frac_last <= BE_FRAC_LIMIT_THEORY + 0.05 else 'this looks like a numerical failure, not a real limit point'}")
    break

log("\n================ limit-point / snap-through report ================")
if results:
    # if we stopped on a descent, the peak is the second-to-last recorded point;
    # otherwise (never turned down) just report the max seen so far.
    if peak_seen and len(results) >= 2:
        peak = results[-2]
    else:
        peak = max(results, key=lambda r: r['M'])
    be_peak = peak['be_fold']
    be_frac = be_peak / beta0_natural
    reduction_pct = (1.0 - be_frac) * 100.0
    meets_theory = be_frac <= BE_FRAC_LIMIT_THEORY + 0.03   # small tolerance band

    log(f"STATUS: {'genuine limit point (M peaked then fell)' if peak_seen else 'no descent observed yet'}")
    log(f"peak M_root  = {peak['M']:.1f} N*mm  at thL = {peak['thL']:.4f}")
    log(f"be at peak   = {np.degrees(be_peak):.2f} deg   (natural = {np.degrees(beta0_natural):.2f} deg)")
    log(f"be / natural = {be_frac*100:.1f}%   ->  reduced by {reduction_pct:.1f}% from natural")
    log(f"theory (Martin 2020 uniform solution) expects ~{BE_FRAC_LIMIT_THEORY*100:.0f}% at the true limit point")
    log(f"  {'MATCHES theory (within tolerance)' if meets_theory else 'DOES NOT match theory -- peak may be premature / numerical'}")
log("=====================================================================")

peak_meta = None
if results:
    _peak = results[-2] if (peak_seen and len(results) >= 2) else max(results, key=lambda r: r['M'])
    peak_meta = dict(M=_peak['M'], thL=_peak['thL'], be_fold=_peak['be_fold'],
                      be_frac_of_natural=_peak['be_fold'] / beta0_natural)

meta = dict(L=L, END_LEFT=(CLAMPS[0][1], CLAMPS[0][2]),
            END_RIGHT=(CLAMPS[-1][1], CLAMPS[-1][2]),
            CLAMPS=CLAMPS, loading='tip_force',
            N_TAPE=N_TAPE, W_SELF=W_SELF, W_PLATE=W_PLATE,
            solver='stacked_single_bvp', peak_traced=bool(peak_seen),
            peak=peak_meta, BE_FRAC_LIMIT_THEORY=BE_FRAC_LIMIT_THEORY)
pickle.dump(dict(meta=meta, results=results), open(f'{OUT_NAME}.pkl', 'wb'))
log(f"saved {OUT_NAME}.pkl ({len(results)} pts, {time.time()-t0:.1f}s)")