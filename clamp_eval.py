"""
Reusable single-config evaluator, refactored from run_case_multiclamp.py.

evaluate_config(interior_positions, ...) builds the CLAMPS list
(root@0, given interior positions, tip@L), runs the stacked-BVP theta_L
sweep with early stop at the descending branch (limit point), and returns
the same peak-diagnostics dict run_case_multiclamp.py prints/saves,
without any file I/O, for use inside a search loop.
"""
import sympy as sp
import numpy as np
import pickle, time
from scipy.integrate import solve_bvp

# ---- load + lambdify symbolic model once at import time -------------------
_d = pickle.load(open('coupled_force_sym.pkl', 'rb'))
_b0s, _b1s, _b2s, _b3s, _be, _be1, _be2, _M, _Ms, _M1s, _M2s = _d['syms']
_f_c4  = sp.lambdify((_b0s, _b1s, _b2s, _b3s, _Ms, _M1s, _M2s), _d['coeff4'], modules='numpy', cse=True)
_f_rem = sp.lambdify((_b0s, _b1s, _b2s, _b3s, _Ms, _M1s, _M2s), _d['rem'],    modules='numpy', cse=True)
_f_TH  = sp.lambdify((_be, _be1, _be2, _M),                     _d['TH'],     modules='numpy', cse=True)

BETA0_NATURAL = 0.95
DECAY_LEN_GUESS = 12.0
BE_FRAC_LIMIT_THEORY = 0.76

TH_LIST_DEFAULT = np.concatenate([np.linspace(0.01, 0.20, 8),
                                   np.linspace(0.22, 0.60, 30)])


def evaluate_config(interior_positions, L=1000.0, N_TAPE=1, G=9.81,
                     TAPE_G_PER_70CM=20.0, PLATE_MASS_G=80.0,
                     clamp_value=np.pi/2, th_list=None,
                     th_start=0.01, th_max=0.70,
                     th_step_coarse=0.030, th_step_fine=0.013,
                     be_frac_diff_med=0.15, be_frac_diff_fine=0.05,
                     th_peak_hint=None, hint_margin=0.06,
                     warm_start=None, snapshot_be_frac=0.85,
                     target_M_hi=None,
                     time_budget_sec=600, fail_time_budget_sec=240,
                     n_bisect_max=6, th_gap_min=1e-3,
                     verbose=False, stop_check=None):
    """
    interior_positions: sorted list/array of clamp positions strictly
        between 0 and L (the "plates" being optimized). Root (0) and tip
        (L) clamps are added automatically.

    target_M_hi: optional early-exit threshold. M_root climbs roughly
        monotonically until near the limit point, so if a converged step
        already has M_root > target_M_hi, the TRUE peak can only be even
        higher (further over target). There's no need to keep tracing to
        the real limit point just to learn the config overshoots -- the
        sweep stops immediately and returns the crossing value as peak_M
        with overshoot_skip=True (peak_traced stays False; this is a
        confirmed lower bound on the overshoot, not the true peak, but
        the OVER/window verdict it implies is already certain). Pass
        WIN_HI from the caller's target window here to skip hopeless
        candidates fast during a search.

    Speed features (accuracy-preserving: mesh/max_nodes/tol untouched):

    th_list: if given, use this EXACT fixed theta_L grid (old behavior,
        bit-for-bit). If None (default), step adaptively:
          coarse (th_step_coarse=0.030) while clearly below the limit
          point, fine (th_step_fine=0.013 -- the original grid spacing,
          which resolves the peak accurately) once close. "Close" means
          be_frac within be_frac_diff_med of the 0.76 theory value, OR
          thL within hint_margin of th_peak_hint if a hint is given.
          Within be_frac_diff_fine of 0.76 the step is ALWAYS fine,
          regardless of the hint (the hint may be wrong for this config).
    th_peak_hint: neighbor config's peak_thL (from the search cache) --
        lets the sweep stay coarse right up to just below the expected
        peak even when be_frac climbs slowly.
    warm_start: dict(x=..., Y=..., p=..., thL=..., n_seg=...) -- a
        converged solution snapshot from a NEIGHBORING config (same
        number of segments). The sweep then STARTS at that thL with that
        solution as the initial guess, skipping the whole climbing branch
        (~10+ solves). If the very first warm solve fails, falls back to
        a cold start from th_start automatically. Snapshots transfer well
        between configs because in the per-segment xi coordinates the
        boundary layers always sit at xi=0/1 regardless of clamp spacing.
    fail_time_budget_sec: cumulative wall-time allowed in FAILED solves
        (the expensive mesh-overflow ones near the limit point). Once
        exceeded, the last success is accepted as the practical peak and
        the sweep stops -- bounds the 300-700s pathological tails without
        touching the mesh of successful solves.

    stop_check: optional zero-arg callable, checked before every theta_L
        step. If it returns True the sweep stops immediately.

    Returns dict with keys: positions, n_plates, peak_M, peak_thL,
        peak_be_frac, s_fold (fold location [mm] at the peak),
        peak_traced, overshoot_skip (bool -- True if stopped early via
        target_M_hi, meaning peak_M understates the real overshoot),
        n_steps, elapsed_sec, and snapshot (warm-start dict for reuse by
        neighboring configs; pop it before pickling logs if size matters).
    """
    W_SELF = (TAPE_G_PER_70CM * 1e-3 * G) / 700.0
    W_PLATE = PLATE_MASS_G * 1e-3 * G

    positions = [0.0] + [float(p) for p in interior_positions] + [L]
    assert all(positions[i] < positions[i+1] for i in range(len(positions)-1)), \
        "interior_positions must be strictly increasing and within (0, L)"
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

    def rhs(xi, Y, p):
        P = p[0]
        out = np.empty_like(Y)
        for i in range(n_seg):
            y = Y[5*i:5*i+5]
            s = positions[i] + xi * D[i]
            Mv = M_of(s, P)
            M1 = M1_of(s, P)
            y4 = -_f_rem(y[0], y[1], y[2], y[3], Mv, M1, W_SELF) \
                 / _f_c4(y[0], y[1], y[2], y[3], Mv, M1, W_SELF)
            out[5*i+0] = D[i] * y[1]
            out[5*i+1] = D[i] * y[2]
            out[5*i+2] = D[i] * y[3]
            out[5*i+3] = D[i] * y4
            out[5*i+4] = D[i] * _f_TH(y[0], y[1], y[2], Mv)
        return out

    def make_bc(thL):
        def bc(ya, yb, p):
            c = [ya[0] - clamp_value, ya[1], ya[4]]
            for j in range(1, n_seg):
                iL, iR = 5*(j-1), 5*j
                c += [yb[iL] - clamp_value, yb[iL+1],
                      ya[iR] - clamp_value, ya[iR+1],
                      ya[iR+4] - yb[iL+4]]
            iT = 5*(n_seg-1)
            c += [yb[iT] - clamp_value, yb[iT+1], yb[iT+4] - thL]
            return np.array(c)
        return bc

    tt = np.linspace(0, 1, 141)
    xi0 = 0.5 * (1 - np.cos(np.pi * tt))
    Y0 = np.zeros((5 * n_seg, xi0.size))
    for i in range(n_seg):
        s = positions[i] + xi0 * D[i]
        Y0[5*i] = BETA0_NATURAL \
            + (clamp_value - BETA0_NATURAL) * np.exp(-(s - positions[i]) / DECAY_LEN_GUESS) \
            + (clamp_value - BETA0_NATURAL) * np.exp(-(positions[i+1] - s) / DECAY_LEN_GUESS)
    p0 = np.array([1e-4])

    # ---- initial state: warm start from a neighbor's snapshot if usable ----
    cold_x, cold_Y, cold_p = xi0, Y0, p0
    x, Y, p = xi0, Y0, p0
    thL = th_start
    warm_active = False
    if warm_start is not None and warm_start.get('n_seg') == n_seg:
        x = np.asarray(warm_start['x']).copy()
        Y = np.asarray(warm_start['Y']).copy()
        p = np.asarray(warm_start['p']).copy()
        thL = float(warm_start['thL'])
        warm_active = True
        print(f"    [warm] starting sweep at thL={thL:.4f} from a neighbor's solution",
              flush=True)
    elif warm_start is not None:
        print(f"    [warm] snapshot has n_seg={warm_start.get('n_seg')} != {n_seg}, "
              f"ignoring (cold start)", flush=True)

    results = []
    th_prev_ok = None
    bisect_left = n_bisect_max
    deadline = time.time() + time_budget_sec
    fail_time = 0.0
    peak_seen = False
    overshoot_skip = False
    snapshot = None
    t0 = time.time()
    _last_t = t0

    use_fixed_grid = th_list is not None
    queue = list(th_list) if use_fixed_grid else None

    while True:
        if use_fixed_grid:
            if not queue:
                break
            thL = queue.pop(0)
            if th_prev_ok is not None and thL <= th_prev_ok + 1e-12:
                continue
        else:
            if thL is None or thL > th_max:
                break

        if time.time() > deadline:
            print("    [stopped] time budget exceeded", flush=True)
            break
        if stop_check is not None and stop_check():
            print("    [stopped] quit requested by caller", flush=True)
            break

        sol = solve_bvp(rhs, make_bc(thL), x, Y, p=p, max_nodes=25000, tol=1e-5)
        step_dt = time.time() - t0 if not results else time.time() - _last_t
        _last_t = time.time()

        if sol.success:
            warm_active = False
            x, Y, p = sol.x, sol.y, sol.p
            P = float(p[0])
            Mroot = float(M_of(0.0, P))
            # min be along tape + its LOCATION (fold position)
            ss = np.linspace(0, L, 800)
            be_all = np.empty_like(ss)
            for i in range(n_seg):
                m = (ss >= positions[i] - 1e-9) & (ss <= positions[i+1] + 1e-9)
                xi_m = (ss[m] - positions[i]) / D[i]
                be_all[m] = sol.sol(np.clip(xi_m, 0, 1))[5*i]
            be_fold = float(be_all.min())
            s_fold = float(ss[int(np.argmin(be_all))])
            be_frac = be_fold / BETA0_NATURAL
            is_desc = len(results) > 0 and Mroot < results[-1]['M'] - 1e-9
            results.append(dict(thL=float(thL), P=P, M=Mroot,
                                be_fold=be_fold, s_fold=s_fold))
            th_prev_ok = thL
            bisect_left = n_bisect_max
            # keep the latest pre-limit-point solution as a reusable
            # warm-start snapshot for neighboring configs
            if be_frac >= snapshot_be_frac or snapshot is None:
                snapshot = dict(x=x.copy(), Y=Y.copy(), p=p.copy(),
                                thL=float(thL), n_seg=n_seg)
            print(f"    [step] thL={thL:.4f} M={Mroot:.1f} be_frac={be_frac*100:.1f}% "
                  f"s_fold={s_fold:.1f}mm nodes={sol.x.size} dt={step_dt:.1f}s", flush=True)
            if target_M_hi is not None and Mroot > target_M_hi:
                print(f"    [skip] M={Mroot:.1f} already exceeds target_M_hi="
                      f"{target_M_hi:.1f} mid-climb -- true peak can only be higher, "
                      f"stopping early (confirmed OVER)", flush=True)
                overshoot_skip = True
                break
            if is_desc:
                peak_seen = True
                break
            if not use_fixed_grid:
                diff = abs(be_frac - BE_FRAC_LIMIT_THEORY)
                if diff <= be_frac_diff_fine:
                    dth = th_step_fine        # near the peak: always fine
                elif th_peak_hint is not None:
                    dth = th_step_coarse if thL < th_peak_hint - hint_margin else th_step_fine
                else:
                    dth = th_step_coarse if diff > be_frac_diff_med else th_step_fine
                thL = thL + dth
            continue

        # ---------------- failure path ----------------
        fail_time += step_dt
        print(f"    [fail] thL={thL:.4f} nodes={sol.x.size} msg={sol.message} "
              f"dt={step_dt:.1f}s (fail_time={fail_time:.0f}s)", flush=True)
        if th_prev_ok is None:
            if warm_active:
                # warm start didn't transfer -- fall back to a cold sweep
                print("    [warm] warm start failed -> falling back to cold start",
                      flush=True)
                x, Y, p = cold_x, cold_Y, cold_p
                thL = th_start
                warm_active = False
                continue
            break
        if fail_time > fail_time_budget_sec:
            print(f"    [stopped] failed-solve time budget ({fail_time_budget_sec:.0f}s) "
                  f"exhausted -- accepting last success as the practical peak", flush=True)
            break
        gap = thL - th_prev_ok
        if bisect_left > 0 and gap > th_gap_min:
            mid = th_prev_ok + 0.5 * gap
            bisect_left -= 1
            if use_fixed_grid:
                queue.insert(0, mid)
            thL = mid
            continue
        break

    if not results:
        return dict(positions=positions[1:-1], n_plates=len(interior_positions),
                     peak_M=None, peak_thL=None, peak_be_frac=None, s_fold=None,
                     peak_traced=False, overshoot_skip=False, n_steps=0,
                     elapsed_sec=time.time()-t0, snapshot=None)

    if overshoot_skip:
        # the crossing point IS the report -- it's a confirmed lower bound
        # on the overshoot, not the true (higher) peak, and no further
        # tracing is needed to know this config is OVER target.
        peak = results[-1]
    else:
        peak = results[-2] if (peak_seen and len(results) >= 2) else max(results, key=lambda r: r['M'])
    return dict(positions=positions[1:-1], n_plates=len(interior_positions),
                peak_M=peak['M'], peak_thL=peak['thL'],
                peak_be_frac=peak['be_fold']/BETA0_NATURAL, s_fold=peak['s_fold'],
                peak_traced=bool(peak_seen), overshoot_skip=bool(overshoot_skip),
                n_steps=len(results), elapsed_sec=time.time()-t0, snapshot=snapshot)