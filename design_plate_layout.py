"""
Sequential constructive plate-layout design for a 1000mm tape spring.
(user-specified algorithm, replacing the old window-search approach)

Definitions
-----------
Clamp 1 sits at s1=0 (root), clamp n+1 at s1=L (tip). L_k is the span
between clamp k and clamp k+1. Segments are counted from the root:
L1 = [0, p1], L2 = [p1, p2], ..., L_m = [p_{n}, L].

Algorithm (as specified)
------------------------
STAGE 1 -- find the reference L1:
    With plates ONLY at both ends of a SHORT tape of total length L,
    find L such that the peak root moment at buckling is at most ~5800
    (target window [5600, 5800]). Empirically L=190 gives ~6000, so 190
    is the default reference (set RUN_L1_SEARCH=True to bisect for a
    proper in-window L1 instead of using the fixed reference).

STAGE 2 -- constructive loop on the full 1m tape:
    Start with plates [L1_ref, midpoint(L1_ref, 1000)]. Fold must start
    in L2 (same length as L3 but carries more moment). Then repeat:

      evaluate -> peak_M, s_fold (fold location at the limit point)

      * peak_M in [5600, 5800]         -> DONE (minimal, balanced layout)
      * peak_M > 5800 (overshoot)      -> undo the last move, halve that
                                          move's step (backoff); if the
                                          step is already at its minimum,
                                          stop and report the best
                                          under-window layout seen.
      * fold in L1                     -> shift ALL interior plates left
                                          by SHIFT_STEP (10mm): L1 gets
                                          shorter and therefore stronger.
      * fold in the LAST segment       -> add a plate at that segment's
        (touches the tip)                 midpoint; the loop then starts
                                          shrinking the new left half.
      * fold in an interior segment Lj -> shrink Lj by moving its RIGHT
                                          boundary clamp left by
                                          SHRINK_STEP (10mm). (L_{j+1}
                                          grows; eventually the fold
                                          migrates rightward -> add.)

    Termination is guaranteed a stop by MAX_ITERS / MAX_PLATES guards
    and the q-quit listener.

Every evaluation is cached immediately in search_log.pkl (results) and
warmstart_store.pkl (converged-solution snapshots reused as initial
guesses / theta_L hints by later evaluations) -- interrupt & resume any
time; nothing is recomputed.

Evaluation notes
----------------
evaluate_config is called with target_M_hi=WIN_HI (5800): a config whose
M_root crosses 5800 mid-climb stops immediately (confirmed OVER; the true
peak is even higher). Because M_root rises monotonically to the limit
point, this early exit can never truncate an in-window peak, so every
non-OVER result is a full trace whose s_fold IS the fold location at the
true limit point -- trustworthy for the move decisions above.

Run:  python3 design_plate_layout.py     ('q' + Enter stops immediately)
"""
import numpy as np
import pickle, os, sys, threading, multiprocessing as mp

# ======================================================================
# CONFIG
# ======================================================================
L_TOTAL = 1000.0
N_TAPE = 1

WIN_LO, WIN_HI = 5600.0, 5800.0    # target window for peak M_root [N*mm]

RUN_L1_SEARCH = False              # True: bisect stage-1 L1 into the window
L1_REF = 190.0                     # fixed stage-1 reference (peak ~6000 measured)
L1_SEARCH_BRACKET = (150.0, 320.0) # bisection bracket if RUN_L1_SEARCH

SHIFT_STEP0 = 10.0                 # fold-in-L1: shift all plates left [mm]
SHRINK_STEP0 = 20.0                # fold-in-interior: shrink that segment [mm]
STEP_MIN = 2.0                     # backoff floor; below this we stop
MIN_GAP = 20.0                     # geometric sanity: min clamp spacing [mm]
MAX_PLATES = 8
MAX_ITERS = 60

LOG_FILE = 'search_log.pkl'
WARM_FILE = 'warmstart_store.pkl'
WARM_MAX_DIST = 160.0
# ======================================================================

# ----------------------------------------------------------------------
# 'q' + Enter -> kill the running evaluation subprocess immediately
# ----------------------------------------------------------------------
_stop_requested = threading.Event()


def _quit_listener():
    print("(press 'q' then Enter at any time to stop immediately)", flush=True)
    try:
        for line in sys.stdin:
            if line.strip().lower() == 'q':
                print("\n[quit requested] killing the current evaluation now...", flush=True)
                _stop_requested.set()
                return
    except Exception:
        pass


def _worker(positions, L, n_tape, q, th_peak_hint=None, warm_snapshot=None,
            target_m_hi=None):
    from clamp_eval import evaluate_config
    r = evaluate_config(list(positions), L=L, N_TAPE=n_tape, verbose=True,
                        th_peak_hint=th_peak_hint, warm_start=warm_snapshot,
                        target_M_hi=target_m_hi)
    q.put(r)


# ----------------------------------------------------------------------
# caching / warm-start stores
# ----------------------------------------------------------------------
def load_pkl(fn, default):
    if os.path.exists(fn):
        try:
            return pickle.load(open(fn, 'rb'))
        except Exception:
            return default
    return default


def save_pkl(fn, obj):
    pickle.dump(obj, open(fn, 'wb'))


def key_of(positions, L):
    return (round(float(L), 1),) + tuple(round(float(p), 1) for p in positions)


def nearest_cached(positions, L, log):
    """closest already-solved config with the same L and plate count."""
    want = key_of(positions, L)
    best_key, best_entry, best_d = None, None, float('inf')
    for key, entry in log.items():
        if len(key) != len(want) or key[0] != want[0] or entry.get('peak_M') is None:
            continue
        d = sum(abs(a - b) for a, b in zip(key[1:], want[1:]))
        if d < best_d:
            best_key, best_entry, best_d = key, entry, d
    return best_key, best_entry, best_d


def evaluate_cached(positions, log, L=L_TOTAL):
    """Evaluate one layout (subprocess, killable), with caching, neighbor
    warm start + theta_L hint, and target_M_hi=WIN_HI overshoot early-exit.
    Non-OVER results are full traces, so their s_fold is the fold location
    at the true limit point (see module docstring)."""
    k = key_of(positions, L)
    if k in log:
        c = log[k]
        print(f"  [cache] L={L:.0f} {list(positions)} -> peak_M={c['peak_M']}"
              f" s_fold={c.get('s_fold')}", flush=True)
        return c

    warm = load_pkl(WARM_FILE, {})
    nk, nentry, nd = nearest_cached(positions, L, log)
    hint, snap = None, None
    if nentry is not None:
        hint = nentry.get('peak_thL')
        if nd <= WARM_MAX_DIST and nk in warm:
            snap = warm[nk]
        print(f"  [neighbor] nearest solved {list(nk[1:])} (dist={nd:.0f}mm): "
              f"hint thL={None if hint is None else round(hint, 4)}, "
              f"warm={'yes' if snap is not None else 'no'}", flush=True)

    print(f"  [run] evaluating L={L:.0f} plates={list(positions)} ...", flush=True)
    q = mp.Queue()
    proc = mp.Process(target=_worker, args=(positions, L, N_TAPE, q, hint, snap, WIN_HI))
    proc.start()
    killed = False
    while proc.is_alive():
        if _stop_requested.is_set():
            print(f"  [killed] stopping {list(positions)} immediately (q pressed)", flush=True)
            proc.terminate(); proc.join(timeout=5)
            if proc.is_alive():
                proc.kill(); proc.join()
            killed = True
            break
        proc.join(timeout=0.2)

    if killed or q.empty():
        print(f"  [incomplete] {list(positions)} interrupted, not caching", flush=True)
        return dict(positions=list(positions), peak_M=None, peak_thL=None,
                    peak_be_frac=None, s_fold=None, peak_traced=False,
                    overshoot_skip=False, n_steps=0, elapsed_sec=0.0)

    r = q.get()
    snap_out = r.pop('snapshot', None)
    log[k] = r
    save_pkl(LOG_FILE, log)
    if snap_out is not None:
        warm[k] = snap_out
        save_pkl(WARM_FILE, warm)
    print(f"  [done] {list(positions)} -> peak_M={r['peak_M']} s_fold={r.get('s_fold')}"
          f"{' (OVER, stopped mid-climb)' if r.get('overshoot_skip') else ''}", flush=True)
    return r


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def find_fold_segment(positions, s_fold, L=L_TOTAL):
    """0-based segment index j: fold lies in [full[j], full[j+1]],
    full = [0] + positions + [L]. j==0 is L1; j==len(full)-2 is the last."""
    full = [0.0] + list(positions) + [L]
    for i in range(len(full) - 1):
        if full[i] - 1e-6 <= s_fold <= full[i + 1] + 1e-6:
            return i, full
    return (0 if s_fold < full[0] else len(full) - 2), full


def geometry_ok(positions, L=L_TOTAL):
    full = [0.0] + list(positions) + [L]
    return all(full[i + 1] - full[i] >= MIN_GAP for i in range(len(full) - 1))


def in_window(M):
    return M is not None and WIN_LO <= M <= WIN_HI


def status(r):
    M = r.get('peak_M')
    if M is None:
        return "FAILED"
    if r.get('overshoot_skip') or M > WIN_HI:
        return f"OVER (>{WIN_HI:.0f})"
    if M < WIN_LO:
        return f"UNDER by {WIN_LO - M:.1f}"
    return "IN WINDOW"


# ----------------------------------------------------------------------
# STAGE 1: reference L1 (both-ends-clamped short tape)
# ----------------------------------------------------------------------
def find_L1(log):
    """Bisect the two-end-clamp tape length L so its peak M_root lands in
    [WIN_LO, WIN_HI]. Peak M_root decreases as L grows (longer segment is
    weaker), so plain bisection works. Only used if RUN_L1_SEARCH."""
    lo, hi = L1_SEARCH_BRACKET      # lo: too strong (peak>WIN_HI), hi: too weak
    for it in range(12):
        if _stop_requested.is_set():
            break
        mid = 0.5 * (lo + hi)
        r = evaluate_cached([], log, L=mid)
        M = r.get('peak_M')
        print(f"[L1-search] L={mid:.1f} -> peak_M={M} ({status(r)})", flush=True)
        if M is None:
            break
        if in_window(M):
            print(f"[L1-search] converged: L1={mid:.1f}", flush=True)
            return mid
        if r.get('overshoot_skip') or M > WIN_HI:
            lo = mid                 # too strong -> lengthen
        else:
            hi = mid                 # too weak -> shorten
    print(f"[L1-search] did not converge, falling back to L1_REF={L1_REF}", flush=True)
    return L1_REF


# ----------------------------------------------------------------------
# STAGE 2: constructive loop
# ----------------------------------------------------------------------
def main():
    threading.Thread(target=_quit_listener, daemon=True).start()
    log = load_pkl(LOG_FILE, {})

    print(f"target window: peak M_root in [{WIN_LO:.0f}, {WIN_HI:.0f}] N*mm, "
          f"L={L_TOTAL:.0f}mm, N_TAPE={N_TAPE}\n", flush=True)

    L1 = find_L1(log) if RUN_L1_SEARCH else L1_REF
    print(f"STAGE 1: L1 reference = {L1:.1f} mm\n", flush=True)

    # initial layout: plate at L1, plate at the midpoint of [L1, L]
    positions = [L1, 0.5 * (L1 + L_TOTAL)]
    shift_step = SHIFT_STEP0
    shrink_step = SHRINK_STEP0
    prev_positions = None            # for overshoot undo
    last_move = None                 # 'shift' | ('shrink', j) | 'add'
    best_under = None                # (positions, result): highest peak_M <= WIN_HI

    print(f"STAGE 2: starting layout {positions}\n", flush=True)

    for it in range(1, MAX_ITERS + 1):
        if _stop_requested.is_set():
            print("[stopped by user]", flush=True)
            break

        print(f"--- iter {it}: plates={['%.0f' % p for p in positions]} "
              f"(k={len(positions)}) ---", flush=True)
        r = evaluate_cached(positions, log)
        M = r.get('peak_M')

        if M is None:
            print("  evaluation failed / interrupted -- stopping", flush=True)
            break

        print(f"  peak_M={M:.1f} ({status(r)}) s_fold={r.get('s_fold')}", flush=True)

        if in_window(M) and not r.get('overshoot_skip'):
            print(f"\n>>> DONE: plates={positions} peak_M={M:.1f} "
                  f"in [{WIN_LO:.0f},{WIN_HI:.0f}] with k={len(positions)} plates",
                  flush=True)
            return positions, r

        if r.get('overshoot_skip') or M > WIN_HI:
            # ---- overshoot: undo the last move and halve its step ----
            if prev_positions is None:
                print("  OVER already at the initial layout -- L1 reference is too "
                      "strong; lengthen L1_REF or enable RUN_L1_SEARCH", flush=True)
                break
            if last_move == 'shift':
                shift_step *= 0.5
                step_now = shift_step
            else:
                shrink_step *= 0.5
                step_now = shrink_step
            print(f"  OVER -> undoing last move ({last_move}), halving its step "
                  f"to {step_now:.1f}mm", flush=True)
            positions = prev_positions
            prev_positions = None
            if step_now < STEP_MIN:
                print("  step below floor -- reporting best under-window layout", flush=True)
                break
            continue

        # track the best under-window layout seen (closest to the window)
        if best_under is None or M > best_under[1]['peak_M']:
            best_under = (list(positions), r)

        s_fold = r.get('s_fold')
        if s_fold is None:
            print("  no fold location available (unexpected on a full trace) -- stopping",
                  flush=True)
            break

        j, full = find_fold_segment(positions, s_fold)
        m = len(full) - 1
        print(f"  fold at {s_fold:.1f}mm -> segment L{j+1} of {m} "
              f"[{full[j]:.0f},{full[j+1]:.0f}]", flush=True)

        # ---- decide the move --------------------------------------------
        if j == 0:
            # fold in L1: shift ALL plates left -> L1 shortens & strengthens
            cand = [p - shift_step for p in positions]
            if not geometry_ok(cand) or cand[0] < MIN_GAP:
                print("  L1-shift would break geometry -- stopping", flush=True)
                break
            print(f"  move: fold in L1 -> shift all plates left by {shift_step:.1f}mm "
                  f"-> {['%.0f' % p for p in cand]}", flush=True)
            prev_positions, positions, last_move = list(positions), cand, 'shift'

        elif j == m - 1:
            # fold in the last segment: add a plate at its midpoint
            if len(positions) >= MAX_PLATES:
                print("  MAX_PLATES reached -- stopping", flush=True)
                break
            insert_at = 0.5 * (full[j] + full[j + 1])
            cand = sorted(positions + [insert_at])
            if not geometry_ok(cand):
                print("  midpoint add would break geometry -- stopping", flush=True)
                break
            print(f"  move: fold in last segment -> add plate at {insert_at:.0f}mm "
                  f"(k={len(positions)}->{len(cand)})", flush=True)
            prev_positions, positions, last_move = list(positions), cand, 'add'
            shrink_step = SHRINK_STEP0     # fresh segment -> reset the shrink step

        else:
            # fold in an interior segment Lj: shrink it by moving its RIGHT
            # boundary clamp (positions[j], since full[j+1] == positions[j])
            # left; the segment to its right grows and will eventually take
            # over the fold -> triggering the add rule above.
            cand = list(positions)
            cand[j] -= shrink_step
            if not geometry_ok(cand):
                print(f"  shrinking L{j+1} would break geometry (gap < {MIN_GAP:.0f}mm) "
                      f"-- stopping", flush=True)
                break
            print(f"  move: fold in L{j+1} -> move clamp at {positions[j]:.0f} left by "
                  f"{shrink_step:.1f}mm -> {cand[j]:.0f}", flush=True)
            prev_positions, positions, last_move = list(positions), cand, ('shrink', j)

    else:
        print("[MAX_ITERS reached]", flush=True)

    print("\n================ summary ================", flush=True)
    if best_under is not None:
        bp, br = best_under
        print(f"best under-window layout: plates={bp} peak_M={br['peak_M']:.1f} "
              f"(window [{WIN_LO:.0f},{WIN_HI:.0f}])", flush=True)
    else:
        print("no completed under-window evaluation recorded", flush=True)
    return (best_under if best_under else (None, None))


if __name__ == '__main__':
    main()
