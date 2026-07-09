"""
Plot tape-spring BVP results in the "force vs moment" 4-panel layout.

Usage
-----
    python3 plot_results.py case_force_free.pkl case_moment.pkl
    python3 plot_results.py case_force_free.pkl          # force only, still works

The script sorts the pkl files you pass by their meta['loading'] tag:
  * loading == 'tip_force'  -> drawn in RED  (root moment P*L is the peaking qty)
  * loading == 'moment'     -> drawn in BLUE (uniform moment M is the peaking qty)
You normally pass ONE of each. Order does not matter.

To reproduce the reference figure you therefore need BOTH pkl files present:
    - a tip-force run   (run_case_force.py  with END_LEFT=(0.95,'tight'),
                         END_RIGHT='free')      -> e.g. case_force_free.pkl
    - a pure-moment run (run_case_moment.py with both ends clamped)
                                                -> case_moment.pkl
plot_results.py only DRAWS pkl files; it never creates them. If a panel looks
empty or "wrong", it almost always means the pkl you passed is a different
physical case than you expected -- check the title, which prints the actual
boundary conditions read from meta.

Panels
------
  top-left  : load-rotation response, both curves, limit point starred
  top-right : fold location vs theta_L, both curves
  bottom-L  : be(s1) profiles of the tip-force run (root clamp at left)
  bottom-R  : moment distribution M(s1)=P(L-s1) at the force limit point,
              vs the uniform-moment capacity, with the fold location marked
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

RED, BLUE = '#c9342b', '#534AB7'
NAT_DEG = np.degrees(0.95)   # natural angle reference (54.4 deg)


def load(fn):
    d = pickle.load(open(fn, 'rb'))
    return fn, d['meta'], d['results']


def is_force(meta):
    return meta.get('loading') == 'tip_force'


files = sys.argv[1:] if len(sys.argv) > 1 else ['case_multiclamp.pkl']

force = None   # (fn, meta, res)
moment = None
for fn in files:
    try:
        item = load(fn)
    except FileNotFoundError:
        print(f"WARNING: {fn} not found, skipping")
        continue
    if not item[2]:
        print(f"WARNING: {fn} has no converged points, skipping")
        continue
    if is_force(item[1]):
        force = item
    else:
        moment = item

if force is None and moment is None:
    sys.exit("No usable pkl files. Pass a tip-force and/or a pure-moment pkl.")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axM, axF, axP, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]


def peak_idx(res):
    return int(np.argmax([r['M'] for r in res]))


# ---------- top-left: load-rotation response ----------
if force is not None:
    _, meta, res = force
    th = [r['thL'] for r in res]; Ms = [r['M'] for r in res]
    ipk = peak_idx(res)
    axM.plot(th, Ms, 'o-', color=RED, lw=2, ms=5, label='tip force: root moment P*L')
    axM.plot(th[ipk], Ms[ipk], '*', color=RED, ms=18)
if moment is not None:
    _, meta_m, res_m = moment
    th = [r['thL'] for r in res_m]; Ms = [r['M'] for r in res_m]
    ipk = peak_idx(res_m)
    axM.plot(th, Ms, 's-', color=BLUE, lw=2, ms=5, label='pure moment: M (uniform)')
    axM.plot(th[ipk], Ms[ipk], '*', color=BLUE, ms=18)

src = force if force is not None else moment
_, meta_t, _ = src
el = meta_t.get('END_LEFT', ('?', '?'))
el_txt = f"be={el[0]:.2f} {el[1]}" if isinstance(el, tuple) else str(el)
axM.set_title(f"Load-rotation response (L={meta_t['L']:.0f}, root clamp {el_txt})")
axM.set_xlabel('end / tip rotation theta_L (rad)')
axM.set_ylabel('moment (N*mm)')
axM.grid(alpha=0.3)
axM.legend(fontsize=10)

# ---------- top-right: fold location vs loading ----------
fold_min = fold_max = None
if force is not None:
    _, _, res = force
    th = [r['thL'] for r in res]; loc = [r['s_fold'] for r in res]
    axF.plot(th, loc, 'o-', color=RED, lw=2, ms=5, label='tip force')
    fold_min, fold_max = min(loc), max(loc)
if moment is not None:
    _, _, res_m = moment
    th = [r['thL'] for r in res_m]; loc = [r['s_fold'] for r in res_m]
    axF.plot(th, loc, 's-', color=BLUE, lw=2, ms=5, label='pure moment')
    axF.axhline(meta_t['L'] / 2, color='gray', ls=':', lw=1, alpha=0.7)
if fold_min is not None:
    axF.set_title(f"Fold location: force case sits ~{fold_min:.0f}-{fold_max:.0f} mm "
                  f"from root, NOT at s1=0")
else:
    axF.set_title("Fold (argmin beta_e) location vs loading")
axF.set_xlabel('theta_L (rad)')
axF.set_ylabel('fold (argmin be) location s1 (mm)')
axF.grid(alpha=0.3)
axF.legend(fontsize=10)

# ---------- bottom-left: be(s1) profiles of the tip-force run ----------
if force is not None:
    _, _, res = force
    step = max(1, len(res) // 8)
    for r in res[::step]:
        axP.plot(r['s'], np.degrees(r['be']), lw=1.4, alpha=0.85)
    axP.axhline(NAT_DEG, color='gray', ls='--', lw=1)
    axP.set_title('tip force: be(s1) profiles (root clamp at left)')
    axP.set_xlabel('s1 (mm)')
    axP.set_ylabel('beta_e (deg)')
    axP.grid(alpha=0.3)
else:
    axP.text(0.5, 0.5, 'no tip-force pkl passed', ha='center', va='center',
             transform=axP.transAxes, color='gray')

# ---------- bottom-right: moment distribution vs capacity ----------
if force is not None and 'Mdist' in force[2][peak_idx(force[2])]:
    _, _, res = force
    ipk = peak_idx(res)
    rpk = res[ipk]
    s = rpk['s']; Mdist = rpk['Mdist']; s_fold = rpk['s_fold']
    M_at_fold = float(np.interp(s_fold, s, Mdist))
    axD.plot(s, Mdist, color=RED, lw=2.4, label='M(s1) at limit point (incl. self-weight)')
    if moment is not None:
        cap = max(r['M'] for r in moment[2])
        axD.axhline(cap, color=BLUE, ls='--', lw=2.2,
                    label=f'uniform-M capacity ({cap:.0f} N*mm)')
    axD.axvline(s_fold, color='k', ls=':', lw=1.3, label=f'fold @ {s_fold:.1f} mm')
    axD.plot([s_fold], [M_at_fold], 'ko', ms=9)
    axD.annotate(f"local M at fold = {M_at_fold:.0f} N*mm",
                 xy=(s_fold, M_at_fold), xytext=(s_fold + 10, M_at_fold + 60),
                 fontsize=11, va='center')
    axD.set_title('Moment distribution vs uniform-moment capacity')
    axD.set_xlabel('s1 (mm)')
    axD.set_ylabel('moment (N*mm)')
    axD.grid(alpha=0.3)
    axD.legend(fontsize=9)
else:
    axD.text(0.5, 0.5, 'moment-distribution panel needs a tip-force pkl',
             ha='center', va='center', transform=axD.transAxes, color='gray')

plt.tight_layout()
plt.savefig('force_vs_moment_multiclamp.png', dpi=130)
print("saved force_vs_moment_multiclamp.png")

# ---------- console summary ----------
for tag, item in (('tip force', force), ('pure moment', moment)):
    if item is None:
        continue
    fn, meta, res = item
    ipk = peak_idx(res)
    r = res[ipk]
    extra = f", P={r['P']:.4f} N" if 'P' in r else ""
    print(f"{tag:11s} {fn}: peak M={r['M']:.1f} N*mm at thL={r['thL']:.4f}, "
          f"fold s1={r['s_fold']:.1f} mm ({np.degrees(r['be_fold']):.1f} deg){extra}")