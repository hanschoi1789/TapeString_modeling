"""
Plot one or more result files produced by run_case.py.
Usage:  python3 plot_results.py case_custom.pkl [more.pkl ...]
"""
import sys, pickle
import numpy as np
import matplotlib.pyplot as plt

files = sys.argv[1:] if len(sys.argv) > 1 else ['case_custom.pkl']
colors = ['#1D9E75', '#534AB7', '#c9342b', '#e08214']

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axM, axF, axP1, axP2 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

for i, fn in enumerate(files):
    d = pickle.load(open(fn, 'rb'))
    meta, res = d['meta'], d['results']
    c = colors[i % len(colors)]
    label = f"{fn}  (L={meta['L']:.0f}, L-end={meta['END_LEFT']}, R-end={meta['END_RIGHT']})"

    th = [r['thL'] for r in res]; Ms = [r['M'] for r in res]
    axM.plot(th, Ms, 'o-', color=c, lw=2, ms=4, label=label[:50])
    ipk = int(np.argmax(Ms))
    axM.plot(th[ipk], Ms[ipk], '*', color=c, ms=16)

    loc = [r['s_fold'] for r in res]
    axF.plot(th, loc, 'o-', color=c, lw=2, ms=4)
    axF.axhline(meta['L']/2, color=c, ls=':', lw=1, alpha=0.5)

    ax = axP1 if i == 0 else axP2
    for r in res[::max(1, len(res)//8)]:
        ax.plot(r['s'], np.degrees(r['be']), lw=1.3, alpha=0.8)
    ax.axhline(np.degrees(0.95), color='gray', ls='--', lw=1)
    ax.set_title(fn); ax.set_xlabel('s1 (mm)'); ax.set_ylabel('beta_e (deg)')
    ax.grid(alpha=0.3)

axM.set_xlabel('end rotation theta_L (rad)'); axM.set_ylabel('M (N*mm)')
axM.set_title('Rotation-controlled M-theta response'); axM.grid(alpha=0.3); axM.legend(fontsize=7)
axF.set_xlabel('theta_L (rad)'); axF.set_ylabel('fold location s1 (mm)')
axF.set_title('Fold (argmin beta_e) location vs loading'); axF.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=130)
print("saved comparison_plot.png")

for fn in files:
    d = pickle.load(open(fn,'rb')); res = d['results']
    Ms = [r['M'] for r in res]; ipk = int(np.argmax(Ms))
    print(f"{fn}: peak M = {Ms[ipk]:.1f} N*mm at thL={res[ipk]['thL']:.4f}, fold at s1={res[ipk]['s_fold']:.1f} mm")
