"""Per-finding AUROC change plot, drawn from analysis_v3's per_label_gain.csv (no recomputation)."""
import ast, csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
for r in rows:
    r['d'] = float(r['delta_auroc']) * 100
    lo, hi = ast.literal_eval(r['ci'])
    r['lo'], r['hi'] = lo * 100, hi * 100
    r['sig'] = float(r['p_holm']) < 0.05
rows.sort(key=lambda r: r['d'])
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(3.4, 3.3))
for i, r in enumerate(rows):
    c = '#b2182b' if r['sig'] else '#4d4d4d'
    ax.plot([r['lo'], r['hi']], [i, i], color=c, lw=1)
    ax.plot(r['d'], i, 'o', color=c, ms=3.5, mfc=c if r['sig'] else 'white')
ax.axvline(0, color='0.6', lw=0.8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['label'].replace('_', ' ')} ({float(r['prevalence_test']) * 100:.1f}%)" for r in rows])
ax.set_xlabel(r'$\Delta$AUROC vs. EVA-X-S (points)')
ax.set_xlim(min(r['lo'] for r in rows) - 0.3, max(min(r['hi'], 3.5) for r in rows) + 0.3)
for i, r in enumerate(rows):
    if r['hi'] > 3.5:
        ax.annotate('', xy=(3.6, i), xytext=(3.2, i), arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))
fig.tight_layout()
fig.savefig(out)
print('saved', out)
