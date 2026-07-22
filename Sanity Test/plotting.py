import os, glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# Load all results from ./data/*.pkl
# Each file is named <RepName>.pkl and contains a dict with keys:
#   "mean_errors" : list of (iteration, error_deg) tuples
#   "final_errors": numpy array of per-sample geodesic errors in degrees
# =============================================================================

DATA_DIR = "./data"
pkl_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))

if not pkl_files:
    raise FileNotFoundError(f"No .pkl files found in {DATA_DIR!r}")

results = {}
for path in pkl_files:
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as f:
        results[name] = pickle.load(f)
    print(f"Loaded {name}: "
          f"{len(results[name]['mean_errors'])} log points, "
          f"{len(results[name]['final_errors'])} eval samples")

order = list(results.keys())   # plot in alphabetical / load order
print(f"\nRepresentations found: {order}")

# =============================================================================
# Colour / style palette — well-known reps get fixed colours so they always
# look the same regardless of which subset is present; anything else gets
# assigned from a fallback cycle.
# =============================================================================

KNOWN_COLORS = {
    "6D":         "red",
    "Quaternion": "green",
    "Axis-angle": "cyan",
    "Euler":      "blue",
    "SVD":        "magenta",
}
KNOWN_STYLES = {
    "6D":         "-",
    "Quaternion": "-",
    "Axis-angle": "-",
    "Euler":      "-",
    "SVD":        "--",
}

FALLBACK_COLORS = ["orange", "brown", "purple", "olive", "pink", "gray"]
FALLBACK_STYLES = ["-", "--", "-.", ":"]

fallback_color_idx = 0
fallback_style_idx = 0
colors, styles = {}, {}
for name in order:
    if name in KNOWN_COLORS:
        colors[name] = KNOWN_COLORS[name]
        styles[name] = KNOWN_STYLES[name]
    else:
        colors[name] = FALLBACK_COLORS[fallback_color_idx % len(FALLBACK_COLORS)]
        styles[name] = FALLBACK_STYLES[fallback_style_idx % len(FALLBACK_STYLES)]
        fallback_color_idx += 1
        fallback_style_idx += 1

# Infer TOTAL from the last logged iteration across all reps
TOTAL = max(
    results[name]["mean_errors"][-1][0]
    for name in order
    if results[name]["mean_errors"]
)

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.text(0.01, 0.97, "Sanity Test", fontsize=13, va='top', ha='left')

# (a) Mean error during training
ax = axes[0]
for name in order:
    iters = [x[0] for x in results[name]["mean_errors"]]
    errs  = [x[1] for x in results[name]["mean_errors"]]
    ax.plot(iters, errs, color=colors[name], linestyle=styles[name],
            linewidth=1.5, label=name)
ax.set_xlim(0, TOTAL)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)//1000}k"))
ax.set_xlabel("a. Mean errors during iterations.", fontsize=9)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

# (b) Percentile of final errors
ax = axes[1]
pcts = np.linspace(0, 100, 1000)
for name in order:
    vals = np.percentile(results[name]["final_errors"], pcts)
    ax.semilogy(pcts, vals, color=colors[name], linestyle=styles[name],
                linewidth=1.5, label=name)
ax.set_xlim(0, 100)
ax.set_ylim(0.1, 200)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}°"))
ax.set_xlabel(f"b. Percentile of errors at {TOTAL//1000}k iteration.", fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, which='both')

# (c) Summary table
ax = axes[2]
ax.axis('off')
col_labels = ["", "Mean(°)", "Max(°)", "Std(°)"]
table_data = []
for name in order:
    e = results[name]["final_errors"]
    table_data.append([name, f"{e.mean():.2f}", f"{e.max():.2f}", f"{e.std():.2f}"])
t = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1.1, 1.6)
for (row, col), cell in t.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight='bold')
ax.set_xlabel("c. Errors at final iteration.", fontsize=9)

plt.tight_layout()
out = "./geodesic.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out}")