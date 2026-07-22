import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =============================================================================
# Plotting
# =============================================================================



colors = {"6D":"red", "Quaternion":"green", "Axis-angle":"cyan", "Euler":"blue", "SVD":"magenta"}
styles = {"6D":"-",   "Quaternion":"-",     "Axis-angle":"-",    "Euler":"-",   "SVD":"--"}
order  = ["6D", "Quaternion", "Axis-angle", "Euler", "SVD"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.text(0.01, 0.97, "Sanity Test", fontsize=13, va='top', ha='left')

ax = axes[0]
for name in order:
    iters = [x[0] for x in results[name]["mean_errors"]]
    errs  = [x[1] for x in results[name]["mean_errors"]]
    ax.plot(iters, errs, color=colors[name], linestyle=styles[name],
            linewidth=1.5, label=name)
ax.set_xlim(0, TOTAL); ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)//1000}k"))
ax.set_xlabel("a. Mean errors during iterations.", fontsize=9)
ax.legend(fontsize=8); ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
pcts = np.linspace(0, 100, 1000)
for name in order:
    vals = np.percentile(results[name]["final_errors"], pcts)
    ax.semilogy(pcts, vals, color=colors[name], linestyle=styles[name],
                linewidth=1.5, label=name)
ax.set_xlim(0, 100); ax.set_ylim(0.1, 200)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}°"))
ax.set_xlabel("b. Percentile of errors at 500k iteration.", fontsize=9)
ax.legend(fontsize=8, loc='upper left'); ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, which='both')

ax = axes[2]; ax.axis('off')
col_labels = ["", "Mean(°)", "Max(°)", "Std(°)"]
table_data = []
for name in order:
    e = results[name]["final_errors"]
    table_data.append([name, f"{e.mean():.2f}", f"{e.max():.2f}", f"{e.std():.2f}"])
t = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.1, 1.6)
for (row, col), cell in t.get_celld().items():
    if row == 0: cell.set_text_props(fontweight='bold')
ax.set_xlabel("c. Errors at 500k iteration.", fontsize=9)

plt.tight_layout()
plt.savefig("./geodesic.png", dpi=150, bbox_inches='tight')
print("\nPlot saved to ./geodesic.png")