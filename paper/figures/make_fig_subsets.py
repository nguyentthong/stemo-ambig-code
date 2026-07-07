"""Figure: entity vs event strict-K per model, dumbbell (section 4.3).

One row per evaluated model, entity and event strict-K connected, ordered by
entity strict-K so the ranking reads top-to-bottom. Every model above the floor
scores higher on the entity subset than the event subset: readings recalled
across episodes are harder to find than readings scanned within one scene.

Values reconcile with the overall strict-K in Table 1.
  closed (GPT-5, Gemini x2): per-subset strict-K from the section 4.1 runs.
  open + GPT-4o: analysis/per_subset_metrics.json / open_weight_iaa_metrics.json.
Output: fig_subsets.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent

# Slate tones (outside the model-family palette of Figs 3/5) so blue/green
# here read as SUBSET, not as Qwen/Gemini. Marker shape also distinguishes them.
C_ENTITY = "#334155"
C_EVENT = "#94a3b8"
INK = "#333333"

# (label, entity strict-K %, event strict-K %)
MODELS = [
    ("Gemini-3-Flash",   28.1, 8.7),
    ("Gemini-3.5-Flash", 26.9, 7.9),
    ("GPT-5",            22.3, 6.4),
    ("Qwen3.6-27B",      16.6, 6.5),
    ("Qwen3.5-27B",      12.8, 5.7),
    ("Qwen3-VL-32B",     11.0, 5.1),
    ("InternVL3.5-8B",    4.0, 2.0),
    ("InternVL3.5-38B",   0.5, 1.8),
    ("GPT-4o",            0.0, 0.0),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# highest entity strict-K at the top
rows = sorted(MODELS, key=lambda m: m[1])

C_ELAB = "#334155"   # entity label (dark slate)
C_VLAB = "#64748b"   # event label (mid slate, readable)
OFF = 0.75           # label offset in data units

fig, ax = plt.subplots(figsize=(3.03, 0.32 * len(rows) + 0.58))

# faint alternating row bands to guide the eye across nine rows
for row in range(0, len(rows), 2):
    ax.axhspan(row - 0.5, row + 0.5, color="#f5f6f8", zorder=0)

# mask so a right-placed low label stays readable where it crosses the connector
WHITE = dict(facecolor="white", edgecolor="none", pad=0.4)

for row, (label, e, v) in enumerate(rows):
    y = row
    coincident = abs(e - v) < 0.05
    ye, yv = (y + 0.12, y - 0.12) if coincident else (y, y)
    ax.plot([v, e], [yv, ye], color="#c3c8d0", lw=1.3, zorder=1,
            solid_capstyle="round")
    ax.scatter([e], [ye], s=28, color=C_ENTITY, marker="s", zorder=3)
    ax.scatter([v], [yv], s=30, color=C_EVENT, marker="o", zorder=3,
               edgecolors=C_ELAB, linewidths=0.4)
    # (value, y, colour) for each endpoint
    ent = (e, ye, C_ELAB)
    evt = (v, yv, C_VLAB)
    if coincident:
        # both markers sit at the same x near the axis: stack both labels to the right
        ax.text(e + OFF, ye, f"{e:.1f}", fontsize=6.0, color=C_ELAB, ha="left", va="center")
        ax.text(v + OFF, yv, f"{v:.1f}", fontsize=6.0, color=C_VLAB, ha="left", va="center")
    else:
        hi, lo = (ent, evt) if e > v else (evt, ent)
        # higher endpoint: label on its right (outer, always clear), level with its dot
        ax.text(hi[0] + OFF, hi[1], f"{hi[0]:.1f}", fontsize=6.0, color=hi[2], ha="left", va="center")
        # lower endpoint, always level with its dot and on its left (outer) side;
        # a white mask keeps the near-axis label readable in the margin
        near_axis = lo[0] - OFF < 0.5
        ax.text(lo[0] - OFF, lo[1], f"{lo[0]:.1f}", fontsize=6.0, color=lo[2],
                ha="right", va="center", bbox=WHITE if near_axis else None)

ax.scatter([], [], s=28, color=C_ENTITY, marker="s", label="Entity")
ax.scatter([], [], s=30, color=C_EVENT, marker="o", edgecolors=C_ELAB,
           linewidths=0.4, label="Event")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([m[0] for m in rows])
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("strict-$K$ (%)")
ax.set_xlim(-1.9, 31.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)
ax.set_axisbelow(True)
leg = ax.legend(frameon=False, loc="lower right", handlelength=0.9,
                fontsize=6.0, markerscale=0.9, borderpad=0.2,
                title="Subset", title_fontsize=6.0)
leg._legend_box.align = "left"
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_subsets.pdf")
fig.savefig(OUT / "fig_subsets.png", dpi=300)
print("wrote", OUT / "fig_subsets.pdf")
