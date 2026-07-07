You are refining a matplotlib figure for an ACL paper on ambiguity in video
question answering. It is the paper's signature figure and must support these
claims from the paper exactly as they are written:

- Every evaluated model sits below the perceive-act parity line: models ask
  about far less ambiguity than they report perceiving.
- Detection spans 4.0% (GPT-4o) to 75.3% (GPT-5), against 84.5% for humans.
- GPT-5 is the model nearest parity, yet it clarifies on 40.8% of questions
  while detecting 75.3%, so it resolves less than half of what it perceives.

Design, as currently implemented: dashed diagonal y = x labeled "parity" in
italic, a very subtle blue-gray tint under the diagonal labeled "perceives
without acting" in italic, GPT-5 as a larger warm-red point with a dark
edge, other models as softer blue points with thin edges and white-halo
labels, a dotted green vertical line at the human detection rate, hollow
green human marker on the diagonal, no grid, top and right spines removed,
equal aspect, x to 100 and y to 90.

Keep the house style: the rcParams block already in the script (Times-like
serif, 7 pt, pdf.fonttype 42), 3.03 in figure width. Improve layout,
spacing, and legibility only. Output the COMPLETE runnable script in one
fenced code block, nothing else. matplotlib/numpy only. The script saves
PDF and PNG (dpi=300) next to itself.

CURRENT SCRIPT:
```python
"""Signature figure: ambiguity detection versus action (section 4.6).

Scatter, one point per model: x = binary detection hit rate on the 100
ambiguous study items, y = clarification rate under the protocol (Table 1).
The diagonal y = x marks perceive-act parity. Humans have a measured
detection rate but no protocol run by design, so they appear as a hollow
marker ON the parity line (parity assumed), with a dotted vertical

Output: fig_scatter.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent

INK = "#333333"
HUMAN_DETECT = 84.5

MODELS = [
    ("GPT-5", 75.3, 40.8, (-2.5, 3.5)),
    ("GPT-4o", 4.0, 1.7, (3.5, 3.5)),
    ("Gemini-3-Flash", 54.9, 15.7, (2.5, 3.0)),
    ("Gemini-3.5-Flash", 52.4, 8.8, (1.5, -6.0)),
    ("Qwen3-VL-32B", 28.5, 17.0, (-2.5, 3.5)),
    ("Humans", None, None, (None, None)),
    ("Qwen3.5-27B", None, None, (None, None)),
    ("Qwen3.6-27B", None, None, (None, None)),
    ("InternVL3.5-8B", None, None, (None, None)),
    ("InternVL3.5-38B", None, None, (None, None)),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import matplotlib.patheffects as pe
fig, ax = plt.subplots(figsize=(3.03, 2.62))
XMAX, YMAX = 100, 90
ax.fill_between([0, XMAX], [0, 0], [0, XMAX], color="#8fa8c4", alpha=0.07, zorder=0)
ax.plot([0, YMAX], [0, YMAX], ls=(0, (5, 3)), lw=1.0, color="#555555", zorder=1)
ax.text(26.0, 31.0, "parity", rotation=45, fontsize=7.5, style="italic",
        color="#444444", ha="center", va="center",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
ax.axvline(HUMAN_DETECT, ls=":", lw=1.0, color="#128a5f", zorder=1)
# humans: measured detection, parity assumed for the y coordinate
ax.scatter([HUMAN_DETECT], [HUMAN_DETECT], s=52, facecolors="white",
           edgecolors="#128a5f", linewidths=1.2, zorder=4)
ax.text(HUMAN_DETECT - 3.0, HUMAN_DETECT + 1.0, "humans (84.5%,\nparity assumed)",
        fontsize=6.3, color="#128a5f", ha="right", va="bottom", linespacing=1.15)
ax.text(58, 30, "perceives without acting", fontsize=7.0, color="#666666",
        style="italic", ha="center")

halo = [pe.withStroke(linewidth=1.8, foreground="white")]
for label, det, clar, off in MODELS:
    is5 = label == "GPT-5"
    ax.scatter([det], [clar], s=52 if is5 else 38,
               color="#e8635a" if is5 else "#3f7fc4",
               edgecolors="#8c2f28" if is5 else "white",
               linewidths=0.8 if is5 else 0.7, zorder=3)
    ax.annotate(label, (det, clar), xytext=(det + off[0], clar + off[1]),
                fontsize=6.5, color=INK, path_effects=halo,
                ha="right" if off[0] < 0 else "left")

ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.set_aspect("equal")
ax.set_xticks(range(0, 101, 25))
ax.set_yticks(range(0, 91, 30))
ax.set_xlabel("ambiguity detected (%)")
ax.set_ylabel("clarification rate (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=2.5, width=0.6)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_scatter.pdf")
fig.savefig(OUT / "fig_scatter.png", dpi=300)
print("wrote", OUT / "fig_scatter.pdf")
```
