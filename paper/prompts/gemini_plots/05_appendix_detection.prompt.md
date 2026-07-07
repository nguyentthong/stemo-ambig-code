You are finishing a matplotlib figure for the appendix of an ACL paper:
binary ambiguity detection per rater, horizontal bars, humans anchored on
top, models below sorted by hit rate descending. A red vertical tick on
each measured bar marks the false-alarm rate on the 40 control items.
Value labels at bar ends with one decimal.

Keep the house style exactly: the rcParams block already in the script
(Times-like serif, 7 pt, pdf.fonttype 42), 3.03 in figure width,
human-green #1baf7a, model-blue #2a78d6, red accent #c5544a, spines
top/right removed. Output the COMPLETE runnable script in one fenced code
block, nothing else. matplotlib/numpy only. The script saves PDF and PNG
(dpi=300) next to itself.


CURRENT SCRIPT:
```python
"""Figure: binary ambiguity detection, humans vs models (section 4.6).

RAcQUEt Figure-3 style: horizontal bars, one row per rater, human anchor on
top. Hit rate on the 100 ambiguous items (solid) with the false-alarm rate
on the 40 controls (thin overlay marker).

Output: paper/figures/fig_detection.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent

C_HUMAN = "#1baf7a"
C_MODEL = "#2a78d6"
INK = "#333333"

ROWS = [
    ("Humans (4 annotators)", 84.5, 3.2),
    ("GPT-5", 75.3, 12.1), 
    ("Qwen3-VL-32B", 28.5, 5.0),
    ("GPT-4o", 4.0, 0.0),
    ("Gemini-3.5-Flash", None, None),
    ("Gemini-3-Flash", None, None),
    ("Qwen3.5-27B", None, None),
    ("Qwen3.6-27B", None, None),
    ("InternVL3.5-38B", None, None),
    ("InternVL3.5-8B", None, None),
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

rows = [r for r in ROWS]
fig, ax = plt.subplots(figsize=(3.03, 0.28 * len(rows) + 0.55))
y = np.arange(len(rows))[::-1]

for yi, (label, hit, fa) in zip(y, rows):
    if hit is None:
        ax.barh(yi, 100, height=0.62, color="#d9d9d9", alpha=0.35)
        continue
    color = C_HUMAN if label.startswith("Humans") else C_MODEL
    ax.barh(yi, hit, height=0.62, color=color)
    ax.text(hit + 1.2, yi, f"{hit:.1f}", va="center", fontsize=6.3, color=INK)
    ax.plot([fa], [yi], marker="|", ms=9, mew=1.4, color="#c5544a")

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 100)
ax.set_xlabel("ambiguity detected (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_detection.pdf")
fig.savefig(OUT / "fig_detection.png", dpi=300)
print("wrote", OUT / "fig_detection.pdf")
```
