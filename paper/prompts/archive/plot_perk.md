You are finishing a matplotlib line chart for an ACL paper: ReQueST score
(%) over K bins (2, 3, 4-6, 7+). Closed-weight models solid lines,
open-weight dashed, one distinct marker per model, compact legend.  Keep
it a line chart: the reader must see the decay shape and where clarification
survives enumeration.
Keep the house style exactly: the rcParams block already in the script
(Times-like serif, 7 pt, pdf.fonttype 42), 3.03 in figure width, palette
entity-blue #2a78d6, event-green #1baf7a, human-green #1baf7a, red accent
#c5544a, spines top/right removed, value labels with one decimal. Output the
COMPLETE runnable script in one fenced code block, nothing else.
matplotlib/numpy only. The script saves PDF and PNG (dpi=300) next to itself.
Full roster and order: GPT-5, GPT-4o, Gemini-3-Flash, Gemini-3.5-Flash,
Qwen3.5-27B, Qwen3.6-27B, Qwen3-VL-32B, InternVL3.5-8B, InternVL3.5-38B
(plus Humans only where a human quantity exists).

CURRENT SCRIPT:
```python
"""Figure: ReQueST score by K bin for the closed-weight models (section 4.2).

Output: paper/figures/fig_perk.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent

INK = "#333333"
COLORS = {"GPT-4o": "#8a8a8a", "Gemini-3-Flash": "#2a78d6",
          "Gemini-3.5-Flash": "#1baf7a"}
MARKERS = {"GPT-4o": "s", "Gemini-3-Flash": "o", "Gemini-3.5-Flash": "^"}

# score (%) by K bin, from the IAA-protocol runs
BINS = ["2", "3", "4–6", "7+"]
SCORE = {
    "GPT-5": [53.1, 34.0, 33.5, 31.7],
    "GPT-4o": [1.1, 0.0, 0.0, 0.0],
    "Gemini-3-Flash": [35.4, 6.9, 11.2, 7.6],
    "Gemini-3.5-Flash": [32.2, 8.6, 12.2, 4.2],
    "Qwen3.5-27B": None,
    "Qwen3.6-27B": None,
    "Qwen3-VL-32B": None,
    "InternVL3.5-8B": None,
    "InternVL3.5-38B": None,
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.03, 1.75))
x = range(len(BINS))
for name, ys in SCORE.items():
    ax.plot(x, ys, marker=MARKERS[name], ms=3.2, lw=1.1,
            color=COLORS[name], label=name)
ax.set_xticks(list(x))
ax.set_xticklabels([f"$K$={b}" for b in BINS])
ax.set_ylabel("\\bench{} score (%)" if False else "ReQueST score (%)")
ax.set_ylim(0, 38)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", handlelength=1.6)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_perk.pdf")
fig.savefig(OUT / "fig_perk.png", dpi=300)
print("wrote", OUT / "fig_perk.pdf")
```
