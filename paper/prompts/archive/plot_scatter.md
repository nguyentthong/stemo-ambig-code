# Prompt: detection-versus-action scatter (fig_scatter)

You are finishing a matplotlib figure for an ACL paper on ambiguity in video
question answering. The figure: one point per model, x = fraction of 100
ambiguous items the model judges ambiguous when asked directly, y = its
clarification rate under the interactive protocol. Dashed diagonal y = x
labeled "perceive-act parity". Humans appear as a vertical dotted reference
line at their detection rate with a short horizontal label (humans have no
protocol run, so they have no y value). A model appears as a point when DATA
holds both quantities for it; other models are omitted from the axes and the
caption carries the scope, so nothing pending is drawn on the plot.
Keep the house style exactly: the rcParams block already in the script
(Times-like serif, 7 pt, pdf.fonttype 42), 3.03 in figure width, palette
entity-blue #2a78d6, event-green #1baf7a, human-green #1baf7a, red accent
#c5544a, spines top/right removed, value labels with one decimal. Output the
COMPLETE runnable script in one fenced code block, nothing else.
matplotlib/numpy only. The script saves PDF and PNG (dpi=300) next to itself.
DATA below is the complete, authoritative record of all measured results:
every number in the figure comes from DATA and only from DATA. A model with
no entry in DATA is not yet measured: render it in the pending style
described below, so the full roster stays visible without a measured value.
Full roster and order: GPT-5, GPT-4o, Gemini-3-Flash, Gemini-3.5-Flash,
Qwen3.5-27B, Qwen3.6-27B, Qwen3-VL-32B, InternVL3.5-8B, InternVL3.5-38B
(plus Humans only where a human quantity exists).

DATA lists the complete roster: a null value means that measurement does not exist yet, and the model renders in the pending style; a non-null value is plotted exactly as given.

DATA:
```json
{
  "detection": {
    "Humans (4 annotators)": {"hit": 84.5, "fa": 3.2},
    "GPT-5":                 {"hit": 75.3, "fa": 12.1},
    "GPT-4o":                {"hit": 4.0,  "fa": 0.0},
    "Gemini-3-Flash":        {"hit": 54.9, "fa": 15.2},
    "Gemini-3.5-Flash":      {"hit": 52.4, "fa": 12.1},
    "Qwen3.5-27B":           null,
    "Qwen3.6-27B":           null,
    "Qwen3-VL-32B":          {"hit": 28.5, "fa": 5.0},
    "InternVL3.5-8B":        null,
    "InternVL3.5-38B":       null
  },
  "clarification_rate": {
    "GPT-5": 40.8,
    "GPT-4o": 1.7,
    "Gemini-3-Flash": 15.7,
    "Gemini-3.5-Flash": 8.8,
    "Qwen3.5-27B": 4.6,
    "Qwen3.6-27B": 4.1,
    "Qwen3-VL-32B": 17.0,
    "InternVL3.5-8B": 3.2,
    "InternVL3.5-38B": 12.5
  }
}
```

CURRENT SCRIPT:
```python
"""Signature figure: ambiguity detection versus action (section 4.6).

Scatter, one point per model: x = binary detection hit rate on the 100
ambiguous study items, y = clarification rate under the protocol (Table 3).
The diagonal y = x marks perceive-act parity: a model on the line asks about
every ambiguity it reports perceiving. Humans have no protocol run by
design, so they appear as a vertical reference line at their detection rate.
False-alarm rates live in the appendix table.

Adopted by unanimous two-model vote (figformat_log.md) replacing the
RAcQUEt-style detection bar chart.

DATA: (label, detection %, clarification %) — None = pending.
Detection: analysis/binary_judgment_<model>.json + training-box runs.
Clarification: Table 3 (paper/main.tex).
Output: paper/figures/fig_scatter.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent

C_MODEL = "#2a78d6"
C_GPT5 = "#c5544a"
C_HUMAN = "#1baf7a"
INK = "#333333"

HUMAN_DETECT = 84.5

# (label, detection %, clarification %, label offset (dx, dy))
MODELS = [
    ("GPT-5", 75.3, None, (-2.0, 3.0)),
                                               # scored protocol run (in flight)
    ("Qwen3-VL-32B", 28.5, 17.0, (2.0, 2.0)),
    ("GPT-4o", 4.0, 1.7, (3.0, 4.0)),
    ("Gemini-3.5-Flash", None, 8.8, None),
    ("Gemini-3-Flash", None, 15.7, None),
    ("Qwen3.5-27B", None, 4.6, None),
    ("Qwen3.6-27B", None, 4.1, None),
    ("InternVL3.5-38B", None, 12.5, None),
    ("InternVL3.5-8B", None, 3.2, None),
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

fig, ax = plt.subplots(figsize=(3.03, 2.55))
lim = 100
ax.plot([0, lim], [0, lim], ls="--", lw=0.7, color="#aaaaaa", zorder=1)
ax.text(52, 57, "perceive\u2013act parity", rotation=45, fontsize=6.0,
        color="#555555", ha="center", va="center")
ax.axvline(HUMAN_DETECT, ls=":", lw=0.9, color=C_HUMAN, zorder=1)
ax.text(HUMAN_DETECT - 1.2, 96, f"human = {HUMAN_DETECT}",
        fontsize=6.0, color=C_HUMAN, ha="right", va="top")
ax.text(62, 11, "perceives without acting", fontsize=6.2, color="#555555",
        style="italic", ha="center")

plotted = 0
for label, det, clar, off in MODELS:
    if det is None or clar is None:
        continue
    color = C_GPT5 if label == "GPT-5" else C_MODEL
    ax.scatter([det], [clar], s=34, color=color, zorder=3,
               edgecolors="white", linewidths=0.6)
    ax.annotate(label, (det, clar), xytext=(det + off[0], clar + off[1]),
                fontsize=6.0, color=INK,
                ha="right" if off[0] < 0 else "left")
    plotted += 1
pending = sum(1 for _, d, c, _ in MODELS if d is None or c is None)
# pending count intentionally NOT drawn on the plot (vision-judge fix):
# the caption carries study scope; sparsity resolves when data lands

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("ambiguity detected (\\%)" if False else "ambiguity detected (%)")
ax.set_ylabel("clarification rate (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#d8d8d8", lw=0.5, alpha=0.9)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_scatter.pdf")
fig.savefig(OUT / "fig_scatter.png", dpi=300)
print(f"wrote fig_scatter.pdf ({plotted} points, {pending} pending)")
```
