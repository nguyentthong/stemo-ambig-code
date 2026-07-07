<!--
PAIRED OUTPUT FILE: paper/figures/make_fig_responses.py
WORKFLOW: copy everything below the marker into Gemini. Save the single
python code block it returns as the paired output file (replace the whole
file). Then tell Claude to render.
-->

----------------- COPY BELOW THIS LINE -----------------

You are writing a NEW matplotlib script, make_fig_responses.py, for an ACL
paper, in the same house style as the reference script below (copy its
rcParams and conventions). The figure: a DIVERGING stacked bar chart of each
model's five-way first-response distribution, aligned at the boundary
between proactive behavior (enumerate, scope-anchored clarification, drawn
leftward from zero) and passive or erroneous behavior (vague clarification,
silent commitment, refusal, drawn rightward), so the bar alignment itself
ranks models by proactivity. Category colors: enumerate #1baf7a,
scope-anchored #2a78d6, vague #9bbce0, commit #c5544a, refuse #d9d9d9.
One-row legend above the axes. Segments wider than 8 points get a white
in-segment percentage label.

Keep the house style exactly: the rcParams block of the reference script
(Times-like serif, 7 pt, pdf.fonttype 42), 3.03 in figure width, spines
top/right removed. Output the COMPLETE runnable script in one fenced code
block, nothing else. matplotlib/numpy only. The script saves PDF and PNG
(dpi=300) next to itself.

DATA lists the complete roster: a null value means that measurement does not
exist yet, and the model renders as a gray full-width pending row; a
non-null value is a five-element percentage list [enumerate, scope, vague,
commit, refuse] summing to 100 within rounding, plotted exactly as given. This figure is
model-only: humans have no protocol run, so no human row appears.

DATA:
```json
{
  "response_distribution": {
    "GPT-5":            [25.6, 36.9, 3.6, 25.5, 8.3],
    "GPT-4o":           null,
    "Gemini-3-Flash":   null,
    "Gemini-3.5-Flash": null,
    "Qwen3.5-27B":      null,
    "Qwen3.6-27B":      null,
    "Qwen3-VL-32B":     null,
    "InternVL3.5-8B":   null,
    "InternVL3.5-38B":  null
  }
}
```

STYLE REFERENCE SCRIPT:
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
    ("GPT-5", 75.3, 40.8, (-2.0, 3.0)),
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
