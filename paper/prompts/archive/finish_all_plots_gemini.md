# Prompt for Gemini: finish all result figures

Copy everything below the line into Gemini in one message.

---

You are finishing the result figures of an ACL paper about ambiguity in video
question answering. Four matplotlib scripts follow, plus one DATA block. DATA
is the complete, authoritative record of all measured results at the time of
this prompt: every number that appears in any figure comes from DATA and only
from DATA. A model that has no entry for a figure's required fields is not yet
measured for that figure: render it as the pending style already used by the
scripts (gray placeholder row, or omitted point for scatter plots), so the
reader can see the roster without reading measured values for it.

Requirements for every figure:
- Include the full roster in every plot, in this order where applicable:
  Humans (4 annotators), GPT-5, GPT-4o, Gemini-3-Flash, Gemini-3.5-Flash,
  Qwen3.5-27B, Qwen3.6-27B, Qwen3-VL-32B, InternVL3.5-8B, InternVL3.5-38B.
  Humans appear only in figures whose quantities exist for humans
  (detection); model-only figures omit the human row entirely.
- Keep the house style exactly: the rcParams block already in each script
  (Times-like serif, 7 pt, pdf.fonttype 42), 3.03 in figure width, palette
  entity-blue #2a78d6, event-green #1baf7a, human-green #1baf7a, red accent
  #c5544a, spines top/right removed, value labels with one decimal.
- Output each COMPLETE runnable script in its own fenced code block, nothing
  else. matplotlib/numpy only. Each script saves PDF and PNG (dpi=300) next
  to itself, like the current versions.

The four figures:
1. make_fig_scatter.py: detection versus clarification-rate scatter with the
   perceive-act parity diagonal and the human vertical reference line. A
   model appears as a point when DATA holds both its detection hit rate and
   its clarification rate.
2. make_fig_subsets.py: entity-versus-event dumbbell, ordered by gap. Models
   without entity_event_strictK entries render as label-only rows with a
   small gray "pending" marker at the axis, keeping the full roster visible.
3. make_fig_perk.py: score-by-K line chart. Closed-weight models solid,
   open-weight dashed, distinct markers, models without score_by_K entries
   listed in a gray footnote line under the legend.
4. make_fig_responses.py (new, write from scratch in the same house style):
   diverging stacked bar chart of the five-way first-response distribution,
   aligned at the boundary between proactive behavior (enumerate,
   scope-anchored clarification, drawn leftward) and passive or erroneous
   behavior (vague clarification, silent commitment, refusal, drawn
   rightward). Models without response_distribution entries render as gray
   pending rows.

DATA:
```json
{
  "detection": {
    "_comment": "hit % on 100 ambiguous items, false-alarm % on 40 controls; sources: human study responses, analysis/binary_judgment_gpt-5.json, training-box binary-judgment runs",
    "Humans (4 annotators)": {"hit": 84.5, "fa": 3.2},
    "GPT-5":                 {"hit": 75.3, "fa": 12.1},
    "Qwen3-VL-32B":          {"hit": 28.5, "fa": 5.0},
    "GPT-4o":                {"hit": 4.0,  "fa": 0.0}
  },
  "clarification_rate": {
    "_comment": "Table 3 of the paper (percent of questions)",
    "GPT-4o": 1.7, "Gemini-3-Flash": 15.7, "Gemini-3.5-Flash": 8.8,
    "Qwen3.5-27B": 4.6, "Qwen3.6-27B": 4.1, "Qwen3-VL-32B": 17.0,
    "InternVL3.5-8B": 3.2, "InternVL3.5-38B": 12.5
  },
  "entity_event_strictK": {
    "_comment": "percent; source: analysis/per_subset_metrics.json (pilot runs)",
    "GPT-4o":       {"entity": 0.0, "event": 0.0},
    "Qwen3.5-27B":  {"entity": 6.7, "event": 2.0},
    "Qwen3.6-27B":  {"entity": 4.5, "event": 2.7},
    "Qwen3-VL-32B": {"entity": 4.9, "event": 1.2}
  },
  "score_by_K": {
    "_comment": "ReQueST score % per K bin [2, 3, 4-6, 7+]; source: draft section 4 runs",
    "GPT-4o":           [0.8, 0.5, 0.3, 0.1],
    "Gemini-3-Flash":   [28.4, 22.6, 15.1, 6.1],
    "Gemini-3.5-Flash": [25.9, 20.8, 13.9, 5.7]
  },
  "response_distribution": {
    "_comment": "five-way first-response shares per model, percent, sums to 100"
  }
}
```

CURRENT SCRIPTS:

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

```python
"""Figure: entity vs event strict-K per base model (section 4.3).

Data: analysis/per_subset_metrics.json (computed from the pilot runs).
Output: paper/figures/fig_subsets.{pdf,png}, one ACL column.
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent

C_ENTITY = "#2a78d6"
C_EVENT = "#1baf7a"
INK = "#333333"

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

data = json.load(open(ROOT / "analysis" / "per_subset_metrics.json"))
MODELS = [("GPT-4o base", "GPT-4o"), ("Qwen3.5 base", "Qwen3.5-27B"),
          ("Qwen3.6 base", "Qwen3.6-27B"), ("Qwen3-VL-32B base", "Qwen3-VL-32B")]
entity = [100 * data[k]["Entity"]["strict"] for k, _ in MODELS]
event = [100 * data[k]["Event"]["strict"] for k, _ in MODELS]

# dumbbell plot, ordered by the entity-event gap (unanimous two-model vote,
# figformat_log.md): one row per model, entity and event dots connected.
# re-scored per-subset breakdown lands (outline item 10).
order = sorted(range(len(MODELS)), key=lambda i: entity[i] - event[i])
fig, ax = plt.subplots(figsize=(3.03, 0.30 * len(MODELS) + 0.55))
for row, i in enumerate(order):
    e, v = entity[i], event[i]
    # coincident pairs (e.g. 0.0/0.0): nudge apart vertically so both show
    ye, yv = (row + 0.10, row - 0.10) if abs(e - v) < 0.05 else (row, row)
    ax.plot([v, e], [yv, ye], color="#bbbbbb", lw=1.1, zorder=1)
    ax.scatter([e], [ye], s=22, color=C_ENTITY, marker="s", zorder=3,
               label="Entity" if row == 0 else None)
    ax.scatter([v], [yv], s=24, color=C_EVENT, marker="o", zorder=3,
               label="Event" if row == 0 else None)
    if e >= 0.2:
        ax.text(e + 0.15, ye, f"{e:.1f}", fontsize=6.0, color=INK,
                ha="left", va="center")
    if v >= 0.2:
        ax.text(v - 0.15, yv, f"{v:.1f}", fontsize=6.0, color=INK,
                ha="right", va="center")
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([MODELS[i][1] for i in order])
ax.set_xlabel("strict-$K$ (%)")
ax.set_xlim(-0.6, 8.2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", handlelength=0.9, fontsize=5.5,
          markerscale=0.85, borderpad=0.2)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_subsets.pdf")
fig.savefig(OUT / "fig_subsets.png", dpi=300)
print("wrote", OUT / "fig_subsets.pdf")

```

```python
"""Figure: ReQueST score by K bin for the closed-weight models (section 4.2).

Data provenance: IAA-protocol runs recorded in paper_draft.md section 4.1.
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
    "GPT-4o": [1.1, 0.0, 0.0, 0.0],
    "Gemini-3-Flash": [35.4, 6.9, 11.2, 7.6],
    "Gemini-3.5-Flash": [32.2, 8.6, 12.2, 4.2],
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
