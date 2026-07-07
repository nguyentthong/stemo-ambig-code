# Prompt: entity-versus-event dumbbell (fig_subsets)

You are finishing a matplotlib dumbbell figure for an ACL paper. One row per
model: a square marker for entity strict-K and a circle for event strict-K,
connected by a gray line, rows ordered by the entity-event gap, values
labeled at the markers with one decimal (labels suppressed below 0.2, and
coincident pairs nudged vertically so both markers stay visible). Models
with no entity_event_strictK entry in DATA render as label-only rows with a
small gray open marker at zero, keeping the full roster visible.
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
  "entity_event_strictK": {
    "GPT-5":            null,
    "GPT-4o":           {"entity": 0.0, "event": 0.0},
    "Gemini-3-Flash":   null,
    "Gemini-3.5-Flash": null,
    "Qwen3.5-27B":      {"entity": 6.7, "event": 2.0},
    "Qwen3.6-27B":      {"entity": 4.5, "event": 2.7},
    "Qwen3-VL-32B":     {"entity": 4.9, "event": 1.2},
    "InternVL3.5-8B":   null,
    "InternVL3.5-38B":  null
  }
}
```

CURRENT SCRIPT:
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
