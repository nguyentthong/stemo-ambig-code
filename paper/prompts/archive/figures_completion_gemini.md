# Prompts for Gemini: complete the paper figures

Paste each prompt into Gemini together with the requested data. Every prompt
is self-contained and includes the current script.

---

## Prompt 1 — complete Figure 6 (ambiguity detection)

You are completing a matplotlib figure script for an ACL paper. The figure
shows binary ambiguity detection (hit rate on 100 ambiguous questions, red
tick for the false-alarm rate on 40 controls), horizontal bars, humans
anchored on top, in the style of RAcQUEt Figure 3. Six model rows are
currently pending placeholders.

Task: replace the six None entries in ROWS with the values I paste below,
keep the row order (humans first, then models sorted by hit rate
descending), and remove the pending-placeholder branch if no None remains.
- Match the house style exactly: the rcParams block already in the script
  (Times-like serif, 7 pt, pdf.fonttype 42), figure width 3.03 in for one
  ACL column, palette entity-blue #2a78d6 / event-green #1baf7a /
  human-green #1baf7a / model-blue #2a78d6 / red accent #c5544a,
  no chart junk (no titles inside the plot, no heavy grids, spines top/right
  removed), value labels at bar ends with one decimal.
- Output the COMPLETE runnable script, nothing else. No new dependencies
  beyond matplotlib/numpy. The script must save both PDF and PNG (dpi=300)
  to the same directory as the script, exactly like the current version.

CURRENT SCRIPT:
```python
"""Figure: binary ambiguity detection, humans vs models (section 4.6).

RAcQUEt Figure-3 style: horizontal bars, one row per rater, human anchor on
top. Hit rate on the 100 ambiguous items (solid) with the false-alarm rate
on the 40 controls (thin overlay marker).

DATA: fill the six pending model rows from the finished binary-judgment run
(same 140 items). Rows with None render as gray "pending" placeholders.
Output: paper/figures/fig_detection.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent

C_HUMAN = "#1baf7a"
C_MODEL = "#2a78d6"
C_PEND = "#d9d9d9"
INK = "#333333"

# (label, hit %, false-alarm %) — None = pending
ROWS = [
    ("Humans (4 annotators)", 84.5, 3.2),
    ("Qwen3-VL-32B", 28.5, 5.0),
    ("GPT-4o", 4.0, 0.0),
    ("Gemini-3-Flash", None, None),      # fill in your values
    ("Gemini-3.5-Flash", None, None),    # fill in your values
    ("Qwen3.5-27B", None, None),         # fill in your values
    ("Qwen3.6-27B", None, None),         # fill in your values
    ("InternVL3.5-38B", None, None),     # fill in your values
    ("InternVL3.5-8B", None, None),      # fill in your values
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
        ax.barh(yi, 100, height=0.62, color=C_PEND, alpha=0.35)
        ax.text(50, yi, "pending", ha="center", va="center",
                fontsize=6.0, color="#8a8a8a", style="italic")
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

---

## Prompt 2 — extend Figure 5 (entity versus event) to all eight models

You are extending a matplotlib figure script. It currently shows entity
versus event strict-K for four models as horizontal grouped bars. Extend it
to the evaluated models using the values I paste. Drop rows whose entity
and event values are both 0.0 (they cannot show a gap): list those models
in a code comment so the caption can name them in prose. Keep bars sorted
so the largest entity value is at the top, keep the value labels, and adjust
the x-limit to fit the new maximum with headroom. The script currently
loads four models from a JSON file: replace that loading with an explicit
DATA list in the script containing all eight models (label, entity %,
event %), with a comment naming the source run.
- Match the house style exactly: the rcParams block already in the script
  (Times-like serif, 7 pt, pdf.fonttype 42), figure width 3.03 in for one
  ACL column, palette entity-blue #2a78d6 / event-green #1baf7a /
  human-green #1baf7a / model-blue #2a78d6 / red accent #c5544a,
  no chart junk (no titles inside the plot, no heavy grids, spines top/right
  removed), value labels at bar ends with one decimal.
- Output the COMPLETE runnable script, nothing else. No new dependencies
  beyond matplotlib/numpy. The script must save both PDF and PNG (dpi=300)
  to the same directory as the script, exactly like the current version.


CURRENT SCRIPT:
```python
"""Figure: entity vs event strict-K per base model (section 4.3).

Data: analysis/per_subset_metrics.json (computed from the pilot runs).
TODO(re-score): regenerate from re-scored runs (outline item 10).
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

# horizontal bars, RAcQUEt-style: readable labels, ranking top-down
# fill in your values: extend MODELS with Gemini + InternVL per-subset values when the
# re-scored per-subset breakdown lands (outline item 10).
fig, ax = plt.subplots(figsize=(3.03, 0.42 * len(MODELS) + 0.55))
ypos = np.arange(len(MODELS))[::-1]
h = 0.36
b1 = ax.barh(ypos + h / 2, entity, h, color=C_ENTITY, label="Entity")
b2 = ax.barh(ypos - h / 2, event, h, color=C_EVENT, label="Event")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.1f}", ha="left", va="center",
                fontsize=6.0, color=INK)
ax.set_yticks(ypos)
ax.set_yticklabels([lbl for _, lbl in MODELS])
ax.set_xlabel("strict-$K$ (%)")
ax.set_xlim(0, 8.2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", handlelength=1.4)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_subsets.pdf")
fig.savefig(OUT / "fig_subsets.png", dpi=300)
print("wrote", OUT / "fig_subsets.pdf")

```

---

## Prompt 3 — refresh Figure 4 (score by K bin) with re-scored values

You are updating a matplotlib line chart of the ReQueST score by K bin
(bins 2 / 3 / 4-6 / 7+). It currently shows three closed-weight models.
Update the SCORE dict with the re-scored values I paste. If I paste more
than five models, keep the chart readable: solid lines for closed-weight
models, dashed for open-weight, distinct markers, and a two-column legend.
Keep it a LINE chart: the x-axis is ordinal and the shape of the decline is
the point.
- Match the house style exactly: the rcParams block already in the script
  (Times-like serif, 7 pt, pdf.fonttype 42), figure width 3.03 in for one
  ACL column, palette entity-blue #2a78d6 / event-green #1baf7a /
  human-green #1baf7a / model-blue #2a78d6 / red accent #c5544a,
  no chart junk (no titles inside the plot, no heavy grids, spines top/right
  removed), value labels at bar ends with one decimal.
- Output the COMPLETE runnable script, nothing else. No new dependencies
  beyond matplotlib/numpy. The script must save both PDF and PNG (dpi=300)
  to the same directory as the script, exactly like the current version.

CURRENT SCRIPT:
```python
"""Figure: ReQueST score by K bin for the closed-weight models (section 4.2).

Data provenance: IAA-protocol runs recorded in paper_draft.md section 4.1.
TODO(re-score): regenerate from re-scored runs (outline item 10).
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

---

## Prompt 4 — create the response-distribution figure (new)

You are writing a NEW matplotlib script, make_fig_responses.py, in the same
house style as the script below (copy its rcParams and general conventions).
The figure is a DIVERGING stacked bar chart (not RAcQUEt's plain 0-100%
stack): one horizontal bar per model showing the five-way first-response
distribution, aligned at the boundary between proactive behavior
(enumerate, scope-anchored clarification, drawn leftward from zero) and
passive or erroneous behavior (vague clarification, silent commitment,
refusal, drawn rightward), so bar alignment ranks models by proactivity. Category colors: enumerate
#1baf7a, scope-anchored #2a78d6, vague #9bbce0, commit #c5544a, refusal
#d9d9d9. Legend above the axes in one row, small font. Models ordered by
commitment share ascending (least-committing model on top). Percentages
above 8 points get a white value label inside their segment.
- Match the house style exactly: the rcParams block already in the script
  (Times-like serif, 7 pt, pdf.fonttype 42), figure width 3.03 in for one
  ACL column, palette entity-blue #2a78d6 / event-green #1baf7a /
  human-green #1baf7a / model-blue #2a78d6 / red accent #c5544a,
  no chart junk (no titles inside the plot, no heavy grids, spines top/right
  removed), value labels at bar ends with one decimal.
- Output the COMPLETE runnable script, nothing else. No new dependencies
  beyond matplotlib/numpy. The script must save both PDF and PNG (dpi=300)
  to the same directory as the script, exactly like the current version.

STYLE REFERENCE SCRIPT (copy its conventions):
```python
"""Figure: binary ambiguity detection, humans vs models (section 4.6).

RAcQUEt Figure-3 style: horizontal bars, one row per rater, human anchor on
top. Hit rate on the 100 ambiguous items (solid) with the false-alarm rate
on the 40 controls (thin overlay marker).

DATA: fill the six pending model rows from the finished binary-judgment run
(same 140 items). Rows with None render as gray "pending" placeholders.
Output: paper/figures/fig_detection.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent

C_HUMAN = "#1baf7a"
C_MODEL = "#2a78d6"
C_PEND = "#d9d9d9"
INK = "#333333"

# (label, hit %, false-alarm %) — None = pending
ROWS = [
    ("Humans (4 annotators)", 84.5, 3.2),
    ("Qwen3-VL-32B", 28.5, 5.0),
    ("GPT-4o", 4.0, 0.0),
    ("Gemini-3-Flash", None, None),      # fill in your values
    ("Gemini-3.5-Flash", None, None),    # fill in your values
    ("Qwen3.5-27B", None, None),         # fill in your values
    ("Qwen3.6-27B", None, None),         # fill in your values
    ("InternVL3.5-38B", None, None),     # fill in your values
    ("InternVL3.5-8B", None, None),      # fill in your values
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
        ax.barh(yi, 100, height=0.62, color=C_PEND, alpha=0.35)
        ax.text(50, yi, "pending", ha="center", va="center",
                fontsize=6.0, color="#8a8a8a", style="italic")
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
