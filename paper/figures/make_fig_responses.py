"""Signature figure: diverging bar chart of model response distributions.

Diverging stacked bar chart showing each model's five-way first-response 
distribution, aligned at the boundary between proactive behavior (leftward)
and passive/erroneous behavior (rightward).

Output: paper/figures/fig_responses.{pdf,png}, one ACL column.
"""
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

OUT = pathlib.Path(__file__).resolve().parent

INK = "#333333"

# Category definitions and colors
CATEGORIES = {
    "enumerate": "#1baf7a",
    "scope-anchored": "#2a78d6",
    "vague": "#9bbce0",
    "commit": "#c5544a",
    "refuse": "#d9d9d9"
}

# (label, response_distribution dict)
# Proactive: enumerate, scope-anchored. Passive/Error: vague, commit, refuse.
# All valid distributions should sum to 100%.
MODELS = [
    # five-way first-response shares (%), source: the scored run rows
    # (analysis/gpt5_metrics.json companion jsonl). None = distribution not
    # yet computed (renders as a gray pending row). Model-only figure:
    # humans have no protocol run by design.
    ("GPT-5", {"enumerate": 25.6, "scope-anchored": 36.9, "vague": 3.6,
               "commit": 25.5, "refuse": 8.3}),
    ("GPT-4o", None),
    ("Gemini-3-Flash", None),
    ("Gemini-3.5-Flash", None),
    ("Qwen3.5-27B", None),
    ("Qwen3.6-27B", None),
    ("Qwen3-VL-32B", None),
    ("InternVL3.5-8B", None),
    ("InternVL3.5-38B", None),
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

fig, ax = plt.subplots(figsize=(3.03, 2.85))

# measured models sorted by proactivity (most proactive on top), then the
# pending models below as gray placeholder rows keeping the roster visible
valid_models = [(l, d, d["enumerate"] + d["scope-anchored"])
                for l, d in MODELS if d is not None]
valid_models.sort(key=lambda x: x[2])
pending_models = [l for l, d in MODELS if d is None][::-1]

labels = pending_models + [x[0] for x in valid_models]
n_pend = len(pending_models)
y_pos = np.arange(n_pend, n_pend + len(valid_models))
for yy in range(n_pend):
    ax.barh(yy, 100, left=-50, height=0.7, color="#d9d9d9", alpha=0.30)
    ax.text(0, yy, "pending", ha="center", va="center",
            fontsize=5.5, color="#8a8a8a", style="italic")
bar_height = 0.7

# Extract distribution arrays
enums = np.array([x[1]["enumerate"] for x in valid_models])
scopes = np.array([x[1]["scope-anchored"] for x in valid_models])
vagues = np.array([x[1]["vague"] for x in valid_models])
commits = np.array([x[1]["commit"] for x in valid_models])
refuses = np.array([x[1]["refuse"] for x in valid_models])

# Plot Leftward (Proactive: scope-anchored nearest to zero, then enumerate)
ax.barh(y_pos, -scopes, height=bar_height, color=CATEGORIES["scope-anchored"], label="scope-anchored")
ax.barh(y_pos, -enums, left=-scopes, height=bar_height, color=CATEGORIES["enumerate"], label="enumerate")

# Plot Rightward (Passive/Error: vague nearest to zero, then commit, then refuse)
ax.barh(y_pos, vagues, height=bar_height, color=CATEGORIES["vague"], label="vague")
ax.barh(y_pos, commits, left=vagues, height=bar_height, color=CATEGORIES["commit"], label="commit")
ax.barh(y_pos, refuses, left=(vagues + commits), height=bar_height, color=CATEGORIES["refuse"], label="refuse")

# Add white text labels for segments wider than 8 points
for k, i in enumerate(y_pos):
    # Leftward segments
    s, e = scopes[k], enums[k]
    if s > 8:
        ax.text(-s / 2, i, f"{s:.1f}", ha='center', va='center', color='white', fontsize=6)
    if e > 8:
        ax.text(-s - (e / 2), i, f"{e:.1f}", ha='center', va='center', color='white', fontsize=6)
        
    # Rightward segments
    v, c, r = vagues[k], commits[k], refuses[k]
    if v > 8:
        ax.text(v / 2, i, f"{v:.1f}", ha='center', va='center', color='white', fontsize=6)
    if c > 8:
        ax.text(v + (c / 2), i, f"{c:.1f}", ha='center', va='center', color='white', fontsize=6)
    if r > 8:
        ax.text(v + c + (r / 2), i, f"{r:.1f}", ha='center', va='center', color='white', fontsize=6)

# Formatting axes
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("response distribution (%)")
ax.spines[["top", "right"]].set_visible(False)

# Format x-axis to show positive percentages on both sides
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):g}"))
ax.axvline(0, color=INK, lw=0.8, zorder=3)
ax.grid(axis='x', color="#d8d8d8", lw=0.5, alpha=0.9)
ax.set_axisbelow(True)

# One-row legend above the axes (using title pad as a spacing trick to avoid overlap)
ax.set_title(" ", pad=18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False, 
          fontsize=5.5, handletextpad=0.3, columnspacing=0.8, borderpad=0)

fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_responses.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_responses.png", dpi=300, bbox_inches="tight")