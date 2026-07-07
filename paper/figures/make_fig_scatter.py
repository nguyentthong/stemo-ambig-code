"""Signature figure: ambiguity detection versus action (section 4.6).

Scatter, one point per model: x = binary detection hit rate on the 100
ambiguous study items, y = clarification rate under the protocol (Table 1).
The diagonal y = x marks perceive-act parity. Humans have a measured
detection rate but no protocol run by design, so they appear as a hollow
marker ON the parity line (parity assumed), with a dotted vertical drop.

Models are identified through a legend (upper-left triangle is empty by
construction, since no model exceeds parity): one Okabe-Ito color per model
family, open versus filled markers to separate siblings within a family.

Output: fig_scatter.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

OUT = pathlib.Path(__file__).resolve().parent

INK = "#333333"
HUMAN_DETECT = 84.5

# Okabe-Ito family colors
C_GPT = "#D55E00"
C_GEMINI = "#009E73"
C_QWEN = "#0072B2"
C_INTERNVL = "#CC79A7"

# (label, detection %, clarification %, color, marker, filled)
# clarification = Table 1 Clar column (open-weight re-scored under the vLLM
# IAA run, analysis/open_weight_iaa_metrics.json); detection = Table 4.
# InternVL3.5-38B omitted: its IAA re-run collapsed (see fig_subsets note).
MODELS = [
    ("GPT-5", 75.3, 40.8, C_GPT, "o", True),
    ("GPT-4o", 4.0, 1.7, C_GPT, "o", False),
    ("Gemini-3-Flash", 54.9, 15.7, C_GEMINI, "s", True),
    ("Gemini-3.5-Flash", 52.4, 8.8, C_GEMINI, "s", False),
    ("Qwen3-VL-32B", 28.5, 22.0, C_QWEN, "^", True),
    ("Qwen3.5-27B", 18.3, 9.7, C_QWEN, "v", False),
    ("Qwen3.6-27B", 21.7, 9.4, C_QWEN, "v", True),
    ("InternVL3.5-8B", 14.2, 16.7, C_INTERNVL, "D", False),
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

# Height 2.8 so `equal` aspect doesn't crush the width below the 3.03in target
fig, ax = plt.subplots(figsize=(3.03, 2.8))
XMAX, YMAX = 100, 90

ax.grid(True, color="#dddddd", lw=0.4, zorder=0)
ax.fill_between([0, XMAX], [0, 0], [0, XMAX], color="#8fa8c4", alpha=0.06, zorder=0.5)
ax.plot([0, XMAX], [0, XMAX], ls=(0, (5, 3)), lw=0.9, color="#666666", zorder=1)

ax.text(58.5, 58.5, "parity", rotation=45, fontsize=7.0, style="italic",
        color="#555555", ha="center", va="bottom",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

# Humans: measured detection, parity assumed (no protocol run by design)
ax.plot([HUMAN_DETECT, HUMAN_DETECT], [0, HUMAN_DETECT], ls=":", lw=0.9,
        color="#555555", zorder=1)
ax.scatter([HUMAN_DETECT], [HUMAN_DETECT], s=50, facecolors="white",
           edgecolors=INK, linewidths=1.1, zorder=4)
ax.text(HUMAN_DETECT - 2.5, HUMAN_DETECT - 2.0, "humans (84.5%,\nparity assumed)",
        fontsize=6.0, color="#555555", ha="right", va="top", linespacing=1.15,
        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

ax.text(62, 26, "perceives without acting", fontsize=6.8, color="#888888",
        style="italic", ha="center",
        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

for label, det, clar, color, marker, filled in MODELS:
    ax.scatter([det], [clar], s=26, marker=marker,
               facecolors=color if filled else "white",
               edgecolors=color, linewidths=0.8,
               label=label, zorder=3)

leg = ax.legend(loc="upper left", fontsize=6.0, frameon=False,
                handletextpad=0.25, labelspacing=0.42, borderaxespad=0.25,
                handlelength=1.0)

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
