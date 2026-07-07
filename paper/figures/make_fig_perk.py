"""Figure: ReQueST score by K bin for all evaluated models.

One line per model, grouped into four colour families (GPT, Gemini, Qwen,
InternVL). Within a family the newer/larger sibling is solid + filled and the
older/smaller one dashed/dotted + open. Per-K ReQueST score (the "iaa" credit):
  closed: analysis/gpt5_metrics.json (GPT-5) + section 4.1 runs (Gemini, GPT-4o)
  open:   analysis/open_weight_iaa_metrics.json (per_K iaa)
All values reconcile with the overall scores in Table 1.
Output: paper/figures/fig_perk.{pdf,png}, one ACL column.
"""
import pathlib

import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

INK = "#333333"
# Okabe-Ito colourblind-safe families. Warm = closed, cool = open.
C_GPT = "#D55E00"
C_GEMINI = "#009E73"
C_QWEN = "#0072B2"
C_INTERN = "#CC79A7"

BINS = ["2", "3", "4–6", "7+"]

# name -> (per-K score %, colour, marker, linestyle, filled)
MODELS = [
    ("GPT-5",            [53.1, 34.0, 33.5, 31.7], C_GPT,    "o", "-",  True),
    ("GPT-4o",           [ 1.1,  0.0,  0.0,  0.0], C_GPT,    "o", "--", False),
    ("Gemini-3-Flash",   [35.4,  6.9, 11.2,  7.6], C_GEMINI, "s", "-",  True),
    ("Gemini-3.5-Flash", [32.2,  8.6, 12.2,  4.2], C_GEMINI, "s", "--", False),
    ("Qwen3-VL-32B",     [25.1,  7.8, 14.0, 24.3], C_QWEN,   "^", "-",  True),
    ("Qwen3.6-27B",      [25.5, 13.8,  5.8,  8.2], C_QWEN,   "^", "--", True),
    ("Qwen3.5-27B",      [25.2, 12.1,  3.8,  3.6], C_QWEN,   "^", ":",  False),
    ("InternVL3.5-38B",  [ 2.5,  1.7,  0.3,  0.0], C_INTERN, "D", "-",  True),
    ("InternVL3.5-8B",   [12.8,  3.4,  4.9,  8.2], C_INTERN, "D", "--", False),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 5.4,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.03, 1.55))
x = range(len(BINS))

for name, ys, color, marker, ls, filled in MODELS:
    ax.plot(x, ys, marker=marker, ms=3.0, lw=1.0, ls=ls, color=color,
            label=name, markerfacecolor=color if filled else "white",
            markeredgecolor=color, markeredgewidth=0.8, clip_on=False,
            zorder=3 if filled else 2)

ax.set_xticks(list(x))
ax.set_xticklabels([f"$K$={b}" for b in BINS])
ax.set_ylabel("ReQueST score (%)")
ax.set_ylim(-2, 56)
ax.set_xlim(-0.15, len(BINS) - 0.85)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
ax.set_axisbelow(True)

# legend below the axis, two compact rows
leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=5, handlelength=1.2, columnspacing=0.7, labelspacing=0.3,
                handletextpad=0.3, borderaxespad=0.0, fontsize=4.8)

fig.savefig(OUT / "fig_perk.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_perk.png", dpi=300, bbox_inches="tight")
print("wrote", OUT / "fig_perk.pdf")
