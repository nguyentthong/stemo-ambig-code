"""Figures for the open-weight IAA eval, from analysis/open_weight_iaa_metrics.json.

Produces (in analysis/figures/):
  fig_scores   — overall IAA score + the Table-1 metric breakdown
  fig_perk     — IAA score vs ambiguity level K (the degradation curve)
  fig_subsets  — strict-K accuracy by Entity / Event / Mixed subset
  overview     — all three panels in one image for quick viewing

Color: one fixed Okabe-Ito (colorblind-safe) hue per model, identical across
every panel, assigned in fixed order (never cycled).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

REPO = Path(__file__).resolve().parents[1]
DATA = json.load(open(REPO / "analysis/open_weight_iaa_metrics.json"))
OUT = REPO / "analysis/figures"
OUT.mkdir(parents=True, exist_ok=True)

# fixed model order + display names + colorblind-safe colors (Okabe-Ito)
# internvl38b is EXCLUDED: its -HF checkpoint collapses to single_commit
# (~93%, clarifies 0.3% vs the 8B's 14%) — verified NOT a tensor-parallel
# effect (identical behavior at tp=2 and tp=4) and NOT vision-blindness
# (it reads on-screen scoreboards/colors). Anomalous instruction-following on
# the conversion, so it is omitted from figures rather than reported.
EXCLUDED = ["internvl38b"]
MODELS = ["internvl8b", "qwen35_27b", "qwen36_27b", "qwen3vl_32b"]
LABEL = {
    "internvl8b": "InternVL3.5-8B",
    "internvl38b": "InternVL3.5-38B",
    "qwen35_27b": "Qwen3.5-27B",
    "qwen36_27b": "Qwen3.6-27B",
    "qwen3vl_32b": "Qwen3-VL-32B",
}
COLOR = {
    "internvl8b": "#0072B2",   # blue
    "internvl38b": "#56B4E9",  # light blue (same family, lighter)
    "qwen35_27b": "#E69F00",   # orange
    "qwen36_27b": "#D55E00",   # vermillion
    "qwen3vl_32b": "#009E73",  # bluish green
}

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.edgecolor": "#666666",
    "figure.dpi": 150,
})


def pct(x):
    return 100.0 * x


def style_axis(ax):
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def bar_value_labels(ax, bars, vals, horiz=False):
    for b, v in zip(bars, vals):
        if horiz:
            ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2,
                    f"{v:.1f}", va="center", ha="left", fontsize=8.5, color="#333333")
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                    f"{v:.1f}", va="bottom", ha="center", fontsize=8.5, color="#333333")


# ---------- panel A: overall IAA score ----------
def draw_scores(ax):
    order = sorted(MODELS, key=lambda m: DATA[m]["iaa"])  # ascending -> best on top
    vals = [pct(DATA[m]["iaa"]) for m in order]
    ypos = range(len(order))
    bars = ax.barh(list(ypos), vals, color=[COLOR[m] for m in order], height=0.62,
                   edgecolor="white", linewidth=0.8)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([LABEL[m] for m in order])
    ax.set_xlabel("IAA score")
    ax.set_title("Overall IAA score", fontweight="bold", loc="left")
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.set_xlim(0, max(vals) * 1.18)
    bar_value_labels(ax, bars, vals, horiz=True)
    style_axis(ax)


# ---------- panel B: score vs K ----------
def draw_perk(ax):
    kbins = list(DATA[MODELS[0]]["per_K"].keys())  # ['2','3','4-6','7+']
    x = range(len(kbins))
    for m in MODELS:
        y = [pct(DATA[m]["per_K"][k]["iaa"]) for k in kbins]
        ax.plot(list(x), y, marker="o", markersize=6, linewidth=2,
                color=COLOR[m], label=LABEL[m], markeredgecolor="white",
                markeredgewidth=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"K={k}" for k in kbins])
    ax.set_xlabel("Ambiguity level (number of interpretations)")
    ax.set_ylabel("IAA score")
    ax.set_title("IAA score vs. ambiguity level", fontweight="bold", loc="left")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.06)
    style_axis(ax)


# ---------- panel C: strict-K by subset ----------
def draw_subsets(ax):
    subs = list(DATA[MODELS[0]]["per_subset"].keys())  # Entity, Event, Mixed
    n = len(MODELS)
    group_w = 0.8
    bw = group_w / n
    for i, m in enumerate(MODELS):
        xs = [j - group_w / 2 + bw * (i + 0.5) for j in range(len(subs))]
        ys = [pct(DATA[m]["per_subset"][s]["strict_K"]) for s in subs]
        ax.bar(xs, ys, width=bw * 0.92, color=COLOR[m], label=LABEL[m],
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([f"{s}\n(n={DATA[MODELS[0]]['per_subset'][s]['n']})" for s in subs])
    ax.set_ylabel("Strict-K accuracy")
    ax.set_title("Strict-K accuracy by ambiguity type", fontweight="bold", loc="left")
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    ax.set_ylim(bottom=0)
    style_axis(ax)


# ---------- panel D: metric breakdown ----------
def draw_metrics(ax):
    metrics = [("recognition", "Recognition"), ("clarification_rate", "Clarification"),
               ("strict_K", "Strict-K"), ("follow_through", "Follow-through")]
    n = len(MODELS)
    group_w = 0.82
    bw = group_w / n
    for i, m in enumerate(MODELS):
        xs = [j - group_w / 2 + bw * (i + 0.5) for j in range(len(metrics))]
        ys = [pct(DATA[m][k]) for k, _ in metrics]
        ax.bar(xs, ys, width=bw * 0.92, color=COLOR[m], label=LABEL[m],
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Behavioral metric breakdown", fontweight="bold", loc="left")
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    ax.set_ylim(bottom=0)
    style_axis(ax)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[figures] wrote {name}.png / .pdf")


# individual figures
f, ax = plt.subplots(figsize=(6.2, 3.4)); draw_scores(ax); save(f, "fig_scores_overall")
f, ax = plt.subplots(figsize=(6.6, 4.0)); draw_perk(ax)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1); save(f, "fig_perk")
f, ax = plt.subplots(figsize=(6.6, 4.0)); draw_subsets(ax)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1); save(f, "fig_subsets")
f, ax = plt.subplots(figsize=(6.6, 4.0)); draw_metrics(ax)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1); save(f, "fig_metrics")

# combined overview (2x2)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
draw_scores(axes[0, 0]); draw_perk(axes[0, 1])
draw_subsets(axes[1, 0]); draw_metrics(axes[1, 1])
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=len(MODELS), loc="lower center",
           bbox_to_anchor=(0.5, 0.02), fontsize=10)
fig.suptitle("Open-weight IAA evaluation — STEMO ambiguous-referent benchmark (1,056 questions)",
             fontweight="bold", fontsize=13, y=1.0)
note = ("InternVL3.5-38B omitted: its -HF checkpoint collapses to single-commit "
        "answers (clarifies 0.3% vs the 8B's 14%); verified independent of "
        "tensor-parallel size, so treated as an invalid run pending re-evaluation.")
fig.text(0.5, -0.005, note, ha="center", va="top", fontsize=8, color="#666666", style="italic")
fig.tight_layout(rect=(0, 0.04, 1, 0.99))
save(fig, "overview")
