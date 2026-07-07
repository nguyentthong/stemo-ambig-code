"""Figure 2: ReQueST dataset statistics (K distribution + temporal spread CDF).

Computed directly from data_v0/stemo_ambig_candidates/all_questions.json.
Output: paper/figures/fig2_stats.{pdf,png}, sized for one ACL column.
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent

ENTITY = {"shared_attribute_different_entities", "entities_in_same_event",
          "multiple_entities", "repeated_entities"}
EVENT = {"repeated_action", "same_entity_multiple_moments", "repeated_temporal_referent"}

C_ENTITY = "#2a78d6"  # categorical slot 1 (blue)
C_EVENT = "#1baf7a"   # categorical slot 2 (aqua)
INK = "#333333"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load():
    qs = json.load(open(ROOT / "data_v0/stemo_ambig_candidates/all_questions.json"))["questions"]
    ent, evt = [], []
    for q in qs:
        sub = q["subcategory"]
        grp = ent if sub in ENTITY else evt if sub in EVENT else None
        if grp is None:
            continue
        k = len(q["interpretations"])
        ts = [t for it in q["interpretations"]
              for sp in (it.get("vlm_proposed_evidence_spans") or []) for t in sp]
        spread = (max(ts) - min(ts)) if ts else None
        grp.append((k, spread))
    return ent, evt


def main():
    ent, evt = load()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.03, 1.42))

    # --- Panel (a): K distribution, binned, grouped bars ---
    bins = [(2, 2, "2"), (3, 4, "3–4"), (5, 8, "5–8"), (9, 16, "9–16"), (17, 999, "17+")]
    x = np.arange(len(bins))
    w = 0.38
    for off, data, color, name, nud in ((-w / 2, ent, C_ENTITY, "Entity", -0.06),
                                        (w / 2, evt, C_EVENT, "Event", 0.06)):
        ks = [k for k, _ in data]
        counts = [sum(1 for k in ks if lo <= k <= hi) for lo, hi, _ in bins]
        bars = ax1.bar(x + off, counts, w, color=color, label=name, zorder=3)
        for b, c in zip(bars, counts):
            ax1.annotate(str(c), (b.get_x() + b.get_width() / 2 + nud, b.get_height() + 6),
                         ha="center", va="bottom", fontsize=5.2, color=INK)
    ax1.set_xticks(x, [b[2] for b in bins])
    ax1.set_xlabel("Readings per question $K$", labelpad=1.5)
    ax1.set_ylabel("Questions")
    ax1.set_ylim(0, 340)
    ax1.yaxis.grid(True, color="#dddddd", linewidth=0.5, zorder=0)
    ax1.legend(frameon=False, borderpad=0.1, handlelength=1.0, handletextpad=0.4,
               labelspacing=0.25, loc="upper right")

    # --- Panel (b): CDF of temporal spread of a question's readings ---
    for data, color, name, ly, fx, ha in ((ent, C_ENTITY, "Entity", 0.72, 0.50, "right"),
                                          (evt, C_EVENT, "Event", 0.42, 1.45, "left")):
        spreads = np.sort([max(s, 1.0) for _, s in data if s is not None])
        cdf = np.arange(1, len(spreads) + 1) / len(spreads)
        ax2.plot(spreads, cdf, color=color, linewidth=1.1, zorder=3)
        ax2.annotate(name, xy=(np.interp(ly, cdf, spreads) * fx, ly), ha=ha,
                     fontsize=6.5, color=color, style="italic", va="center")
    ax2.set_xscale("log")
    ax2.set_xticks([1, 10, 60, 300], ["1s", "10s", "1min", "5min"])
    ax2.set_xlim(1, 700)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0, 0.5, 1.0], ["0", ".5", "1"])
    ax2.set_xlabel("Temporal spread of readings", labelpad=1.5)
    ax2.set_ylabel("Cum. fraction")
    ax2.grid(True, color="#dddddd", linewidth=0.5, zorder=0)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)

    fig.tight_layout(pad=0.25, w_pad=1.0)
    fig.savefig(OUT / "fig2_stats.pdf")
    fig.savefig(OUT / "fig2_stats.png", dpi=300)
    print("wrote", OUT / "fig2_stats.pdf")


if __name__ == "__main__":
    main()
