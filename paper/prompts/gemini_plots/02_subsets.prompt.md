<!--
PAIRED OUTPUT FILE: paper/figures/make_fig_subsets.py
FIGURE NUMBER IN COMPILED PAPER: Figure 6 (appendix F), \label{fig:subsets}
WORKFLOW: copy everything below the marker into Gemini. Save the single
python code block it returns as the paired output file (replace the whole
file). Then tell Claude to render: every number is checked against the DATA
block, the output is vision-judged, and the paper is recompiled.
-->

----------------- COPY BELOW THIS LINE -----------------

You are writing a matplotlib figure for an ACL paper (appendix figure,
one column). Dumbbell plot: one row per model, a square marker for
entity-subset strict-K and a circle for event-subset strict-K, connected
by a gray line. strict-K is the percent of questions where the model
correctly answers all K readings.

## Data (authoritative, do not alter or extend)

```python
# (label, entity strict-K %, event strict-K %); None = per-subset breakdown
# not yet computed under the current scoring.
# Source: analysis/per_subset_metrics.json (pilot runs, entity n=555,
# event n=490).
MODELS = [
    ("GPT-5", None, None),
    ("GPT-4o", 0.0, 0.0),
    ("Gemini-3-Flash", None, None),
    ("Gemini-3.5-Flash", None, None),
    ("Qwen3.5-27B", 6.7, 2.0),
    ("Qwen3.6-27B", 4.5, 2.7),
    ("Qwen3-VL-32B", 4.9, 1.2),
    ("InternVL3.5-8B", None, None),
    ("InternVL3.5-38B", None, None),
]
# Pending rows render as a label plus a small italic "pending" note so the
# full roster stays visible. Never invent values for them.
# Humans are NOT a row: the human study measures detection (91.8% entity
# vs 76.2% event), a different metric that cannot share this axis.
```

## Findings the figure must not violate

These are claims the paper text and tables commit to. The figure must stay
consistent with every one of them. If a requested change would break one,
keep the data and flag the conflict instead.

1. Caption (main.tex, fig:subsets): "Readings recalled across episodes are
   harder to find than readings scanned within one scene." Every measured
   row must show event <= entity. The layout must make the direction
   legible: entity and event markers clearly distinguishable and the
   connecting line visible even for small gaps.
2. Main text S4.3 says the tested models "fall by a factor of three to
   four from entity to event." The current data satisfies this for
   Qwen3.5-27B (3.4x) and Qwen3-VL-32B (4.1x) but NOT for Qwen3.6-27B
   (1.7x). The figure plots the DATA block verbatim regardless: if the
   re-scored runs keep the 1.7x, the sentence in S4.3 must change, not
   the plot.
3. Table 1 cross-check, currently FAILING: the 555:490-weighted mean of
   each row should reproduce Table 1's overall strict-K, but the pilot
   values imply 4.5 for Qwen3.5-27B against 2.2 in Table 1 (similarly 3.6
   vs 2.6 and 3.2 vs 2.4). The pilot numbers predate the repaired turn-2
   extractor that Table 1 uses. Until the re-scored per-subset runs land,
   the figure must carry the footnote "pilot scoring; re-scored runs
   pending" (figtext, 5 pt, italic, #999999) so it does not silently
   contradict Table 1.
4. GPT-4o's row is 0.0/0.0: nudge the two coincident markers apart
   vertically so both stay visible, and suppress value labels below 0.2.
5. Roster and row order: measured rows sorted by entity-event gap, pending
   rows grouped after them. All nine models from Table 1 appear; no row
   may be dropped, or the roster would imply the pending runs do not
   exist.

## Style (house style)

- rcParams exactly: Times-like serif stack ["Times New Roman",
  "Nimbus Roman", "STIXGeneral", "DejaVu Serif"], mathtext.fontset stix,
  font.size 7, axes.labelsize 7, tick labelsize 6.5, legend.fontsize 6,
  axes.linewidth 0.6, pdf.fonttype 42, ps.fonttype 42.
- figsize (3.03, 0.30 * n_rows + 0.55). Spines top/right removed. Light x
  grid #dddddd, lw 0.5, behind the data.
- Entity square #2a78d6 (s=22), event circle #1baf7a (s=24), connector
  #bbbbbb lw 1.1. Value labels 6 pt, #333333, one decimal, entity to the
  right of its marker, event to the left.
- x label "strict-$K$ (%)", xlim (-0.6, 8.2).
- Frameless legend (Entity, Event), lower right, 5.5 pt.
- The script computes nothing from external files: the DATA block is
  inlined verbatim.

Output the COMPLETE runnable script in one fenced python code block,
nothing else. matplotlib only, stdlib pathlib for paths. The script saves
fig_subsets.pdf and fig_subsets.png (dpi=300) next to itself and prints
the pdf path.
