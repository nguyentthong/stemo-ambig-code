You are writing a matplotlib figure for an ACL paper (appendix figure,
one column). Line plot: ReQueST score (%) by K bin, one line per
closed-weight model. K is the number of readings a question admits; the
score pools enumeration and clarification credit.

## Findings the figure must not violate

These are claims the paper text and tables commit to. The figure must stay
consistent with every one of them. If a requested change would break one,
keep the data and flag the conflict instead.

1. Caption (main.tex, fig:perk): the figure shows "the closed-weight
   models" only. Exactly the four models above appear as lines.
2. Main text S4.2: "clarification keeps ... the GPT-5 score at 31.7%"
   at K >= 7. GPT-5's last point is exactly 31.7 and must be the highest
   line in every bin.
3. Main text S4.2: "Difficulty grows with K". Each model's K=2 value is
   its maximum. Plot all four bins for every model; no bin may be dropped
   even though the Gemini curves are non-monotonic in the middle bins.
4. Caption: "Clarification keeps the Gemini scores above their strict-K in
   the tail." Both Gemini lines must remain visibly above the GPT-4o floor
   at K=7+ (7.6 and 4.2 against 0.0), so the y-axis must not truncate or
   compress the 0-10 range away.
5. Table 1 cross-check (holds for the data above, must keep holding): the
   bin-size-weighted mean of each line reproduces the model's overall
   ReQueST score in Table 1 within rounding: GPT-5 41.9, GPT-4o 0.5,
   Gemini-3-Flash 20.2, Gemini-3.5-Flash 18.4.
6. The y-axis is the ReQueST score, not strict-K. Label it
   "ReQueST score (%)". The strict-K-below-2% claim in S4.2 refers to a
   metric this figure does not show; never relabel or mix the two.

## Style (house style, match Figure 3)

- rcParams exactly: Times-like serif stack ["Times New Roman",
  "Nimbus Roman", "STIXGeneral", "DejaVu Serif"], mathtext.fontset stix,
  font.size 7, axes.labelsize 7, tick labelsize 6.5, legend.fontsize 6,
  axes.linewidth 0.6, pdf.fonttype 42, ps.fonttype 42.
- figsize (3.03, 2.0). Spines top/right removed. Light y grid #dddddd,
  lw 0.5, behind the data.
- Okabe-Ito family palette shared with Figure 3: GPT family #D55E00,
  Gemini family #009E73. Siblings within a family: newer model solid line
  with filled marker (GPT-5 circle, Gemini-3-Flash square), older model
  dashed line with open (white-faced) marker (GPT-4o circle,
  Gemini-3.5-Flash square). Marker size ~3.2, line width ~1.1.
- x tick labels "$K$=2", "$K$=3", "$K$=4–6", "$K$=7+".
- Frameless legend, upper right, all four models.
- Per-point value labels at 4.5 pt in the line's color, one decimal,
  offset above the point (below for Gemini-3.5-Flash so the two Gemini
  labels never collide). There is no companion per-K table, so the labels
  carry the record.
- Footnote from the DATA block: figtext, 5 pt, italic, #999999, centered
  below the axes.
- The script computes nothing from external files: the DATA block is
  inlined verbatim.

Output the COMPLETE runnable script in one fenced python code block,
nothing else. matplotlib only, stdlib pathlib for paths. The script saves
fig_perk.pdf and fig_perk.png (dpi=300) next to itself and prints the pdf
path.
