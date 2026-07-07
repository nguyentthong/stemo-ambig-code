# Source video frames for Figure 1 and Figure 3

These are the individual video-frame panels that make up the teaser
(`../figure1.png`, **Figure 1** in the paper) and, by extension, the protocol
overview (`../fig_overview.pdf`, **Figure 3**), which reuses the row-A frames.

Each file is a rectangular crop of `figure1.png` (2000 × 2114). The in-frame
badges are part of the panel: the top-left black chip is the video timestamp,
and the top-right green/red chip is the gold yes/no answer for that moment.

| File | Row / example | Timestamp |
| --- | --- | --- |
| `fig1-3_rowA_1_ball-door_0m12.png` | A — repeated event (ball / doors) | 0:12 |
| `fig1-3_rowA_2_ball-door_0m58.png` | A — repeated event | 0:58 |
| `fig1-3_rowA_3_ball-door_3m26.png` | A — repeated event | 3:26 |
| `fig1_rowB_1_bag_3m31.png` | B — multiple entities (bag) | 3:31 |
| `fig1_rowB_2_cakestand_5m22.png` | B — multiple entities (cake stand) | 5:22 |
| `fig1_rowC_1_cards_0m08.png` | C — same entity, multiple moments (cards) | 0:08 |
| `fig1_rowC_2_cards_0m28.png` | C — same entity, multiple moments | 0:28 |
| `fig1_rowC_3_cards_1m00.png` | C — same entity, multiple moments | 1:00 |

The three `rowA` frames (prefixed `fig1-3_`) are the frames `make_fig_overview.py`
crops from `figure1.png` to build Figure 3. Rows B and C appear only in Figure 1.
