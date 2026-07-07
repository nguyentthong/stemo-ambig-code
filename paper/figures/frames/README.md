# Original source video frames for Figure 1 and Figure 3

These are the **original, full-resolution video frames** used to build the teaser
(`../figure1.png`, **Figure 1**) and the protocol overview
(`../fig_overview.pdf`, **Figure 3**, which reuses the row-A frames). They carry
no figure annotations (no timestamp chips, gold-answer badges, or question
boxes); those are added when the figures are composed.

Each frame was re-downloaded from its original YouTube upload at full resolution
(the `stemo/videos_h264/` copies are downscaled encodes) and extracted at the
teaser timestamp:

    yt-dlp -f "bv*" --download-sections "*<t-2>-<t+2>" \
        -o clip.%(ext)s "https://www.youtube.com/watch?v=<youtube_id>"
    ffmpeg -y -ss 2 -i clip.mp4 -frames:v 1 -q:v 1 <name>.png

| File | Figure(s) | YouTube ID | Timestamp | Resolution |
| --- | --- | --- | --- | --- |
| `rowA_1_ball-door_0m12.png` | 1, 3 | `RnNdyCbwzn0` | 0:12 | 1920×1080 |
| `rowA_2_ball-door_0m58.png` | 1, 3 | `RnNdyCbwzn0` | 0:58 | 1920×1080 |
| `rowA_3_ball-door_3m26.png` | 1, 3 | `RnNdyCbwzn0` | 3:26 | 1920×1080 |
| `rowB_1_bag_3m31.png` | 1 | `Yha3Zvw_bus` | 3:31 | 1920×1080 |
| `rowB_2_cakestand_5m22.png` | 1 | `Yha3Zvw_bus` | 5:22 | 1920×1080 |
| `rowC_1_cards_0m08.png` | 1 | `nBqyhpG5d98` | 0:08 | 1080×1920 |
| `rowC_2_cards_0m28.png` | 1 | `nBqyhpG5d98` | 0:28 | 1080×1920 |
| `rowC_3_cards_1m00.png` | 1 | `nBqyhpG5d98` | 1:00 | 1080×1920 |

- **Row A** (ball / numbered doors, *"Which Door Will The Ball Hit?"*): the "repeated event" example, shared by Figure 1 and Figure 3.
- **Row B** (bag, then cake stands, *"Is It Cake?"*): the "multiple entities" example (Figure 1 only).
- **Row C** (numbered cards): the "same entity, multiple moments" example (Figure 1 only).

Benchmark video IDs prefix these YouTube IDs in `stemo/videos_h264/`:
`0029_RnNdyCbwzn0`, `0041_Yha3Zvw_bus`, `0077_nBqyhpG5d98`.
