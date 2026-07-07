# Original source video frames for Figure 1 and Figure 3

These are the **original video frames** used to build the teaser
(`../figure1.png`, **Figure 1**) and the protocol overview
(`../fig_overview.pdf`, **Figure 3**, which reuses the row-A frames). They are
extracted directly from the benchmark source videos in `stemo/videos_h264/`, so
they carry no figure annotations (no timestamp chips, gold-answer badges, or
question boxes). Those elements are added when the figures are composed.

Extraction (one frame per timestamp):

    ffmpeg -y -ss <seconds> -i stemo/videos_h264/<video_id>.mp4 \
        -frames:v 1 -q:v 1 <name>.png

| File | Figure(s) | Video ID | Timestamp | Resolution |
| --- | --- | --- | --- | --- |
| `rowA_1_ball-door_0m12.png` | 1, 3 | `0029_RnNdyCbwzn0` | 0:12 | 854×480 |
| `rowA_2_ball-door_0m58.png` | 1, 3 | `0029_RnNdyCbwzn0` | 0:58 | 854×480 |
| `rowA_3_ball-door_3m26.png` | 1, 3 | `0029_RnNdyCbwzn0` | 3:26 | 854×480 |
| `rowB_1_bag_3m31.png` | 1 | `0041_Yha3Zvw_bus` | 3:31 | 640×360 |
| `rowB_2_cakestand_5m22.png` | 1 | `0041_Yha3Zvw_bus` | 5:22 | 640×360 |
| `rowC_1_cards_0m08.png` | 1 | `0077_nBqyhpG5d98` | 0:08 | 240×426 |
| `rowC_2_cards_0m28.png` | 1 | `0077_nBqyhpG5d98` | 0:28 | 240×426 |
| `rowC_3_cards_1m00.png` | 1 | `0077_nBqyhpG5d98` | 1:00 | 240×426 |

- **Row A** (ball / numbered doors): the "repeated event" example, shared by Figure 1 and Figure 3.
- **Row B** (bag, then cake stands): the "multiple entities" example (Figure 1 only).
- **Row C** (numbered cards): the "same entity, multiple moments" example (Figure 1 only).

Resolution matches each source video in `stemo/videos_h264/` (the card video is a
low-resolution portrait clip). Re-extract at higher resolution from the original
uploads if a crisper teaser is needed.
