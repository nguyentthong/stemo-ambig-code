# Gemini figure round-trip

One prompt file per figure, paired with the script it replaces:

| Prompt | Paste Gemini's code into |
|---|---|
| 01_scatter.prompt.md   | paper/figures/make_fig_scatter.py |
| 02_subsets.prompt.md   | paper/figures/make_fig_subsets.py |
| 03_perk.prompt.md      | paper/figures/make_fig_perk.py |
| 04_responses.prompt.md | paper/figures/make_fig_responses.py (new file) |

Workflow per figure:
1. Open the prompt file, copy everything below the COPY marker into Gemini.
2. Gemini returns one python code block: save it as the paired file,
   replacing the previous content entirely.
3. Tell Claude to render: every number is checked against the DATA block
   sources, the output is vision-judged, then the paper is recompiled.

To add new results (from the training box): give them to Claude first, so
the DATA blocks are updated from run artifacts, then re-copy the prompt.
bash paper/figures/render_all.sh renders every figure at once.
