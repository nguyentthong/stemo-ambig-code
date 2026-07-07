#!/bin/bash
# render all paper figures; run from anywhere
cd "$(dirname "$0")"
for f in make_fig2_stats.py make_fig_overview.py make_fig_scatter.py \
         make_fig_subsets.py make_fig_perk.py make_fig_responses.py; do
  [ -f "$f" ] || { echo "skip $f (absent)"; continue; }
  python3 "$f" && echo "OK  $f" || echo "FAIL $f"
done
