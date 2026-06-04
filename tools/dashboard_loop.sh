#!/usr/bin/env bash
# Refresh STATUS.md every 30 min, sync to the narrow-scope dashboard repo, push to GitHub.
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
DASH=$REPO/dashboard_repo
cd $REPO

INTERVAL=${INTERVAL:-1800}

while true; do
  # 1. Regenerate STATUS.md
  /home/thong/anaconda3/bin/python tools/dashboard.py >> tmp/dashboard.log 2>&1

  # 2. Sync to dashboard_repo (only the few small artifacts)
  cp STATUS.md $DASH/STATUS.md 2>/dev/null
  cp paper_draft.md $DASH/paper_draft.md 2>/dev/null
  cp figures/*.png $DASH/figures/ 2>/dev/null
  cp analysis/*.json $DASH/analysis/ 2>/dev/null

  # 3. Commit + push if anything changed
  cd $DASH
  git add -A
  if git diff --cached --quiet 2>/dev/null; then
    : # no changes
  else
    git commit -m "dashboard: $(date -u +%Y-%m-%dT%H:%MZ)" >> $REPO/tmp/dashboard.log 2>&1
    git push origin main >> $REPO/tmp/dashboard.log 2>&1 || true
  fi
  cd $REPO

  sleep $INTERVAL
done
