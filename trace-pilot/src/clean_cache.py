"""Drop bad rows from the distill cache.

Bad rows are:
  - empty/whitespace trace
  - cache_keys explicitly listed in --bad-keys (output of audit_traces.py)

Output: rewrites the cache JSONL with bad rows removed. The original is renamed
to cache.jsonl.before_clean.<ts>. After this, re-run make_sft_data.py to regenerate
the dropped rows (it will resume from the cleaned cache).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bad-keys", type=Path, default=None)
    ap.add_argument("--drop-truncated-heuristic", action="store_true",
                    help="Also drop rows that look truncated (mid-sentence at end, length>=2000)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.cache).read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} rows from {args.cache}")

    bad_keys = set()
    if args.bad_keys and Path(args.bad_keys).exists():
        for line in Path(args.bad_keys).read_text().splitlines():
            if line.strip():
                bad_keys.add(line.strip())
    print(f"explicit bad keys: {len(bad_keys)}")

    kept, dropped = [], []
    import re
    for r in rows:
        t = (r.get("trace") or "").strip()
        reason = None
        if not t:
            reason = "empty"
        elif r["cache_key"] in bad_keys:
            reason = "bad_key"
        elif args.drop_truncated_heuristic and len(t) >= 2000 and not re.search(r"[.!?\"\']\s*$", t):
            reason = "heuristic_truncated"
        if reason:
            dropped.append((r["cache_key"], reason))
        else:
            kept.append(r)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = args.cache.with_suffix(args.cache.suffix + f".before_clean.{ts}")
    shutil.copy2(args.cache, backup)
    args.cache.write_text("\n".join(json.dumps(r) for r in kept) + "\n")

    print(f"dropped {len(dropped)} rows  kept {len(kept)} rows")
    from collections import Counter
    by_reason = Counter(r[1] for r in dropped)
    for reason, n in by_reason.most_common():
        print(f"  {reason}: {n}")
    print(f"backup: {backup}")
    print(f"rewrote: {args.cache}")
    print("\nNext step: re-run make_sft_data.py to regenerate the dropped rows.")


if __name__ == "__main__":
    main()
