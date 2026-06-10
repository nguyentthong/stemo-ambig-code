"""STEMO-Ambig experiment dashboard.

Scans the repo for experiment progress + generates STATUS.md.
Cron'd every 30 min and pushed to GitHub.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Model tags. The `scope` set lists which phases we plan to run.
# Phases not in scope are marked 🚫 in tables and excluded from progress denominators.
MODEL_PIPELINES = [
    # tag, label, base_tags_for_eval_runs, in_scope_phases
    ("qwen35",      "Qwen3.5-27B",     ["qwen35_base"],
        {"base", "v3", "v4_sample", "v4_filter", "v4_train", "v4_strict", "videomme", "mvbench", "v5"}),
    ("qwen36",      "Qwen3.6-27B",     ["qwen36_base"],
        {"base", "v3", "v4_sample", "v4_filter", "v4_train", "v4_strict", "videomme", "mvbench", "v5"}),
    ("qwen3vl32b",  "Qwen3-VL-32B",    ["base", "qwen3vl32b_base"],
        {"base", "v3", "v4_sample", "v4_filter", "v4_train", "v4_strict", "videomme", "mvbench", "v5"}),
    ("qwen36_9b",   "Qwen3.6-9B",      ["qwen36_9b_base"],
        # v5 added to scope (training is live as of 2026-06-06)
        {"base", "v3", "v4_sample", "v4_filter", "v4_train", "v4_strict", "videomme", "mvbench", "fft", "v5"}),
    # InternVL3.5-8B/38B fully out of scope: custom modeling code is incompatible
    # with transformers >= 4.49 (all_tied_weights_keys AttributeError). Paper's
    # open-weight coverage is the Qwen family; cross-family contrast comes from
    # the three closed APIs. Rows kept for visibility with empty scope.
    ("internvl8b",  "InternVL3.5-8B",  ["internvl8b_base"], set()),
    ("internvl38b", "InternVL3.5-38B", ["internvl38b_base"], set()),
]
V3_TAGS = {  # legacy v3 tag map
    "qwen35": ["qwen35_v3"],
    "qwen36": ["qwen36_v3"],
    "qwen3vl32b": ["sft_v3_final", "qwen3vl32b_v3"],
    "qwen36_9b": ["qwen36_9b_v3"],
    "internvl8b": ["internvl8b_v3"],
    "internvl38b": ["internvl38b_v3"],
}
BLACK_BOX = [
    ("gpt4o_base",          "GPT-4o (bare)"),
    ("gpt4o_fewshot",       "GPT-4o (few-shot)"),
    ("gpt4o_maximal",       "GPT-4o (maximal)"),
    ("gemini3flash_base",   "Gemini-3-Flash (bare)"),
    ("gemini3flash_fewshot","Gemini-3-Flash (few-shot)"),
    ("gemini3flash_maximal","Gemini-3-Flash (maximal)"),
    ("gemini35flash_base",  "Gemini-3.5-Flash (bare)"),
    ("gemini35flash_fewshot","Gemini-3.5-Flash (few-shot)"),
    ("gemini35flash_maximal","Gemini-3.5-Flash (maximal)"),
]
ABLATIONS = [
    ("qwen3vl32b_maxprompt",  "Maximal-prompting (qwen3vl32b)"),
    ("qwen35_v4_noparap",     "Paraphrase-ablation (qwen35)"),
    ("qwen35_prompt_neutral", "Prompt-sensitivity (qwen35 neutral)"),
    ("qwen35_prompt_fewshot", "Prompt-sensitivity (qwen35 fewshot)"),
    ("qwen35_prompt_explicit","Prompt-sensitivity (qwen35 explicit)"),
    ("qwen36_prompt_neutral", "Prompt-sensitivity (qwen36 neutral)"),
    ("qwen36_prompt_fewshot", "Prompt-sensitivity (qwen36 fewshot)"),
    ("qwen36_prompt_explicit","Prompt-sensitivity (qwen36 explicit)"),
    ("qwen36_9b_fft_v4",      "FFT (qwen36_9b)"),
    # internvl8b_fft_v4 dropped — InternVL out of scope (transformers compat blocker)
]


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    except Exception:
        return 0


def get_metrics(tag: str) -> dict | None:
    f = REPO / f"eval_runs/{tag}/stemo_ambig_metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())["overall"]
    except Exception:
        return None


def find_metrics_any(tags: list[str]) -> tuple[dict, str] | tuple[None, None]:
    for t in tags:
        m = get_metrics(t)
        if m:
            return m, t
    return None, None


def sampling_state(tag: str) -> dict:
    """Return dict with sampled (unique), total, last_write_min, rate_per_hour."""
    shards = REPO / f"data_v0/stemo_ambig_sft_{tag}_v4/star_shards"
    total = 2179
    out = {"done": 0, "total": total, "age_min": None, "rate_per_hour": None}
    if not shards.exists():
        return out
    ids = set()
    last_write = 0.0
    earliest_write = float("inf")
    for f in shards.glob("preds_*.jsonl"):
        try:
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
            last_write = max(last_write, f.stat().st_mtime)
            earliest_write = min(earliest_write, f.stat().st_mtime)
        except Exception:
            pass
    out["done"] = len(ids)
    if last_write:
        out["age_min"] = (time.time() - last_write) / 60
    if earliest_write != float("inf") and out["done"] > 0:
        elapsed_h = (last_write - earliest_write) / 3600
        if elapsed_h > 0.05:
            out["rate_per_hour"] = out["done"] / elapsed_h
    return out


def filter_state(tag: str) -> tuple[bool, int]:
    """Return (file_exists_with_data, n_kept). Empty file counts as not_done."""
    f = REPO / f"data_v0/stemo_ambig_sft_{tag}_v4/star_kept.jsonl"
    if not f.exists() or f.stat().st_size < 50:
        return False, 0
    n = _count_jsonl_lines(f)
    return n > 0, n


def adapter_done(tag: str, suffix: str = "v4") -> bool:
    return (REPO / f"checkpoints/{tag}_stemo_ambig_lora_{suffix}/adapter_model.safetensors").exists()


def training_progress(tag: str, suffix: str = "v4") -> dict:
    """Read latest training step from chain log or trainer_state.json."""
    out = {"step": None, "total": None, "loss": None}
    ts_path = REPO / f"checkpoints/{tag}_stemo_ambig_lora_{suffix}/trainer_state.json"
    if ts_path.exists():
        try:
            ts = json.loads(ts_path.read_text())
            out["step"] = ts.get("global_step")
            out["total"] = ts.get("max_steps")
            log = ts.get("log_history", [])
            if log:
                out["loss"] = log[-1].get("loss")
        except Exception:
            pass
    return out


# Expected output files for each master-queue phase. When a phase emits its
# "done" marker but its expected output is missing, that's a silent failure.
PHASE_EXPECTED_OUTPUTS = {
    # phase log token : list of (description, glob-relative-to-REPO)
    "maximal_prompting_ablation": [
        ("qwen3vl32b_maxprompt metrics", "eval_runs/qwen3vl32b_maxprompt/stemo_ambig_metrics.json"),
    ],
    "paraphrase_ablation": [
        ("qwen35_v4_noparap metrics", "eval_runs/qwen35_v4_noparap/stemo_ambig_metrics.json"),
        ("qwen35_v4_noparap adapter", "checkpoints/qwen35_stemo_ambig_lora_v4_noparap/adapter_model.safetensors"),
    ],
    "prompt_sensitivity_ablation": [
        # Scope reduced 2026-06-04: only qwen35_prompt_neutral kept; the other 5
        # configs were dropped to unblock Phase C. Closed-API prompt-sensitivity
        # ablation (3 APIs × 3 prompts, parallel) covers the rest of §6.2.
        ("qwen35 prompt-sensitivity neutral (kept config)", "eval_runs/qwen35_prompt_neutral/stemo_ambig_metrics.json"),
    ],
    # Phase A used to wrap qwen3vl32b chain. After the June 3 kill + re-queue,
    # the actual qwen3vl32b run lives inside Phase C (extended_chains). So Phase A
    # is no longer expected to produce qwen3vl32b outputs; those expectations
    # were moved to "extended chains" below.
    "extended chains": [
        ("qwen3vl32b v4 metrics", "eval_runs/qwen3vl32b_v4/stemo_ambig_metrics.json"),
        ("qwen3vl32b v4 adapter", "checkpoints/qwen3vl32b_stemo_ambig_lora_v4/adapter_model.safetensors"),
        ("qwen36_9b v4 metrics", "eval_runs/qwen36_9b_v4/stemo_ambig_metrics.json"),
        ("internvl8b v4 metrics", "eval_runs/internvl8b_v4/stemo_ambig_metrics.json"),
        ("internvl38b v4 metrics", "eval_runs/internvl38b_v4/stemo_ambig_metrics.json"),
        ("qwen36 MVBench rerun (real)", "eval_runs/qwen36_v4/mvbench_metrics.json"),
        ("qwen36_9b v3 metrics", "eval_runs/qwen36_9b_v3/stemo_ambig_metrics.json"),
        ("internvl8b v3 metrics", "eval_runs/internvl8b_v3/stemo_ambig_metrics.json"),
        ("internvl38b v3 metrics", "eval_runs/internvl38b_v3/stemo_ambig_metrics.json"),
        ("qwen36_9b base metrics", "eval_runs/qwen36_9b_base/stemo_ambig_metrics.json"),
        ("internvl8b base metrics", "eval_runs/internvl8b_base/stemo_ambig_metrics.json"),
        ("internvl38b base metrics", "eval_runs/internvl38b_base/stemo_ambig_metrics.json"),
        ("qwen36_9b FFT metrics", "eval_runs/qwen36_9b_fft_v4/stemo_ambig_metrics.json"),
        ("internvl8b FFT metrics", "eval_runs/internvl8b_fft_v4/stemo_ambig_metrics.json"),
    ],
    "v5 RL qwen3.5": [
        ("v5 RL qwen3.5 metrics", "eval_runs/qwen35_v5/stemo_ambig_metrics.json"),
    ],
    "v5 RL remaining": [
        ("v5 RL qwen3.6 metrics", "eval_runs/qwen36_v5/stemo_ambig_metrics.json"),
        ("v5 RL qwen3-vl-32b metrics", "eval_runs/qwen3vl32b_v5/stemo_ambig_metrics.json"),
        ("v5 RL internvl38b metrics", "eval_runs/internvl38b_v5/stemo_ambig_metrics.json"),
    ],
}


def silent_failures() -> list[dict]:
    """Scan master_queue.log for 'done' markers. For each completed phase, check
    that its expected outputs exist. Also flag phases that have been 'start' but
    not 'done' for >2x the expected wall-clock (default 4h). Return list of
    missing-output OR stalled-phase records."""
    log = REPO / "tmp/master_queue.log"
    if not log.exists():
        return []
    try:
        lines = log.read_text().splitlines()
    except Exception:
        return []
    completed_phases = set()
    started_phases: dict[str, float] = {}  # phase -> start timestamp (epoch s)
    # If the whole queue has finished, no phase can still be "running", so timing
    # alarms must not fire. (The per-phase 'done' lines say e.g. "[master] phase D
    # done" without the descriptive phase name, so name-matching alone misses them.)
    # NOTE: this only suppresses timing alarms — it does NOT mark phases complete
    # for the missing-output check, because work superseded into other queues
    # (completion queue, offline chains) is scheduled, not silently failed.
    queue_finished = any("FULLY ALL DONE" in l for l in lines)
    # FIRST PASS: identify all completed phases.
    for l in lines:
        if "done" not in l.lower():
            continue
        for phase_key in PHASE_EXPECTED_OUTPUTS.keys():
            if phase_key in l:
                completed_phases.add(phase_key)
    # SECOND PASS: only record start times for phases that have NOT yet completed.
    for l in lines:
        if "start" not in l.lower():
            continue
        for phase_key in PHASE_EXPECTED_OUTPUTS.keys():
            if phase_key not in l or phase_key in completed_phases:
                continue
            ts_match = re.search(r"(\w{3} \w{3} +\d+ [\d:]+ \w+ \d{4})", l)
            if ts_match:
                try:
                    started_phases[phase_key] = time.mktime(
                        time.strptime(ts_match.group(1), "%a %b %d %H:%M:%S %Z %Y"))
                except Exception:
                    pass
    failures = []
    for phase_key in completed_phases:
        for desc, rel_path in PHASE_EXPECTED_OUTPUTS[phase_key]:
            target = REPO / rel_path
            if not target.exists() or target.stat().st_size < 50:
                failures.append({
                    "phase": phase_key,
                    "kind": "missing_output",
                    "missing_output": desc,
                    "expected_path": rel_path,
                })
    # Phases running far beyond reasonable wall-clock: flag as potentially stalled.
    PHASE_BUDGET_H = {
        "maximal_prompting_ablation": 6,
        "paraphrase_ablation": 12,
        "prompt_sensitivity_ablation": 24,
        "phase A": 24,
        "extended chains": 96,
        "v5 RL qwen3.5": 48,
        "v5 RL remaining": 120,
    }
    now = time.time()
    if queue_finished:
        started_phases.clear()  # queue exited — nothing is still running
    for phase_key, started in started_phases.items():
        budget_h = PHASE_BUDGET_H.get(phase_key, 12)
        elapsed_h = (now - started) / 3600
        if elapsed_h > budget_h:
            failures.append({
                "phase": phase_key,
                "kind": "slow",
                "missing_output": f"phase elapsed {elapsed_h:.1f}h > budget {budget_h}h",
                "expected_path": "(timing alarm)",
            })
    return failures


def master_queue_state() -> dict:
    log = REPO / "tmp/master_queue.log"
    out = {"alive": False, "phase": "unknown", "last_lines": [], "started": None}
    if log.exists():
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        out["last_lines"] = lines[-6:]
        for l in reversed(lines):
            if l.startswith("[master]"):
                m = re.match(r"\[master\]\s+(.*)$", l)
                if m:
                    out["phase"] = m.group(1)[:100]
                    break
        for l in lines:
            if " start" in l.lower():
                m = re.search(r"(\w+ \w+ +\d+ [\d:]+ \w+ \d+)", l)
                if m:
                    out["started"] = m.group(1)
                    break
    try:
        result = subprocess.run(["pgrep", "-f", "master_queue.sh"],
                                capture_output=True, text=True, timeout=2)
        out["alive"] = bool(result.stdout.strip())
    except Exception:
        pass
    return out


def fmt_pct(done: int, total: int) -> str:
    if total <= 0:
        return "—"
    pct = 100 * done / total
    return f"{pct:5.1f}%"


def bar(done: int, total: int, width: int = 20) -> str:
    """Unicode progress bar like tqdm. Uses full + partial block chars."""
    if total <= 0:
        return "[" + " " * width + "]   —"
    pct = min(1.0, max(0.0, done / total))
    filled = pct * width
    full = int(filled)
    frac = filled - full
    partials = " ▏▎▍▌▋▊▉█"
    partial_char = partials[int(frac * 8)]
    bar_str = "█" * full + (partial_char if full < width else "") + " " * max(0, width - full - 1)
    return f"`[{bar_str}]` {100*pct:5.1f}%"


def bar_with_eta(done: int, total: int, age_min: float | None,
                  rate_per_hour: float | None, width: int = 20) -> str:
    """tqdm-style bar with embedded ETA."""
    base = bar(done, total, width)
    if done >= total:
        return f"{base} ✅"
    if rate_per_hour and rate_per_hour > 1 and age_min is not None and age_min < 60:
        eta_h = (total - done) / rate_per_hour
        if eta_h < 1:
            eta = f"{int(eta_h*60)}m"
        elif eta_h < 48:
            eta = f"{eta_h:.1f}h"
        else:
            eta = f"{eta_h/24:.1f}d"
        return f"{base} 🟢 ETA {eta}"
    if age_min is not None and age_min > 60:
        return f"{base} ⚠️ stalled {age_min/60:.1f}h"
    return base


def fmt_eta_hours(remaining: int, rate_per_hour: float | None) -> str:
    if not rate_per_hour or rate_per_hour < 1:
        return "—"
    h = remaining / rate_per_hour
    if h < 1:
        return f"~{int(h*60)}m"
    if h < 48:
        return f"~{h:.1f}h"
    return f"~{h/24:.1f}d"


def fmt_metrics(m: dict | None) -> str:
    if not m:
        return "—"
    return (f"enum={m['enumeration_rate']:.3f} "
            f"commit={m['single_commit_rate']:.3f} "
            f"strict={m['strict_ambig_aware_accuracy']:.3f}")


def build_dashboard() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("# STEMO-Ambig Experiment Dashboard")
    lines.append(f"_Last updated: {now}_  (auto-refresh every 30 min)")
    lines.append("")

    # Overall progress count
    # Denominators are scope-aware: a model only counts toward a category if that
    # phase is in its scope set (InternVL rows have empty scope → excluded everywhere).
    n_base_targets = sum(1 for _t, _l, _bt, sc in MODEL_PIPELINES if "base" in sc)
    n_base = sum(1 for tag, _, btags, sc in MODEL_PIPELINES
                 if "base" in sc and find_metrics_any(btags + [f"{tag}_v4_base"])[0])
    n_v3_targets = sum(1 for _t, _l, _bt, sc in MODEL_PIPELINES if "v3" in sc)
    n_v3 = sum(1 for tag, _, _bt, sc in MODEL_PIPELINES
               if "v3" in sc and find_metrics_any(V3_TAGS.get(tag, []))[0])
    n_v4_targets = sum(1 for _t, _l, _bt, sc in MODEL_PIPELINES if "v4_strict" in sc)
    n_v4 = sum(1 for tag, _, _bt, sc in MODEL_PIPELINES
               if "v4_strict" in sc and get_metrics(f"{tag}_v4"))
    n_v5_targets = sum(1 for _t, _l, _bt, sc in MODEL_PIPELINES if "v5" in sc)

    def _v5_done(tag):
        # v5 metrics may live under several conventions (online vs offline, 9B alias)
        if get_metrics(f"{tag}_v5") or get_metrics(f"{tag}_v5_offline"):
            return True
        candidates = [f"eval_runs/{tag}_iaa_v5_offline/iaa_metrics.json"]
        if tag == "qwen36_9b":
            candidates.append("eval_runs/qwen35_9b_iaa_v5/iaa_metrics.json")
        return any((REPO / c).exists() for c in candidates)

    n_v5 = sum(1 for tag, _, _bt, sc in MODEL_PIPELINES if "v5" in sc and _v5_done(tag))
    n_fft_targets = sum(1 for _t, _l, _bt, sc in MODEL_PIPELINES if "fft" in sc)
    n_fft = sum(1 for tag, _, _bt, _scope in MODEL_PIPELINES if get_metrics(f"{tag}_fft_v4"))
    n_blackbox = sum(1 for tag, _ in BLACK_BOX if get_metrics(tag))
    n_ablations = sum(1 for tag, _ in ABLATIONS if get_metrics(tag))
    # IAA runs: 3 closed-API + 6 open-weight (qwen3vl32b/qwen36/qwen35 each base+v4)
    iaa_run_tags = [
        "gpt4o_iaa", "gemini3flash_iaa", "gemini35flash_iaa",
        "qwen35_iaa_base", "qwen35_iaa_v4",
        "qwen36_iaa_base", "qwen36_iaa_v4",
        "qwen3vl32b_iaa_base", "qwen3vl32b_iaa_v4",
    ]
    n_iaa = sum(1 for t in iaa_run_tags if (REPO / f"eval_runs/{t}/iaa_metrics.json").exists())
    n_iaa_targets = len(iaa_run_tags)

    # Total target counts only in-scope cells (InternVL rows have empty scope)
    total_target = (n_base_targets
                    + n_v3_targets
                    + n_v4_targets
                    + n_v5_targets
                    + n_fft_targets
                    + len(BLACK_BOX)
                    + len(ABLATIONS)
                    + n_iaa_targets)
    completed = n_base + n_v3 + n_v4 + n_v5 + n_fft + n_blackbox + n_ablations + n_iaa
    lines.append("## Overall progress")
    lines.append("")
    lines.append("_Total experiments for the paper: four Qwen open-weight models, three closed-source APIs, ablations, and IAA multi-turn evaluations. InternVL rows are out of scope (transformers compat blocker)._")
    lines.append("")
    lines.append(f"{bar(completed, total_target, width=40)}  ({completed}/{total_target} cells)")
    lines.append("")
    # Per-category bars
    lines.append("| Category | Progress |")
    lines.append("|---|---|")
    lines.append(f"| Base evals       | {bar(n_base, n_base_targets, 20)}  {n_base}/{n_base_targets} |")
    lines.append(f"| v3 SFT           | {bar(n_v3, n_v3_targets, 20)}  {n_v3}/{n_v3_targets} |")
    lines.append(f"| v4 SFT (LoRA)    | {bar(n_v4, n_v4_targets, 20)}  {n_v4}/{n_v4_targets} |")
    lines.append(f"| v4 FFT (full)    | {bar(n_fft, n_fft_targets, 20)}  {n_fft}/{n_fft_targets} _(scoped: qwen36_9b)_ |")
    lines.append(f"| v5 RL            | {bar(n_v5, n_v5_targets, 20)}  {n_v5}/{n_v5_targets} _(scoped: 4 Qwen models)_ |")
    lines.append(f"| Black-box (base) | {bar(n_blackbox, len(BLACK_BOX), 20)}  {n_blackbox}/{len(BLACK_BOX)} |")
    lines.append(f"| Ablations        | {bar(n_ablations, len(ABLATIONS), 20)}  {n_ablations}/{len(ABLATIONS)} |")
    lines.append(f"| IAA (headline)   | {bar(n_iaa, n_iaa_targets, 20)}  {n_iaa}/{n_iaa_targets} _(3 closed + 6 open-weight)_ |")
    lines.append("")
    lines.append("**Target venue:** ARR August 2026 (deadline 3 Aug → EMNLP)")
    lines.append("")

    # Active processes summary
    try:
        proc = subprocess.run(
            ["pgrep", "-af",
             "run_qwen_video|run_internvl_video|train_sft|train_rl|judge_stemo|run_mcq|run_stemo|chain_v4|paraphrase_questions|star_filter|maximal_prompting|paraphrase_ablation|prompt_sensitivity|extended_chains|fft_variant|chain_v3|run_iaa_closed|run_iaa_open"],
            capture_output=True, text=True, timeout=4,
        )
        procs = [l for l in proc.stdout.splitlines() if l.strip()]
    except Exception:
        procs = []

    # Detect what specific experiments are running RIGHT NOW
    running_now = set()
    proc_blob = "\n".join(procs)
    # Per-model phases
    for tag, _, _, _ in MODEL_PIPELINES:
        if f"chain_v4.sh {tag}" in proc_blob or f"qwen3vl32b_maxprompt" in proc_blob and tag == "qwen3vl32b":
            if f"chain_v4.sh {tag}" in proc_blob:
                running_now.add(("pipeline", tag, "chain"))
        # Detect sampling: run_qwen_video with this tag in output path
        if f"_sft_{tag}_v4/star_shards" in proc_blob:
            running_now.add(("pipeline", tag, "sample"))
        if f"_sft_{tag}_v4/star_input" in proc_blob and "star_filter" in proc_blob:
            running_now.add(("pipeline", tag, "filter"))
        # Only mark v4 LoRA training if the running config is THE v4 config — not
        # an ablation variant like _noparap or _fft_v4.
        if (f"sft_lora_{tag}_v4.yaml" in proc_blob
                and ("accelerate" in proc_blob or "train_sft" in proc_blob)
                and not adapter_done(tag, "v4")):
            running_now.add(("pipeline", tag, "train"))
        if f"eval_runs/{tag}_v4" in proc_blob and ("run_qwen_video" in proc_blob or "run_internvl_video" in proc_blob):
            # Distinguish base (no adapter) vs LoRA-merged, and which benchmark
            base_run = f"eval_runs/{tag}_v4_base/" in proc_blob
            lora_run = (f"eval_runs/{tag}_v4/" in proc_blob and f"eval_runs/{tag}_v4_base/" not in proc_blob) \
                       or (f"eval_runs/{tag}_v4/" in proc_blob and f"eval_runs/{tag}_v4_base/" in proc_blob)
            if "shards_mvbench" in proc_blob:
                bench = "MVBench"
            elif "shards_videomme" in proc_blob:
                bench = "VideoMME"
            else:
                bench = "STEMO-Ambig"
            if base_run:
                running_now.add(("pipeline", tag, f"v4_eval_base_{bench}"))
            if lora_run:
                running_now.add(("pipeline", tag, f"v4_eval_lora_{bench}"))
        # Match v5 RL precisely: look for tag's RL config path, not just substring.
        # tag="qwen35" was falsely matching qwen35_9b's config (rl_grpo_qwen35_9b.yaml).
        # Some tags use different physical model slugs (qwen36_9b → Qwen3.5-9B because
        # Qwen3.6 has no 9B variant), so accept known aliases.
        rl_config_aliases = {"qwen36_9b": ["qwen36_9b", "qwen35_9b"]}
        candidate_slugs = rl_config_aliases.get(tag, [tag])
        rl_patterns = [rf"rl_grpo_{re.escape(s)}\.yaml" for s in candidate_slugs]
        if "train_rl_grpo" in proc_blob and any(re.search(p, proc_blob) for p in rl_patterns):
            running_now.add(("pipeline", tag, "v5"))
        # Base eval (sharded run_stemo_ambig_eval / run_qwen_video into eval_runs/{tag}_base/)
        if f"eval_runs/{tag}_base/" in proc_blob and ("run_qwen_video" in proc_blob or "run_internvl_video" in proc_blob):
            running_now.add(("pipeline", tag, "base_eval"))
        # v3 SFT training for this tag
        if f"sft_lora_{tag}_v3" in proc_blob and ("accelerate" in proc_blob or "train_sft" in proc_blob):
            running_now.add(("pipeline", tag, "v3_train"))
        # v3 eval
        if f"eval_runs/{tag}_v3/" in proc_blob and ("run_qwen_video" in proc_blob or "run_internvl_video" in proc_blob):
            running_now.add(("pipeline", tag, "v3_eval"))
    # Ablations
    if "maximal_prompting_ablation" in proc_blob or "qwen3vl32b_maxprompt" in proc_blob:
        running_now.add(("ablation", "qwen3vl32b_maxprompt"))
    if "paraphrase_ablation" in proc_blob or "_v4_noparap" in proc_blob:
        running_now.add(("ablation", "qwen35_v4_noparap"))
    if "prompt_sensitivity_ablation" in proc_blob or "_prompt_" in proc_blob:
        running_now.add(("ablation", "prompt_sensitivity"))
    if "fft_variant" in proc_blob or "_fft_v4" in proc_blob:
        running_now.add(("ablation", "fft"))
    # GPU memory
    try:
        nv = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=4)
        gpu_mem = [int(x.strip()) for x in nv.stdout.splitlines() if x.strip().isdigit()]
    except Exception:
        gpu_mem = []
    lines.append("## Live state")
    lines.append("")
    lines.append("_Snapshot of which GPUs are in use right now and which type of job is running. Useful to confirm experiments are progressing in real time, not stalled._")
    lines.append("")
    if gpu_mem:
        active_gpus = sum(1 for m in gpu_mem if m > 1000)
        max_mem = max(gpu_mem)
        lines.append(f"- **GPUs:** {active_gpus}/{len(gpu_mem)} active (max {max_mem} MiB)")
    # Identify the "main" job
    main_job_keywords = [
        ("v5 RL", "train_rl_grpo"),
        ("Training", "train_sft"),
        ("IAA multi-turn", "run_iaa_closed"),
        ("IAA multi-turn (open)", "run_iaa_open"),
        ("Sampling/inference", "run_qwen_video"),
        ("InternVL inference", "run_internvl_video"),
        ("Judge", "judge_stemo_traces"),
        ("Paraphrase ablation", "paraphrase_ablation"),
        ("Prompt sensitivity", "prompt_sensitivity_ablation"),
        ("Maximal prompting", "maximal_prompting_ablation"),
        ("Star filter", "star_filter"),
        ("MCQ eval", "run_mcq_eval"),
        ("Chain", "chain_v4.sh"),
    ]
    main_label = None
    for label, kw in main_job_keywords:
        if any(kw in p for p in procs):
            main_label = label
            break
    if main_label:
        lines.append(f"- **Active job type:** {main_label} ({sum(1 for p in procs if any(kw in p for _, kw in main_job_keywords))} procs)")
    else:
        lines.append("- **Active job type:** (idle — possibly transitioning between phases)")

    # Specific experiment(s) running now
    if running_now:
        lines.append("- **Currently running experiments:**")
        nice_names = {
            ("ablation", "qwen3vl32b_maxprompt"): "Maximal-prompting ablation (Qwen3-VL-32B)",
            ("ablation", "qwen35_v4_noparap"): "Paraphrase ablation (Qwen3.5-27B no-paraphrase v4)",
            ("ablation", "prompt_sensitivity"): "Prompt-sensitivity ablation",
            ("ablation", "fft"): "Full-parameter fine-tuning ablation",
        }
        items = []
        for key in sorted(running_now):
            if key[0] == "ablation":
                items.append(nice_names.get(key, f"Ablation {key[1]}"))
            elif key[0] == "pipeline":
                _, tag, phase = key
                static_labels = {
                    "chain": "v4 chain (wrapping)",
                    "sample": "v4 sampling (STaR rollouts)",
                    "filter": "v4 Gemini judge filter",
                    "train": "v4 LoRA training",
                    "v4_eval": "v4 STEMO-Ambig eval",
                    "v5": "v5 GRPO RL training",
                    "base_eval": "base STEMO-Ambig eval",
                    "v3_train": "v3 SFT training",
                    "v3_eval": "v3 STEMO-Ambig eval",
                }
                if phase.startswith("v4_eval_base_"):
                    phase_label = f"v4 BASE eval — {phase.split('_')[-1]}"
                elif phase.startswith("v4_eval_lora_"):
                    phase_label = f"v4 LoRA eval — {phase.split('_')[-1]}"
                else:
                    phase_label = static_labels.get(phase, phase)
                items.append(f"{tag} — {phase_label}")
        for it in items:
            lines.append(f"  - 🟢 {it}")
    lines.append("")

    mq = master_queue_state()
    lines.append("## Master Queue")
    lines.append("")
    lines.append("_The orchestrator process that runs experiments sequentially in five phases (A: cross-Qwen v4; B: ablations; C: smaller models + InternVL + FFT; D: v5 RL on Qwen3.5; E: v5 RL on others). Shows the current phase plus recent log lines._")
    lines.append("")
    lines.append(f"- **Alive:** {'✅' if mq['alive'] else '❌'}")
    lines.append(f"- **Current phase:** `{mq['phase']}`")
    if mq["started"]:
        lines.append(f"- **Started:** {mq['started']}")
    lines.append("")
    lines.append("Recent log:")
    lines.append("```")
    for l in mq["last_lines"]:
        lines.append(l[:200])
    lines.append("```")
    lines.append("")

    # Per-model pipeline progress bars
    lines.append("## Per-model pipeline progress")
    lines.append("")
    lines.append("_How far along each model is through its planned pipeline. Each model has a different set of in-scope phases (see Detailed phase status below); the bar shows progress only against the in-scope phases for that model._")
    lines.append("")
    lines.append("| Model | Pipeline progress |")
    lines.append("|---|---|")
    for tag, label, base_tags, scope in MODEL_PIPELINES:
        done = 0; total_scope = len(scope) - (1 if "videomme" in scope and "mvbench" in scope else 0)
        # Actually: count each phase as 1, with videomme/mvbench combined into 1 'mcq' phase
        # For simplicity treat them separately
        if "base" in scope and find_metrics_any(base_tags + [f"{tag}_v4_base"])[0]:
            done += 1
        if "v3" in scope and find_metrics_any(V3_TAGS.get(tag, []))[0]:
            done += 1
        s = sampling_state(tag)
        if "v4_sample" in scope and (s["done"] >= s["total"] or adapter_done(tag, "v4") or get_metrics(f"{tag}_v4")):
            done += 1
        filt_ok, _ = filter_state(tag)
        if "v4_filter" in scope and (filt_ok or adapter_done(tag, "v4") or get_metrics(f"{tag}_v4")):
            done += 1
        if "v4_train" in scope and adapter_done(tag, "v4"):
            done += 1
        if "v4_strict" in scope and get_metrics(f"{tag}_v4"):
            done += 1
        videomme_f = REPO / f"eval_runs/{tag}_v4/videomme_metrics.json"
        mvb_f = REPO / f"eval_runs/{tag}_v4/mvbench_metrics.json"
        def _acc(f):
            if not f.exists(): return None
            try: return json.loads(f.read_text())["accuracy"]
            except Exception: return None
        if "videomme" in scope and _acc(videomme_f) and _acc(videomme_f) > 0.05:
            done += 1
        if "mvbench" in scope and _acc(mvb_f) and _acc(mvb_f) > 0.05:
            done += 1
        if "v5" in scope and (get_metrics(f"{tag}_v5") or adapter_done(tag, "v5")):
            done += 1
        if "fft" in scope and get_metrics(f"{tag}_fft_v4"):
            done += 1
        total_phases = len(scope)
        lines.append(f"| **{label}** | {bar(done, total_phases, 24)}  ({done}/{total_phases} phases) |")
    lines.append("")

    # Detailed pipeline table
    lines.append("### Detailed phase status")
    lines.append("")
    lines.append("_Numeric cells show strict-K accuracy (0.000–1.000) or VideoMME/MVBench accuracy. 🚫 = deliberately out of scope (not planned). — = not yet started. ✅ = completed (non-numeric milestone). ⚠️ = anomaly (e.g., known broken eval, rerun queued)._")
    lines.append("")
    lines.append("| Model | base strict | v3 strict | v4 sample | v4 filter | v4 adapter | v4 strict | v4 VideoMME | v4 MVBench | v4 FFT strict | v5 RL strict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, label, base_tags, scope in MODEL_PIPELINES:
        # Base
        bm, btag = find_metrics_any(base_tags + [f"{tag}_v4_base"])
        base_str = f"{bm['strict_ambig_aware_accuracy']:.3f}" if bm else "—"

        # v3
        vm, vtag = find_metrics_any(V3_TAGS.get(tag, []))
        v3_str = f"{vm['strict_ambig_aware_accuracy']:.3f}" if vm else "—"

        # v4 sampling
        s = sampling_state(tag)
        # If downstream phases (filter/train/eval) are done, treat sample as done
        # even if some shards didn't reach the full 2179 (small leftover items).
        adapter_present = adapter_done(tag, "v4")
        eval_present = bool(get_metrics(f"{tag}_v4"))
        downstream_done = adapter_present or eval_present
        running_sample = ("pipeline", tag, "sample") in running_now
        if s["done"] == 0 and not running_sample:
            sample_str = "—"
        elif s["done"] >= s["total"] or downstream_done:
            sample_str = "✅"
        else:
            # tqdm-style bar with ETA
            sample_str = bar_with_eta(s["done"], s["total"], s["age_min"], s["rate_per_hour"], width=12)
            if running_sample:
                sample_str += " 🟢"

        # v4 filter (also treat as ✅ if downstream done with smaller kept count)
        filt_ok, n_kept = filter_state(tag)
        if n_kept > 0:
            filt_str = f"{n_kept}"
        elif downstream_done:
            filt_str = "✅"
        else:
            filt_str = "—"

        # v4 train
        running_train = ("pipeline", tag, "train") in running_now
        train_str = "✅" if adapter_present else ("🟢" if running_train else "—")
        # v4 filter — also mark 🟢 if filter is actively running
        if ("pipeline", tag, "filter") in running_now and filt_str == "—":
            filt_str = "🟢"

        # v4 strict + MCQ
        m = get_metrics(f"{tag}_v4")
        v4_strict = f"{m['strict_ambig_aware_accuracy']:.3f}" if m else "—"
        def _bench_cell(tag, bench_dir, bench_name, metrics_file):
            """Return cell value: metric if file exists; else 🟢 progress if shards actively writing; else —."""
            f = REPO / f"eval_runs/{tag}_v4/{metrics_file}"
            if f.exists():
                try:
                    v = json.loads(f.read_text())["accuracy"]
                    return f"{v:.3f}" if v > 0.001 else "0.000 ⚠️"
                except Exception:
                    return "—"
            # Check if BOTH lora and base shards are actively being written
            shard_dir = REPO / f"eval_runs/{tag}_v4/{bench_dir}"
            base_shard_dir = REPO / f"eval_runs/{tag}_v4_base/{bench_dir}"
            now = time.time()
            done_lines, total_lines, recent = 0, 0, False
            for sd in (shard_dir, base_shard_dir):
                if not sd.exists():
                    continue
                for p in sd.glob("preds_*.jsonl"):
                    try:
                        st = p.stat()
                        if now - st.st_mtime < 1800:
                            recent = True
                        with open(p) as fh:
                            done_lines += sum(1 for _ in fh)
                    except Exception:
                        pass
            running_eval = any(k[0] == "pipeline" and k[1] == tag and isinstance(k[2], str)
                               and k[2].startswith("v4_eval_") and k[2].endswith(bench_name)
                               for k in running_now)
            if running_eval or recent:
                return f"🟢 {done_lines}"
            return "—"

        vm_acc = _bench_cell(tag, "shards_videomme", "VideoMME", "videomme_metrics.json")
        mvb_acc = _bench_cell(tag, "shards_mvbench", "MVBench", "mvbench_metrics.json")

        # v5 — mark 🚫 if out of scope. v5 may be reached via two paths:
        #  - "online" GRPO (current 9B): adapter at checkpoints/{tag}_stemo_ambig_lora_v5/
        #  - "offline" STaR-style (current 27B): adapter at .../lora_v5_offline/
        # The cell shows the most-advanced state reachable from either path.
        if "v5" not in scope:
            v5_str = "🚫"
        else:
            # Metrics for v5 may be stored under multiple eval-tag conventions
            v5_iaa_path = REPO / f"eval_runs/{tag}_iaa_v5_offline/iaa_metrics.json"
            v5_iaa = None
            if v5_iaa_path.exists():
                try:
                    m = json.loads(v5_iaa_path.read_text())
                    # IAA file uses 'iaa' as primary metric; map to strict_K column for table
                    v5_iaa = {"strict_ambig_aware_accuracy": m.get("iaa", 0)}
                except Exception:
                    pass
            v5_metrics = v5_iaa or get_metrics(f"{tag}_v5") or get_metrics(f"{tag}_v5_offline")
            v5_adapter_online = (REPO / f"checkpoints/{tag}_stemo_ambig_lora_v5" /
                                 "adapter_model.safetensors").exists()
            v5_adapter_offline = (REPO / f"checkpoints/{tag}_stemo_ambig_lora_v5_offline" /
                                  "adapter_model.safetensors").exists()
            # qwen36_9b alias: physical adapter lives under qwen35_9b
            if tag == "qwen36_9b":
                v5_adapter_online = v5_adapter_online or (
                    REPO / "checkpoints/qwen35_9b_stemo_ambig_lora_v5" /
                    "adapter_model.safetensors").exists()
            v5_adapter = v5_adapter_online or v5_adapter_offline
            # Offline chain stage tracking for in-flight 27B runs
            chain_dir = REPO / f"data_v0/stemo_ambig_v5_offline_{tag}"
            stage = None
            if chain_dir.exists() and not v5_adapter_offline:
                if (chain_dir / "star_kept_v5.jsonl").exists():
                    stage = "🟢 SFT"
                elif (chain_dir / "judged_rollouts.jsonl").exists():
                    stage = "🟢 select"
                elif (chain_dir / "rollouts.jsonl").exists():
                    stage = "🟢 judge"
                elif (chain_dir / "rollout_shards").exists():
                    tot = sum(sum(1 for _ in open(p)) for p in (chain_dir / "rollout_shards").glob("preds_*.jsonl"))
                    # The chain samples from star_input.jsonl which is the SFT training
                    # pool (~2179 items), not the 1056-item benchmark.
                    star_input = REPO / f"data_v0/stemo_ambig_sft_{tag}_v4/star_input.jsonl"
                    target = sum(1 for _ in open(star_input)) if star_input.exists() else 2179
                    stage = f"🟢 sample {tot}/{target} ({tot*100//max(target,1)}%)"
            if v5_metrics:
                v5_str = f"{v5_metrics['strict_ambig_aware_accuracy']:.3f}"
            elif v5_adapter:
                v5_str = "✅train"
            elif stage:
                v5_str = stage
            elif ("pipeline", tag, "v5") in running_now:
                v5_str = "🟢"
            else:
                v5_str = "—"

        # FFT — mark 🚫 if out of scope
        if "fft" not in scope:
            fft_str = "🚫"
        else:
            fft_metrics = get_metrics(f"{tag}_fft_v4")
            fft_str = f"{fft_metrics['strict_ambig_aware_accuracy']:.3f}" if fft_metrics else "—"

        lines.append(
            f"| {label} | {base_str} | {v3_str} | {sample_str} | {filt_str} | {train_str} | "
            f"{v4_strict} | {vm_acc} | {mvb_acc} | {fft_str} | {v5_str} |"
        )
    lines.append("")
    lines.append("_Legend: 🟢 active write <30min + ETA; ⚠️ stalled >1h or anomalous; ✅ done; — not started_")
    lines.append("_Numbers under base/v3/v4 strict columns are strict-K accuracy._")
    lines.append("")

    # Black-box
    lines.append("## Black-box baselines (STEMO-Ambig)")
    lines.append("")
    lines.append("_How well frontier closed-source APIs handle our benchmark with no fine-tuning. These are reference points for the cross-family failure claim; we cannot fine-tune these models so v3/v4/v5 columns do not apply (🚫)._")
    lines.append("")
    lines.append("| Model | enum | single_commit | strict-K | interp_cov | pi_overall |")
    lines.append("|---|---|---|---|---|---|")
    for tag, label in BLACK_BOX:
        m = get_metrics(tag)
        if m:
            lines.append(
                f"| {label} | {m['enumeration_rate']:.3f} | "
                f"{m['single_commit_rate']:.3f} | "
                f"{m['strict_ambig_aware_accuracy']:.3f} | "
                f"{m.get('interp_coverage', 0):.3f} | "
                f"{m.get('per_interp_accuracy_overall', 0):.3f} |"
            )
        else:
            lines.append(f"| {label} | — | — | — | — | — |")
    lines.append("")

    # Interactive Ambig-Aware Accuracy (IAA) — multi-turn protocol
    lines.append("## IAA — Interactive Ambig-Aware Accuracy")
    lines.append("")
    lines.append("_Multi-turn protocol (PROTOCOL_IAA.md): turn-1 inference, sub-judge classifies, turn-2 disambiguator if clarified. **IAA is the headline benchmark metric**; strict-K and AAR-loose are diagnostic sub-metrics._")
    lines.append("")
    iaa_runs = [
        ("gemini3flash_iaa",    "Gemini-3-flash (base)"),
        ("gemini35flash_iaa",   "Gemini-3.5-flash (base)"),
        ("gpt4o_iaa",           "GPT-4o (base)"),
        ("qwen35_iaa_base",     "Qwen3.5-27B base"),
        ("qwen35_iaa_v4",       "Qwen3.5-27B v4"),
        ("qwen36_iaa_base",     "Qwen3.6-27B base"),
        ("qwen36_iaa_v4",       "Qwen3.6-27B v4"),
        ("qwen3vl32b_iaa_base", "Qwen3-VL-32B base"),
        ("qwen3vl32b_iaa_v4",   "Qwen3-VL-32B v4"),
    ]
    lines.append("| Model | IAA | strict-K | AAR-loose | clar-rate | follow-through | n |")
    lines.append("|---|---|---|---|---|---|---|")
    iaa_running = "run_iaa_closed.py" in proc_blob or "run_iaa_open.py" in proc_blob
    for tag, label in iaa_runs:
        mp = REPO / f"eval_runs/{tag}/iaa_metrics.json"
        pred = REPO / f"eval_runs/{tag}/iaa_predictions.jsonl"
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
                ft = m.get("follow_through_rate")
                ft_str = f"{ft:.3f}" if isinstance(ft, (int, float)) else "—"
                lines.append(
                    f"| {label} | **{m.get('iaa', 0):.3f}** | {m.get('strict_K', 0):.3f} | "
                    f"{m.get('aar_loose', 0):.3f} | {m.get('clarification_rate', 0):.3f} | "
                    f"{ft_str} | {m.get('n', 0)} |"
                )
                continue
            except Exception:
                pass
        # No metrics yet — also check open-weight shard files
        shard_files = list((REPO / f"eval_runs/{tag}/iaa_shards").glob("preds_*.jsonl")) if (REPO / f"eval_runs/{tag}/iaa_shards").exists() else []
        if not pred.exists() and shard_files:
            # Concatenate shard records in-memory for live partial metrics
            try:
                n = 0
                iaa_sum = 0.0
                strict_n = aar_n = clar_n = ft_n = ft_correct = 0
                latest_mtime = 0
                for sf in shard_files:
                    latest_mtime = max(latest_mtime, sf.stat().st_mtime)
                    for line in open(sf):
                        if not line.strip(): continue
                        r = json.loads(line)
                        if r.get("error") or not r.get("score"): continue
                        n += 1
                        iaa_sum += r["score"]["iaa_score"]
                        if r["score"]["strict_K_correct"]: strict_n += 1
                        if r["score"]["aar_loose_correct"]: aar_n += 1
                        if r["classification"]["category"] in {"clarified_scope", "clarified_vague"}:
                            clar_n += 1; ft_n += 1
                            if r["score"]["follow_through_correct"]: ft_correct += 1
                age_min = (time.time() - latest_mtime) / 60 if latest_mtime else 999
                mark = "🟢" if age_min < 30 else "⚠️"
                if n > 0:
                    iaa_v = iaa_sum / n
                    ft_str = f"{ft_correct/ft_n:.3f}" if ft_n > 0 else "—"
                    lines.append(
                        f"| {label} | {mark} **{iaa_v:.3f}** | {strict_n/n:.3f} | "
                        f"{aar_n/n:.3f} | {clar_n/n:.3f} | {ft_str} | {n} (live shards) |"
                    )
                else:
                    lines.append(f"| {label} | {mark} starting (sharded) | — | — | — | — | 0 |")
                continue
            except Exception:
                pass

        # No metrics yet — compute live partial IAA from prediction file
        if pred.exists():
            try:
                n = 0
                iaa_sum = 0.0
                strict_n = 0
                aar_n = 0
                clar_n = 0
                ft_n = 0
                ft_correct = 0
                for line in open(pred):
                    if not line.strip(): continue
                    r = json.loads(line)
                    if r.get("error") or not r.get("score"): continue
                    n += 1
                    iaa_sum += r["score"]["iaa_score"]
                    if r["score"]["strict_K_correct"]: strict_n += 1
                    if r["score"]["aar_loose_correct"]: aar_n += 1
                    if r["classification"]["category"] in {"clarified_scope", "clarified_vague"}:
                        clar_n += 1
                        ft_n += 1
                        if r["score"]["follow_through_correct"]:
                            ft_correct += 1
                age_min = (time.time() - pred.stat().st_mtime) / 60
                mark = "🟢" if age_min < 30 else "⚠️"
                if n > 0:
                    iaa_v = iaa_sum / n
                    ft_str = f"{ft_correct/ft_n:.3f}" if ft_n > 0 else "—"
                    lines.append(
                        f"| {label} | {mark} **{iaa_v:.3f}** | {strict_n/n:.3f} | "
                        f"{aar_n/n:.3f} | {clar_n/n:.3f} | {ft_str} | {n} (live) |"
                    )
                else:
                    lines.append(f"| {label} | {mark} starting | — | — | — | — | 0 |")
                continue
            except Exception as e:
                lines.append(f"| {label} | ⚠️ partial-err | — | — | — | — | — |")
                continue
        if iaa_running and tag in proc_blob:
            lines.append(f"| {label} | 🟢 starting | — | — | — | — | — |")
        else:
            lines.append(f"| {label} | — | — | — | — | — | — |")
    lines.append("")
    lines.append("_IAA = headline. Asks: can the model EITHER enumerate K interpretations OR clarify and then answer correctly when disambiguated? See `docs/PROTOCOL_IAA.md`._")
    lines.append("")

    # Ablations
    lines.append("## Ablations")
    lines.append("")
    lines.append("_Targeted experiments that each test one specific claim in the paper. Maximal-prompting tests \"models can solve it if you prompt them hard enough\"; paraphrase ablation tests whether question-paraphrase augmentation matters; prompt-sensitivity tests whether numbers change a lot under different system prompts; FFT tests whether full fine-tuning beats LoRA at small scale._")
    lines.append("")
    lines.append("| Experiment | strict-K | enum | commit | Status |")
    lines.append("|---|---|---|---|---|")
    def _ablation_running(tag: str) -> bool:
        if tag == "qwen3vl32b_maxprompt":
            return ("ablation", "qwen3vl32b_maxprompt") in running_now
        if tag == "qwen35_v4_noparap":
            return ("ablation", "qwen35_v4_noparap") in running_now
        if "_prompt_" in tag:
            return ("ablation", "prompt_sensitivity") in running_now
        if "_fft_v4" in tag:
            return ("ablation", "fft") in running_now
        return False
    for tag, label in ABLATIONS:
        m = get_metrics(tag)
        if m:
            status = "✅ done"
            lines.append(
                f"| {label} | {m['strict_ambig_aware_accuracy']:.3f} | "
                f"{m['enumeration_rate']:.3f} | "
                f"{m['single_commit_rate']:.3f} | {status} |"
            )
        else:
            status = "🟢 running" if _ablation_running(tag) else "— pending"
            lines.append(f"| {label} | — | — | — | {status} |")
    lines.append("")

    # (v5 training-progress section removed per 2026-06-10 request —
    # training is complete; v5 results live in the phase table.)

    # (v5 offline chain stages are now folded into the v5 column of the Detailed
    # phase status table above.)

    # Recent signals from key logs — only show logs modified in last 6h
    lines.append("## Recent log signals (last 6h)")
    lines.append("")
    lines.append("_Recent phase markers and any error tracebacks from active log files. Helps spot crashes or stalls early — if you see a Traceback here that wasn't there yesterday morning, something went wrong overnight._")
    lines.append("")
    log_paths = [
        ("master_queue", REPO / "tmp/master_queue.log"),
        ("qwen3vl32b chain", REPO / "tmp/v4_qwen3vl32b.log"),
        ("extended chains", REPO / "tmp/extended_chains.log"),
        ("fft variant", REPO / "tmp/fft_variant.log"),
        ("v5 RL qwen35", REPO / "tmp/v5_rl_qwen35.log"),
        ("v5 RL remaining", REPO / "tmp/v5_rl_remaining.log"),
        ("maximal-prompting", REPO / "tmp/maximal_prompting_ablation.log"),
        ("paraphrase ablation", REPO / "tmp/paraphrase_ablation.log"),
        ("prompt sensitivity", REPO / "tmp/prompt_sensitivity_ablation.log"),
        ("qwen36 MVBench rerun", REPO / "tmp/qwen36_mvbench_rerun.log"),
        ("IAA gemini-3-flash", REPO / "tmp/iaa_gemini3flash.log"),
        ("IAA gemini-3.5-flash", REPO / "tmp/iaa_gemini35flash.log"),
        ("IAA gpt-4o", REPO / "tmp/iaa_gpt4o.log"),
        ("IAA qwen35 base", REPO / "tmp/iaa_qwen35_base.log"),
        ("IAA qwen35 v4", REPO / "tmp/iaa_qwen35_v4.log"),
        ("IAA qwen36 base", REPO / "tmp/iaa_qwen36_base.log"),
        ("IAA qwen36 v4", REPO / "tmp/iaa_qwen36_v4.log"),
        ("IAA qwen3vl32b base", REPO / "tmp/iaa_qwen3vl32b_base.log"),
        ("IAA qwen3vl32b v4", REPO / "tmp/iaa_qwen3vl32b_v4.log"),
    ]
    six_hr_ago = time.time() - 6 * 3600
    for name, path in log_paths:
        if not path.exists() or path.stat().st_mtime < six_hr_ago:
            continue
        try:
            tail = path.read_text().splitlines()[-300:]
        except Exception:
            continue
        errs = [l for l in tail if re.search(r"Traceback|ERROR|FAILED|Killed|exitcode: 1", l)]
        phases = [l for l in tail if re.match(r"^\[", l) or "done" in l.lower()][-3:]
        if errs or phases:
            lines.append(f"### {name} (`{path.name}`)")
            for l in phases:
                lines.append(f"- {l[:160]}")
            for l in errs[-2:]:
                lines.append(f"- ⚠️ `{l[:160]}`")
            lines.append("")

    # Silent-failure detector — auto-flag phases that finished without producing
    # the expected output file. Updates bug_fixes.jsonl so the user gets notified.
    sf = silent_failures()
    if sf:
        lines.append("## ⚠️ Silent failures detected")
        lines.append("")
        lines.append("_These master-queue phases emitted a 'done' marker but their expected output files are missing or empty. Investigate before downstream phases consume the missing results._")
        lines.append("")
        lines.append("| Phase | Missing output | Expected path |")
        lines.append("|---|---|---|")
        for f in sf:
            lines.append(f"| {f['phase']} | {f['missing_output']} | `{f['expected_path']}` |")
        lines.append("")
        # Auto-append to bug log only if NOT already recorded (dedup by phase+missing_output).
        bug_log_path = REPO / "tools/bug_fixes.jsonl"
        existing = []
        if bug_log_path.exists():
            try:
                for l in bug_log_path.read_text().splitlines():
                    if l.strip():
                        existing.append(json.loads(l))
            except Exception:
                pass
        existing_keys = set()
        for e in existing:
            t = e.get("title", "")
            if t.startswith("Silent failure:"):
                existing_keys.add(t)
        new_entries = []
        current_failure_keys = set()
        for f in sf:
            title = f"Silent failure: {f['phase']} → missing {f['missing_output']}"
            current_failure_keys.add(title)
            if title not in existing_keys:
                new_entries.append({
                    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "severity": "high",
                    "title": title,
                    "action": ("Auto-detected by dashboard: phase emitted 'done' but expected "
                               "output file is missing/empty, or phase is running far past its "
                               "budget. Needs investigation."),
                    "affected": [f.get("expected_path", "")],
                    "resolved": False,
                })
        # Auto-resolve previously-open auto-detected entries whose failure no longer fires.
        changed = False
        for e in existing:
            t = e.get("title", "")
            if (t.startswith("Silent failure:") and e.get("resolved") is not True
                    and t not in current_failure_keys):
                e["resolved"] = True
                e["action"] = (e.get("action","")
                               + " | Auto-resolved: failure condition no longer detected.")
                changed = True
        if new_entries or changed:
            all_entries = existing + new_entries
            with bug_log_path.open("w") as fh:
                for e in all_entries:
                    fh.write(json.dumps(e) + "\n")

    # Bug-fix log
    bug_log = REPO / "tools/bug_fixes.jsonl"
    if bug_log.exists():
        lines.append("## Bug-fix log")
        lines.append("")
        lines.append("_Issues discovered during automation runs and what was done. Each entry is recorded the moment a patch lands. Severity: low (cosmetic), medium (single experiment cell), high (blocks pipeline)._")
        lines.append("")
        lines.append("| When | Severity | Title | Affected | Status |")
        lines.append("|---|---|---|---|---|")
        try:
            entries = [json.loads(l) for l in bug_log.read_text().splitlines() if l.strip()]
        except Exception:
            entries = []
        # Most recent first
        for e in reversed(entries):
            ts = e.get("ts", "")[:16].replace("T", " ")
            sev = e.get("severity", "?")
            sev_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(sev, "⚪")
            title = e.get("title", "")[:80]
            affected = "; ".join(e.get("affected", []))[:80]
            res = e.get("resolved", False)
            res_str = "✅ fixed" if res is True else "🔄 in-progress" if res == "in-progress" else "❌ open"
            lines.append(f"| {ts} | {sev_icon} {sev} | {title} | {affected} | {res_str} |")
        lines.append("")
        # Latest detail expanded
        if entries:
            latest = entries[-1]
            lines.append("**Latest fix detail:**")
            lines.append(f"- **{latest.get('title','')}**")
            lines.append(f"- Action: {latest.get('action','')}")
            lines.append("")

    # GCS upload progress
    uploads_dir = REPO / "tmp/uploads"
    if uploads_dir.exists():
        lines.append("## GCS upload progress")
        lines.append("")
        lines.append("_Tar+upload of large artifacts to `gs://video_data_bucket-19052026/` for portability. Local tarballs in `tmp/uploads/`, uploaded once tar completes._")
        lines.append("")
        lines.append("| Tarball | Local size | Local last write | GCS uploaded |")
        lines.append("|---|---|---|---|")
        UPLOADS = [
            ("stemo_ambig_adapters.tar.gz", "adapters_runner.log", "LoRA adapters"),
            ("stemo_ambig_eval_runs.tar.gz", "eval_runner.log", "eval predictions+judgments"),
            ("stemo_ambig_sft_data.tar.gz", "sft_data_runner.log", "SFT training data"),
            ("stemo_ambig_sft_videos.tar.gz", "sft_videos_runner.log", "SFT videos (NextQA clips, ~2.9 GB)"),
        ]
        for tarname, logname, desc in UPLOADS:
            local = uploads_dir / tarname
            log = uploads_dir / logname
            # Local
            if local.exists():
                size_mb = local.stat().st_size / (1024 * 1024)
                size_str = f"{size_mb/1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.0f} MB"
                age_min = (time.time() - local.stat().st_mtime) / 60
                if age_min < 1:
                    age_str = f"{int(age_min*60)}s ago 🟢 packing"
                elif age_min < 10:
                    age_str = f"{age_min:.1f}m ago"
                else:
                    age_str = f"{age_min:.0f}m ago ✓ tar done"
            else:
                size_str = "—"; age_str = "not yet"
            # Upload
            upload_status = "—"
            if log.exists():
                try:
                    txt = log.read_text()
                    if "UPLOAD_DONE" in txt:
                        # find size from gsutil ls
                        try:
                            r = subprocess.run(
                                ["gsutil", "ls", "-lh",
                                 f"gs://video_data_bucket-19052026/{tarname}"],
                                capture_output=True, text=True, timeout=8)
                            for line in r.stdout.splitlines():
                                if tarname in line and "MiB" in line or "GiB" in line:
                                    parts = line.split()
                                    upload_status = f"✅ {parts[0]} {parts[1]}"
                                    break
                            else:
                                upload_status = "✅ uploaded"
                        except Exception:
                            upload_status = "✅ uploaded"
                    elif "TAR_DONE" in txt:
                        upload_status = "🟢 uploading"
                    else:
                        upload_status = "🟢 tarring"
                except Exception:
                    pass
            lines.append(f"| `{tarname}` ({desc}) | {size_str} | {age_str} | {upload_status} |")
        lines.append("")
        lines.append(f"_Bootstrap on new server: `gsutil -m cp gs://video_data_bucket-19052026/stemo_ambig_*.tar.gz . && for f in stemo_ambig_*.tar.gz; do tar -xzf $f; done`_")
        lines.append("")

    # Paper status
    lines.append("## Paper")
    pd = REPO / "paper_draft.md"
    if pd.exists():
        n_chars = pd.stat().st_size
        n_sections = sum(1 for l in pd.read_text().splitlines() if l.startswith("## "))
        lines.append(f"- **Draft length:** {n_chars/1024:.1f}k chars across {n_sections} H2 sections")
        lines.append("- **Target venue:** ARR August 2026 (deadline 3 Aug) → EMNLP")
        lines.append("- **Fallback:** CVPR Nov 2026")
        lines.append("")

    # Figures
    figs_dir = REPO / "figures"
    if figs_dir.exists():
        pngs = sorted(figs_dir.glob("*.png"))
        lines.append("## Figures")
        for p in pngs:
            lines.append(f"- ![{p.stem}]({p.relative_to(REPO)})")
        lines.append("")

    return "\n".join(lines)


def main():
    out = REPO / "STATUS.md"
    out.write_text(build_dashboard())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
