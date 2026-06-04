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
        {"base", "v3", "v4_sample", "v4_filter", "v4_train", "v4_strict", "videomme", "mvbench", "fft"}),  # v5 out-of-scope
    ("internvl8b",  "InternVL3.5-8B",  ["internvl8b_base"],
        {"base", "v3", "v4_sample", "v4_filter", "v4_train", "v4_strict", "videomme", "mvbench", "fft"}),  # v5 out-of-scope
    ("internvl38b", "InternVL3.5-38B", ["internvl38b_base"],
        {"base", "v3", "v4_sample", "v4_filter", "v4_train", "v4_strict", "videomme", "mvbench", "v5"}),   # fft out-of-scope
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
    ("internvl8b_fft_v4",     "FFT (internvl8b)"),
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
    n_base = sum(1 for tag, _, btags, _scope in MODEL_PIPELINES
                 if find_metrics_any(btags + [f"{tag}_v4_base"])[0])
    n_v3 = sum(1 for tag, _, _bt, _scope in MODEL_PIPELINES
               if find_metrics_any(V3_TAGS.get(tag, []))[0])
    n_v4 = sum(1 for tag, _, _bt, _scope in MODEL_PIPELINES if get_metrics(f"{tag}_v4"))
    n_v5_targets = sum(1 for _t, _l, _bt, sc in MODEL_PIPELINES if "v5" in sc)
    n_v5 = sum(1 for tag, _, _bt, _scope in MODEL_PIPELINES if get_metrics(f"{tag}_v5"))
    n_fft_targets = sum(1 for _t, _l, _bt, sc in MODEL_PIPELINES if "fft" in sc)
    n_fft = sum(1 for tag, _, _bt, _scope in MODEL_PIPELINES if get_metrics(f"{tag}_fft_v4"))
    n_blackbox = sum(1 for tag, _ in BLACK_BOX if get_metrics(tag))
    n_ablations = sum(1 for tag, _ in ABLATIONS if get_metrics(tag))

    # Total target counts only in-scope cells (v5 only for some models, FFT only for some)
    total_target = (len(MODEL_PIPELINES)  # base
                    + len(MODEL_PIPELINES)  # v3
                    + len(MODEL_PIPELINES)  # v4
                    + n_v5_targets         # v5 (only some models)
                    + n_fft_targets        # FFT (only some)
                    + len(BLACK_BOX)
                    + len(ABLATIONS))
    completed = n_base + n_v3 + n_v4 + n_v5 + n_fft + n_blackbox + n_ablations
    lines.append("## Overall progress")
    lines.append("")
    lines.append("_Total experiments we are running for the paper, across all six open-weight models, three closed-source APIs, and ten ablation studies._")
    lines.append("")
    lines.append(f"{bar(completed, total_target, width=40)}  ({completed}/{total_target} cells)")
    lines.append("")
    # Per-category bars
    lines.append("| Category | Progress |")
    lines.append("|---|---|")
    lines.append(f"| Base evals       | {bar(n_base, len(MODEL_PIPELINES), 20)}  {n_base}/{len(MODEL_PIPELINES)} |")
    lines.append(f"| v3 SFT           | {bar(n_v3, len(MODEL_PIPELINES), 20)}  {n_v3}/{len(MODEL_PIPELINES)} |")
    lines.append(f"| v4 SFT (LoRA)    | {bar(n_v4, len(MODEL_PIPELINES), 20)}  {n_v4}/{len(MODEL_PIPELINES)} |")
    lines.append(f"| v4 FFT (full)    | {bar(n_fft, n_fft_targets, 20)}  {n_fft}/{n_fft_targets} _(scoped: 2 small models)_ |")
    lines.append(f"| v5 RL            | {bar(n_v5, n_v5_targets, 20)}  {n_v5}/{n_v5_targets} _(scoped: 4 mid-large models)_ |")
    lines.append(f"| Black-box (base) | {bar(n_blackbox, len(BLACK_BOX), 20)}  {n_blackbox}/{len(BLACK_BOX)} |")
    lines.append(f"| Ablations        | {bar(n_ablations, len(ABLATIONS), 20)}  {n_ablations}/{len(ABLATIONS)} |")
    lines.append("")
    lines.append("**Target venue:** ARR August 2026 (deadline 3 Aug → EMNLP)")
    lines.append("")

    # Active processes summary
    try:
        proc = subprocess.run(
            ["pgrep", "-af",
             "run_qwen_video|run_internvl_video|train_sft|train_rl|judge_stemo|run_mcq|run_stemo|chain_v4|paraphrase_questions|star_filter|maximal_prompting|paraphrase_ablation|prompt_sensitivity|extended_chains|fft_variant|chain_v3"],
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
        if f"eval_runs/{tag}_v4" in proc_blob and "run_qwen_video" in proc_blob:
            running_now.add(("pipeline", tag, "v4_eval"))
        if "train_rl_grpo" in proc_blob and tag in proc_blob:
            running_now.add(("pipeline", tag, "v5"))
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
                phase_label = {
                    "chain": "v4 chain (wrapping)",
                    "sample": "v4 sampling (STaR rollouts)",
                    "filter": "v4 Gemini judge filter",
                    "train": "v4 LoRA training",
                    "v4_eval": "v4 STEMO-Ambig eval",
                    "v5": "v5 GRPO RL training",
                }.get(phase, phase)
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
        videomme_f = REPO / f"eval_runs/{tag}_v4/videomme_metrics.json"
        vm_acc = "—"
        if videomme_f.exists():
            try:
                vm_acc = f"{json.loads(videomme_f.read_text())['accuracy']:.3f}"
            except Exception:
                pass
        mvb_f = REPO / f"eval_runs/{tag}_v4/mvbench_metrics.json"
        mvb_acc = "—"
        if mvb_f.exists():
            try:
                v = json.loads(mvb_f.read_text())["accuracy"]
                mvb_acc = f"{v:.3f}" if v > 0.001 else "0.000 ⚠️"
            except Exception:
                pass

        # v5 — mark 🚫 if out of scope
        if "v5" not in scope:
            v5_str = "🚫"
        else:
            v5_metrics = get_metrics(f"{tag}_v5")
            v5_adapter = adapter_done(tag, "v5")
            if v5_metrics:
                v5_str = f"{v5_metrics['strict_ambig_aware_accuracy']:.3f}"
            elif v5_adapter:
                v5_str = "✅train"
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
