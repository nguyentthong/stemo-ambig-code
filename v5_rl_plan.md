# v5: RL on STEMO-Ambig

## Goal
Test whether RL on top of v4 SFT can push strict-K accuracy beyond ~10%.
Two acceptable outcomes for the paper:
1. **RL also fails to crack strict-K** → bulletproofs "ambig is hard" thesis.
2. **RL exhibits clear reward hacking** (gains enum/interp-coverage at cost of pi_overall, or false-positive enumerates unambig questions) → documented as failure mode, supports same thesis.

Either way, paper claim "even targeted training under realistic budgets fails to truly resolve referential ambiguity" gets stronger.

## Algorithm: GRPO
- No critic — group-relative advantages from N rollouts per prompt.
- Supported by HuggingFace TRL (`GRPOTrainer`).
- Compatible with LoRA via PEFT.

## Reward design
Primary reward (continuous, per item):
```
r = (n_correct / K)               if model enumerates and is ambig-aware
  - 0.5                           if model enumerates an UNAMBIG control item (false-positive)
  - 0.1 * (max(0, len-2048)/2048) light length penalty to discourage verbose hack
```
Judge: `gemini-3-flash-preview` (same as v4 STaR filter, identical JSON schema).

**Cache judge calls** keyed on `(item_id, hash(response_text))` — same response across rollouts/epochs returns same reward without re-paying.

## Anti-reward-hacking safeguards
- **Unambig control set** in training pool: ~500 unambig questions sampled from VideoMME / MVBench. Force model to NOT enumerate them. Penalty above.
- **KL penalty** to v4 reference model (β ≈ 0.04) — prevents drift far from v4.
- **Length cap** (max_new_tokens = 4096; lower than v4's 16384 to discourage padding-for-coverage).
- **Watch metrics weekly**: enum_unambig_rate, interp_overcount (model enumerates K' > gold K).

## Data
- Ambig: same 2179 STEMO-Ambig source items (`star_input.jsonl`).
- Unambig control: 500 random VideoMME items, gold answer used to score.
- Mix 80/20 (ambig/unambig) per batch.

## Init
- Start from **v4 LoRA adapter** (CoT preserved, 20/80 mix).
- Keep LoRA r=128, target same modules.
- LR 1e-6 (10× lower than SFT — RL needs gentle).

## Infra
- `trl.GRPOTrainer` with custom `reward_funcs` list:
  - `reward_ambig_strict_k(samples, items) -> List[float]`
  - `reward_unambig_no_enum(samples, items) -> List[float]`
  - `reward_length_penalty(samples) -> List[float]`
- N rollouts = 8 per prompt, temperature 0.8.
- 6 GPUs, DeepSpeed Zero-3, bf16.
- LoRA training only (frozen base).

## Phases / TODO

### Phase 1 — Infrastructure (1.5 d)
- [ ] **1.1** Write `trace-pilot/src/rl_reward.py` — judge-based reward functions (ambig strict-K, unambig anti-enum, length).
- [ ] **1.2** Implement persistent judge cache (`gemini_judge_cache.jsonl`) to avoid re-judging identical (item, response) pairs across epochs.
- [ ] **1.3** Prepare unambig control pool: sample 500 VideoMME items, write to `data_v0/stemo_ambig_rl/unambig_control.jsonl`.
- [ ] **1.4** Write `trace-pilot/src/prep_rl_input.py` — merge ambig + unambig, tag each row, save `rl_train.jsonl`.
- [ ] **1.5** Write `trace-pilot/scripts/launch_rl.sh` — `accelerate launch` wrapper, identical structure to `launch_sft.sh`.

### Phase 2 — Smoke test on qwen35 (1 d)
- [ ] **2.1** GRPO smoke run on 50 items × 2 epochs to validate reward signal, cache hits, KL drift.
- [ ] **2.2** Inspect rollouts manually: ensure reward variance is non-zero, judge cache is hit, KL stays bounded.
- [ ] **2.3** Fix any reward-collapse (all rollouts get same reward → no signal).

### Phase 3 — Full v5 training, qwen35 (2 d)
- [ ] **3.1** Launch full GRPO on 2179 ambig + 500 unambig, 3 epochs, N=8 rollouts.
- [ ] **3.2** Monitor enum_unambig_rate, interp_overcount, KL during training — bail if hacking is obvious by step 200.
- [ ] **3.3** Save adapter `checkpoints/qwen35_stemo_ambig_lora_v5`.

### Phase 4 — Eval (0.5 d)
- [ ] **4.1** Run STEMO-Ambig eval (sharded across 8 GPUs).
- [ ] **4.2** Run VideoMME, MVBench regression evals.
- [ ] **4.3** **New eval**: false-positive enumeration rate on unambig probe set (200 items). Defines `enum_unambig_rate` as the "hacking diagnostic".
- [ ] **4.4** Per-K breakdown of strict-K acc + interp_coverage.

### Phase 5 — Generalize to qwen36, qwen3vl32b (2 d)
- [ ] **5.1** Run v5 on qwen36 (init from qwen36's v4 adapter once that finishes ~tomorrow).
- [ ] **5.2** Run v5 on qwen3vl32b (init from its v4 adapter).
- [ ] **5.3** Cross-model summary: did RL help, hurt, or hack? Per-model breakdown.

### Phase 6 — Paper writeup additions (1 d)
- [ ] **6.1** Add Sec 5.4 "Does RL solve referential ambiguity?" with results table.
- [ ] **6.2** Add Sec 5.5 "Reward hacking analysis" with examples (model overcounts interpretations, enumerates unambig).
- [ ] **6.3** Cross-reference [[feedback-stemo-ambig-paper-framing]]: "Even RL on top of strong SFT does not close the gap at K≥4."

## Estimated compute
- Per model: ~6 GPUs × 2 days RL + 0.5 day eval = ~12 GPU-days.
- 3 models: 36 GPU-days.
- Gemini judge: ~87k calls per RL run (8 rollouts × 2179 × 5 epochs / cache_hit ≈ 0.4). Cost: ~$50–$100 per run × 3 models = ~$300.

## Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| Reward hacking (overenum) | **High** | Unambig control + length penalty + KL; document if observed |
| Gemini judge instability | Med | Cache + retry; spot-check 50 judgments manually |
| KL divergence too high | Med | β=0.04 start, tune up if model drifts on VideoMME |
| Training instability (advantage variance) | Med | Standard GRPO group norm; bail if grad norm > 5 |
| Compute overrun | Low | Phase 2 smoke gate before full runs |

## Success criteria
- **v5 STEMO-Ambig strict-K**: report whatever it is, honestly.
- **v5 enum_unambig_rate**: if < 5 %, "RL did not hack"; if > 20 %, "hacking confirmed".
- **v5 VideoMME / MVBench**: must not drop more than 2pp from v4 (sanity).
- **Per-K curve**: report K=2 vs K≥4 strict-K separately. If gap widens vs v4, that's the headline.

## Out of scope
- DPO from STaR best/rejected pairs (alternative; skip unless GRPO fails entirely).
- Reward model distillation (skip; online Gemini judge with cache is fine for our scale).
- Multi-turn / tool use (irrelevant to question).
- Full RLHF with human pairs (no labelers available before ARR).
