"""Flask UI for inspecting Qwen3.5-27B traces + Gemini-judge outputs.

  python trace-pilot/src/inspect_ui.py            # port 5002 by default
  http://localhost:5002

Filters: k_group, truncated, enumerated, single_commit, judged-only.
Shows question, K gold interpretations + gold answers, model thinking trace,
model final answer, judge verdict per interpretation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_traces.jsonl"
DEFAULT_JUDGMENTS = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_judgments.jsonl"
DEFAULT_GOLD = REPO_ROOT / "data_v0" / "stemo_ambig_candidates" / "all_questions.json"


def load_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def load_gold(p):
    data = json.loads(Path(p).read_text())
    return {q["id"]: q for q in data["questions"]}


def build_app(traces_path, judgments_path, gold_path):
    traces = load_jsonl(traces_path)
    judgments = {j["id"]: j for j in load_jsonl(judgments_path)}
    gold = load_gold(gold_path)

    # Join: one record per trace, with judge + gold attached.
    items = []
    for t in traces:
        j = judgments.get(t["id"])
        g = gold.get(t["id"])
        items.append({"trace": t, "judge": j, "gold": g})

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/api/items")
    def api_items():
        k = request.args.get("k_group", "")
        trunc = request.args.get("truncated", "")
        enum = request.args.get("enumerated", "")
        commit = request.args.get("single_commit", "")
        out = []
        for i, it in enumerate(items):
            j = it["judge"] or {}
            if k and it["trace"].get("k_group") != k:
                continue
            if trunc == "yes" and not j.get("truncated"):
                continue
            if trunc == "no" and j.get("truncated"):
                continue
            if enum == "yes" and not j.get("enumerated"):
                continue
            if enum == "no" and j.get("enumerated"):
                continue
            if commit == "yes" and not j.get("single_commit"):
                continue
            if commit == "no" and j.get("single_commit"):
                continue
            t = it["trace"]
            out.append({
                "idx": i,
                "id": t["id"],
                "k_group": t.get("k_group"),
                "question": t.get("question"),
                "final_answer": (t.get("final_answer") or "").strip()[:80],
                "truncated": bool(j.get("truncated")),
                "enumerated": bool(j.get("enumerated")),
                "single_commit": bool(j.get("single_commit")),
                "n_matched": j.get("n_matched"),
                "n_addressed": j.get("n_addressed"),
                "n_total": j.get("n_interpretations_total"),
                "thinking_chars": t.get("thinking_char_count"),
            })
        return jsonify(out)

    @app.route("/api/item/<int:idx>")
    def api_item(idx):
        if idx < 0 or idx >= len(items):
            abort(404)
        it = items[idx]
        t, j, g = it["trace"], it["judge"], it["gold"]
        gold_interps = []
        if g:
            for gi in g["interpretations"]:
                gold_interps.append({
                    "interp_id": gi["interpretation_id"],
                    "referent": gi["referent_description"],
                    "disambig_q": gi["disambiguated_question"],
                    "gold": gi["predicted_answer"],
                })
        per_interp = j.get("per_interp") if j else []
        return jsonify({
            "id": t["id"],
            "video_id": t.get("video_id"),
            "question": t.get("question"),
            "category": t.get("category"),
            "subcategory": t.get("subcategory"),
            "k_group": t.get("k_group"),
            "thinking_trace": t.get("thinking_trace", ""),
            "final_answer": t.get("final_answer", ""),
            "thinking_chars": t.get("thinking_char_count"),
            "elapsed_sec": t.get("elapsed_sec"),
            "gold_interpretations": gold_interps,
            "answer_changes": (g or {}).get("answer_changes_across_interpretations"),
            "judge": {
                "truncated": (j or {}).get("truncated"),
                "enumerated": (j or {}).get("enumerated"),
                "single_commit": (j or {}).get("single_commit"),
                "per_interp": per_interp,
                "notes": (j or {}).get("judge_notes"),
            } if j else None,
        })

    @app.route("/api/k_groups")
    def api_k_groups():
        ks = sorted({t["trace"].get("k_group") for t in items if t["trace"].get("k_group")},
                    key=lambda s: int(s[1:]))
        return jsonify(ks)

    @app.route("/api/stats")
    def api_stats():
        n = len(items)
        with_judge = sum(1 for it in items if it["judge"])
        trunc = sum(1 for it in items if (it["judge"] or {}).get("truncated"))
        enum = sum(1 for it in items if (it["judge"] or {}).get("enumerated"))
        commit = sum(1 for it in items if (it["judge"] or {}).get("single_commit"))
        return jsonify({"n_total": n, "n_judged": with_judge,
                        "n_truncated": trunc, "n_enumerated": enum,
                        "n_single_commit": commit})

    return app


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>STEMO-Ambig trace inspector</title>
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; height: 100vh; display: grid; grid-template-columns: 380px 1fr; }
#left { border-right: 1px solid #ddd; overflow-y: auto; background: #fafafa; }
#right { padding: 16px 24px; overflow-y: auto; }
#filters { padding: 10px; border-bottom: 1px solid #ddd; background: #fff; position: sticky; top: 0; z-index: 5; }
#filters select { margin-right: 4px; font-size: 12px; padding: 2px; }
#stats { font-size: 11px; color: #666; padding: 6px 10px; background: #f0f0f0; }
.row { padding: 8px 10px; border-bottom: 1px solid #eee; cursor: pointer; font-size: 12px; }
.row:hover { background: #eef; }
.row.sel { background: #e0e7ff; }
.row .id { font-family: monospace; color: #555; font-size: 10px; }
.row .ans { color: #333; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-right: 3px; }
.tag.k { background: #ddd; }
.tag.t { background: #fee; color: #c00; }
.tag.e { background: #cfc; color: #060; }
.tag.s { background: #fec; color: #960; }
h2 { margin-top: 0; }
h3 { margin-bottom: 4px; border-bottom: 1px solid #ddd; padding-bottom: 2px; }
.thinking { background: #f7f7f7; padding: 8px; border-radius: 4px; max-height: 60vh; overflow-y: auto; white-space: pre-wrap; font-family: monospace; font-size: 12px; }
.final { background: #f0f7ff; padding: 8px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 4px 0; }
th, td { border: 1px solid #ccc; padding: 4px 6px; font-size: 12px; text-align: left; }
th { background: #f3f3f3; }
.match-true { background: #cfffcf; }
.match-false { background: #ffe0e0; }
.judge-box { padding: 8px; background: #fffbe6; border-radius: 4px; border: 1px solid #fce; }
</style>
</head><body>
<div id="left">
  <div id="filters">
    <div>
      K: <select id="f_k"><option value="">all</option></select>
      Trunc: <select id="f_t"><option value="">any</option><option value="yes">yes</option><option value="no">no</option></select>
    </div>
    <div style="margin-top:4px">
      Enum: <select id="f_e"><option value="">any</option><option value="yes">yes</option><option value="no">no</option></select>
      Commit: <select id="f_s"><option value="">any</option><option value="yes">yes</option><option value="no">no</option></select>
    </div>
  </div>
  <div id="stats"></div>
  <div id="list"></div>
</div>
<div id="right"><h2>Select a trace</h2><p>Pick an item from the list to inspect.</p></div>
<script>
let curIdx = null;
async function loadKs() {
  const ks = await fetch('/api/k_groups').then(r => r.json());
  const sel = document.getElementById('f_k');
  for (const k of ks) { sel.innerHTML += `<option value="${k}">${k}</option>`; }
}
async function loadStats() {
  const s = await fetch('/api/stats').then(r => r.json());
  document.getElementById('stats').textContent =
    `${s.n_total} traces · ${s.n_judged} judged · ${s.n_truncated} trunc · ${s.n_enumerated} enum · ${s.n_single_commit} single-commit`;
}
async function loadList() {
  const k = document.getElementById('f_k').value;
  const t = document.getElementById('f_t').value;
  const e = document.getElementById('f_e').value;
  const s = document.getElementById('f_s').value;
  const u = `/api/items?k_group=${k}&truncated=${t}&enumerated=${e}&single_commit=${s}`;
  const items = await fetch(u).then(r => r.json());
  const list = document.getElementById('list');
  list.innerHTML = '';
  for (const it of items) {
    const tags = [];
    if (it.k_group) tags.push(`<span class="tag k">${it.k_group}</span>`);
    if (it.truncated) tags.push(`<span class="tag t">trunc</span>`);
    if (it.enumerated) tags.push(`<span class="tag e">enum</span>`);
    if (it.single_commit) tags.push(`<span class="tag s">commit</span>`);
    const ans = it.final_answer || '(empty)';
    list.innerHTML += `<div class="row" data-idx="${it.idx}">
      <div class="id">${it.id}</div>
      <div>${tags.join('')} ${it.question}</div>
      <div class="ans">→ <i>${ans}</i></div>
    </div>`;
  }
  document.querySelectorAll('.row').forEach(r => {
    r.onclick = () => selectItem(parseInt(r.dataset.idx));
  });
}
async function selectItem(idx) {
  curIdx = idx;
  document.querySelectorAll('.row').forEach(r => {
    r.classList.toggle('sel', parseInt(r.dataset.idx) === idx);
  });
  const d = await fetch(`/api/item/${idx}`).then(r => r.json());
  const r = document.getElementById('right');

  let goldTable = '<table><tr><th>interp_id</th><th>referent</th><th>disambiguated question</th><th>gold</th></tr>';
  for (const g of d.gold_interpretations) {
    goldTable += `<tr><td><code>${g.interp_id}</code></td><td>${g.referent}</td><td>${g.disambig_q}</td><td><b>${g.gold}</b></td></tr>`;
  }
  goldTable += '</table>';

  let judgeBlock = '';
  if (d.judge) {
    let perTbl = '<table><tr><th>interp_id</th><th>gold</th><th>addressed</th><th>model_answer</th><th>match</th></tr>';
    for (const p of (d.judge.per_interp || [])) {
      const cls = p.match ? 'match-true' : 'match-false';
      perTbl += `<tr class="${cls}"><td><code>${p.interp_id}</code></td><td>${p.gold}</td><td>${p.addressed}</td><td>${p.model_answer}</td><td>${p.match}</td></tr>`;
    }
    perTbl += '</table>';
    judgeBlock = `<h3>Judge verdict</h3>
      <div class="judge-box">
        truncated=<b>${d.judge.truncated}</b> · enumerated=<b>${d.judge.enumerated}</b> · single_commit=<b>${d.judge.single_commit}</b>
        ${d.judge.notes ? `<div><i>notes:</i> ${d.judge.notes}</div>` : ''}
      </div>
      ${perTbl}`;
  } else {
    judgeBlock = '<h3>Judge verdict</h3><p><i>not yet judged</i></p>';
  }

  r.innerHTML = `
    <h2>${d.id}</h2>
    <p><b>K=${d.k_group}</b> · video=<code>${d.video_id}</code> · subcat=${d.subcategory} · thinking=${d.thinking_chars}ch · elapsed=${d.elapsed_sec}s</p>
    <h3>Surface question</h3>
    <p>${d.question}</p>
    <h3>Gold interpretations (answer_changes=${d.answer_changes})</h3>
    ${goldTable}
    <h3>Model final answer</h3>
    <div class="final">${d.final_answer ? d.final_answer : '(EMPTY — likely truncated inside &lt;think&gt;)'}</div>
    ${judgeBlock}
    <h3>Thinking trace (${d.thinking_chars} chars)</h3>
    <div class="thinking">${d.thinking_trace ? d.thinking_trace.replace(/</g,'&lt;') : '(none)'}</div>
  `;
}
['f_k','f_t','f_e','f_s'].forEach(id => document.getElementById(id).onchange = loadList);
loadKs().then(loadList);
loadStats();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    ap.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = build_app(args.traces, args.judgments, args.gold)
    print(f"Inspector at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
