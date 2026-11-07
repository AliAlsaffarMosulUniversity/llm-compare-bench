"""Score stored responses and produce the paired comparison report."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import scoring, stats
from .tasks import Task, load_tasks


def load_responses(path: Path, rep: int = 0) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            if o["rep"] == rep:
                out[o["task_id"]] = o        # last write wins (re-runs)
    return out


def score_all(tasks: list[Task], responses: dict[str, dict]) -> dict[str, dict]:
    scored = {}
    for t in tasks:
        rec = responses.get(t.id)
        text = None if rec is None or rec.get("error") else rec.get("response")
        s, note = scoring.score(t, text)
        scored[t.id] = {"score": s, "note": note, "rec": rec}
    return scored


def _fmt_ci(lo, hi):
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def compare(tasks_path: str, results_dir: str, a_label: str, b_label: str, rep: int = 0,
            alpha: float = 0.05, bootstrap_reps: int = 5000, seed: int = 0) -> dict:
    rdir = Path(results_dir)
    tasks = load_tasks(tasks_path)
    sa = score_all(tasks, load_responses(rdir / f"{a_label}.jsonl", rep))
    sb = score_all(tasks, load_responses(rdir / f"{b_label}.jsonl", rep))
    manifest = json.loads((rdir / "manifest.json").read_text(encoding="utf-8")) \
        if (rdir / "manifest.json").exists() else {}

    dims: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        if t.scorer != "rubric":
            dims[t.dimension].append(t)

    rows, pvals = [], []
    for dim, ts in sorted(dims.items()):
        a = [sa[t.id]["score"] for t in ts]
        b = [sb[t.id]["score"] for t in ts]
        n11, n10, n01, n00 = stats.paired_counts(a, b)
        diff, lo, hi = stats.paired_bootstrap_diff(a, b, bootstrap_reps, alpha, seed)
        p = stats.mcnemar_exact(n10, n01)
        pvals.append(p)
        rows.append({"dimension": dim, "n": len(ts),
                     "acc_a": sum(a) / len(a), "ci_a": stats.wilson_ci(sum(a), len(a)),
                     "acc_b": sum(b) / len(b), "ci_b": stats.wilson_ci(sum(b), len(b)),
                     "diff": diff, "diff_ci": (lo, hi), "a_only": n10, "b_only": n01, "p": p})
    adj = stats.holm(pvals) if pvals else []
    for r, q in zip(rows, adj):
        r["p_holm"] = q

    all_a = [sa[t.id]["score"] for ts in dims.values() for t in ts]
    all_b = [sb[t.id]["score"] for ts in dims.values() for t in ts]
    overall = None
    if all_a:
        n11, n10, n01, n00 = stats.paired_counts(all_a, all_b)
        d, lo, hi = stats.paired_bootstrap_diff(all_a, all_b, bootstrap_reps, alpha, seed)
        overall = {"n": len(all_a), "acc_a": sum(all_a) / len(all_a), "acc_b": sum(all_b) / len(all_b),
                   "diff": d, "diff_ci": (lo, hi), "a_only": n10, "b_only": n01,
                   "p": stats.mcnemar_exact(n10, n01)}

    ops = {}
    for label, sc in ((a_label, sa), (b_label, sb)):
        recs = [v["rec"] for v in sc.values() if v["rec"] and not v["rec"].get("error")]
        lat = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
        tin = sum(r.get("input_tokens") or 0 for r in recs)
        tout = sum(r.get("output_tokens") or 0 for r in recs)
        spec = next((p for p in manifest.get("providers", []) if p.get("label") == label), {})
        price = spec.get("price_per_mtok") or {}
        cost = None
        if price.get("input") is not None and price.get("output") is not None:
            cost = tin / 1e6 * price["input"] + tout / 1e6 * price["output"]
        ops[label] = {"model": spec.get("model"), "answered": len(recs), "errors": len(sc) - len(recs),
                      "latency_median_s": stats.median(lat), "latency_p95_s": stats.percentile(lat, 0.95),
                      "input_tokens": tin, "output_tokens": tout, "cost_usd": cost}
    return {"a": a_label, "b": b_label, "rep": rep, "alpha": alpha, "rows": rows, "overall": overall,
            "operations": ops, "manifest": {k: manifest.get(k) for k in ("started_utc", "tasks_sha256", "n_tasks")}}


def to_markdown(res: dict) -> str:
    a, b = res["a"], res["b"]
    L = [f"# Paired comparison: {a} (A) vs {b} (B)", ""]
    m = res["manifest"]
    if m.get("started_utc"):
        L += [f"Run started (UTC): {m['started_utc']}  ·  tasks sha256: `{(m.get('tasks_sha256') or '')[:12]}`", ""]
    L += [f"| Dimension | n | Acc A % [95% CI] | Acc B % [95% CI] | A−B (pp) [95% CI] | A only | B only | p (McNemar) | p (Holm) |",
          "|---|---:|---|---|---|---:|---:|---:|---:|"]
    for r in res["rows"]:
        L.append(f"| {r['dimension']} | {r['n']} | {100*r['acc_a']:.1f} {_fmt_ci(*r['ci_a'])} | "
                 f"{100*r['acc_b']:.1f} {_fmt_ci(*r['ci_b'])} | {100*r['diff']:+.1f} {_fmt_ci(*r['diff_ci'])} | "
                 f"{r['a_only']} | {r['b_only']} | {r['p']:.4f} | {r['p_holm']:.4f} |")
    o = res["overall"]
    if o:
        L.append(f"| **Overall** | {o['n']} | {100*o['acc_a']:.1f} | {100*o['acc_b']:.1f} | "
                 f"{100*o['diff']:+.1f} {_fmt_ci(*o['diff_ci'])} | {o['a_only']} | {o['b_only']} | {o['p']:.4f} | – |")
    L += ["", "| System | Model | Answered | Errors | Median latency (s) | p95 latency (s) | Tokens in/out | Cost (USD) |",
          "|---|---|---:|---:|---:|---:|---|---:|"]
    for label, v in res["operations"].items():
        cost = "n/a" if v["cost_usd"] is None else f"{v['cost_usd']:.4f}"
        L.append(f"| {label} | {v['model']} | {v['answered']} | {v['errors']} | {v['latency_median_s']:.2f} | "
                 f"{v['latency_p95_s']:.2f} | {v['input_tokens']}/{v['output_tokens']} | {cost} |")
    L += ["", "_Dimensions with few tasks have wide intervals; treat single-dimension p-values as exploratory "
          "(Holm correction is applied across dimensions). Abstention scoring is a keyword heuristic and "
          "should be audited manually._", ""]
    return "\n".join(L)
