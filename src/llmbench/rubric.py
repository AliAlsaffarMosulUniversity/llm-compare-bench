"""Blind human rating for open-ended (rubric) tasks.

``export_blind`` shuffles the two systems' answers into anonymous slots (X/Y)
per item and writes (1) a CSV for raters and (2) a private key. Raters fill the
``score`` column (e.g., 0/1/2). ``agreement`` computes Fleiss' kappa across
rater files; ``unblind`` merges the mean score back to the systems.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from . import stats
from .report import load_responses
from .tasks import load_tasks


def export_blind(tasks_path: str, results_dir: str, a_label: str, b_label: str, out_csv: str,
                 key_json: str, seed: int = 0, rep: int = 0) -> int:
    tasks = [t for t in load_tasks(tasks_path) if t.scorer == "rubric"]
    ra = load_responses(Path(results_dir) / f"{a_label}.jsonl", rep)
    rb = load_responses(Path(results_dir) / f"{b_label}.jsonl", rep)
    rng = random.Random(seed)
    key, rows = {}, []
    for t in tasks:
        pair = [(a_label, (ra.get(t.id) or {}).get("response") or ""),
                (b_label, (rb.get(t.id) or {}).get("response") or "")]
        rng.shuffle(pair)
        key[t.id] = {"X": pair[0][0], "Y": pair[1][0]}
        rubric = t.meta.get("rubric", "")
        for slot, (_, text) in zip(("X", "Y"), pair):
            rows.append({"item_id": f"{t.id}|{slot}", "prompt": t.prompt, "rubric": rubric,
                         "response": text, "score": ""})
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "prompt", "rubric", "response", "score"])
        w.writeheader()
        w.writerows(rows)
    Path(key_json).write_text(json.dumps(key, indent=2), encoding="utf-8")
    return len(rows)


def _read_scores(path: str) -> dict[str, int]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["item_id"]: int(r["score"]) for r in csv.DictReader(fh) if r["score"].strip() != ""}


def agreement(rater_csvs: list[str]) -> dict:
    sets = [_read_scores(p) for p in rater_csvs]
    common = sorted(set.intersection(*[set(s) for s in sets]))
    if len(common) < 2:
        raise ValueError("raters share fewer than 2 rated items")
    mats = [[s[i] for i in common] for s in sets]
    out = {"items": len(common), "raters": len(sets), "fleiss_kappa": stats.fleiss_kappa(mats)}
    if len(sets) == 2:
        out["cohen_kappa"] = stats.cohen_kappa(*mats)
    return out


def unblind(rater_csvs: list[str], key_json: str) -> dict:
    key = json.loads(Path(key_json).read_text(encoding="utf-8"))
    sets = [_read_scores(p) for p in rater_csvs]
    totals: dict[str, list[float]] = {}
    for item in sorted(set.intersection(*[set(s) for s in sets])):
        task_id, slot = item.rsplit("|", 1)
        mean = sum(s[item] for s in sets) / len(sets)
        totals.setdefault(key[task_id][slot], []).append(mean)
    return {label: {"n": len(v), "mean_score": sum(v) / len(v)} for label, v in totals.items()}
