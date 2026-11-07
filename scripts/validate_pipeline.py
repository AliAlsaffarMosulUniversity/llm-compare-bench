"""Validate the statistical pipeline with synthetic responders.

Experiment A  (end to end): synthetic 300-item task set -> mock providers with a known
              accuracy gap -> runner -> scorer -> report. Repeated over many seeds to
              check that the 95% CI covers the true gap and that H0 is rejected at the
              expected rate. Uses the SAME code path as a real run.
Experiment B  (Monte Carlo grid): power / type-I error of the exact McNemar test as a
              function of task-set size and true accuracy gap.

Everything here is synthetic. It validates the *method*; it says nothing about the
real ChatGPT or Gemini.

Usage:  python scripts/validate_pipeline.py [--out results/example_validation] [--quick]
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmbench import power, runner  # noqa: E402
from llmbench.report import compare  # noqa: E402
from llmbench.tasks import Task, save_tasks  # noqa: E402


def synthetic_tasks(n=300, dims=6):
    return [Task(f"s{i:04d}", f"dim{i % dims}", f"Question {i}\nA) a\nB) b\nC) c\nD) d", "mcq", "ABCD"[i % 4])
            for i in range(n)]


def experiment_a(out: Path, reps: int, n=300, acc_a=0.85, acc_b=0.75, rho=0.5):
    true_gap = acc_a - acc_b
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tasks_path = Path(td) / "tasks.jsonl"
        save_tasks(synthetic_tasks(n), tasks_path)
        for r in range(reps):
            cfg = {"retries": 0, "workers": 1, "providers": [
                {"label": "a", "type": "mock", "seed": 10_000 + r, "shared_difficulty_seed": r, "rho": rho,
                 "accuracy": {"default": acc_a}},
                {"label": "b", "type": "mock", "seed": 20_000 + r, "shared_difficulty_seed": r, "rho": rho,
                 "accuracy": {"default": acc_b}}]}
            rd = Path(td) / f"r{r}"
            runner.run(cfg, str(tasks_path), str(rd), log=lambda *_: None)
            res = compare(str(tasks_path), str(rd), "a", "b", bootstrap_reps=1000, seed=r)
            o = res["overall"]
            rows.append({"rep": r, "diff": o["diff"], "lo": o["diff_ci"][0], "hi": o["diff_ci"][1], "p": o["p"],
                         "covers": int(o["diff_ci"][0] <= true_gap <= o["diff_ci"][1]),
                         "reject": int(o["p"] < 0.05)})
    with open(out / "experiment_a_replicates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summ = {"n_tasks": n, "replicates": reps, "true_gap": true_gap,
            "mean_estimated_gap": sum(r["diff"] for r in rows) / reps,
            "ci_coverage": sum(r["covers"] for r in rows) / reps,
            "power_at_alpha_0.05": sum(r["reject"] for r in rows) / reps}
    (out / "experiment_a_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def experiment_b(out: Path, reps: int):
    ns = [50, 100, 150, 200, 300, 500, 600, 800, 1000]
    deltas = [0.0, 0.03, 0.05, 0.10]
    rows = []
    for acc in (0.80, 0.90):
        for d in deltas:
            for n in ns:
                r = power.simulate_rejection_rate(n, acc, acc - d, rho=0.5, reps=reps, seed=n * 7 + int(d * 1000))
                rows.append({"acc_a": acc, "gap": d, "n": n, "rho": 0.5, "rejection_rate": r.rejection_rate,
                             "mean_discordant": round(r.mean_discordant, 1)})
    for rho in (0.3, 0.7):
        for d in (0.0, 0.05):
            r = power.simulate_rejection_rate(300, 0.80, 0.80 - d, rho=rho, reps=reps, seed=99)
            rows.append({"acc_a": 0.80, "gap": d, "n": 300, "rho": rho, "rejection_rate": r.rejection_rate,
                         "mean_discordant": round(r.mean_discordant, 1)})
    with open(out / "experiment_b_power_grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/example_validation")
    ap.add_argument("--quick", action="store_true", help="fewer replicates (smoke run)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    reps_a, reps_b = (20, 500) if a.quick else (200, 4000)
    print("A:", json.dumps(experiment_a(out, reps_a), indent=2))
    rows = experiment_b(out, reps_b)
    print("B: wrote", len(rows), "grid rows to", out / "experiment_b_power_grid.csv")


if __name__ == "__main__":
    main()
