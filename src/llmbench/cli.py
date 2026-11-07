"""Command-line interface: ``llmbench <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import power, rubric, runner
from .report import compare, to_markdown
from .tasks import make_needle_suite, save_tasks


def _cmd_run(a):
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    runner.run(cfg, a.tasks, a.out, repeats=a.repeats, limit=a.limit)


def _cmd_report(a):
    res = compare(a.tasks, a.out, a.a, a.b, rep=a.rep, bootstrap_reps=a.bootstrap)
    md = to_markdown(res)
    Path(a.out, "report.md").write_text(md, encoding="utf-8")
    Path(a.out, "report.json").write_text(json.dumps(res, indent=2, default=list), encoding="utf-8")
    print(md)


def _cmd_needles(a):
    save_tasks(make_needle_suite(tuple(a.lengths), tuple(a.depths), tuple(range(1, a.seeds + 1))), a.output)
    print(f"wrote {a.output}")


def _cmd_blind(a):
    n = rubric.export_blind(a.tasks, a.out, a.a, a.b, a.csv, a.key, seed=a.seed)
    print(f"wrote {n} rows to {a.csv}; keep {a.key} private until rating is finished")


def _cmd_agree(a):
    print(json.dumps(rubric.agreement(a.raters), indent=2))
    if a.key:
        print(json.dumps(rubric.unblind(a.raters, a.key), indent=2))


def _cmd_power(a):
    print("n,p_a,p_b,rho,rejection_rate")
    for n in a.n:
        for d in a.delta:
            r = power.simulate_rejection_rate(n, a.acc, a.acc - d, a.rho, a.reps, seed=a.seed)
            print(f"{n},{a.acc},{a.acc - d:.3f},{a.rho},{r.rejection_rate:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="llmbench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="query the providers in a config file")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", default="data/tasks_sample.jsonl")
    p.add_argument("--out", default="results/run1")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("report", help="score responses and compare two systems")
    p.add_argument("--tasks", default="data/tasks_sample.jsonl")
    p.add_argument("--out", default="results/run1")
    p.add_argument("--a", required=True, help="label of system A")
    p.add_argument("--b", required=True, help="label of system B")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.set_defaults(fn=_cmd_report)

    p = sub.add_parser("gen-needles", help="generate long-context retrieval tasks")
    p.add_argument("--output", default="data/tasks_needles.jsonl")
    p.add_argument("--lengths", type=int, nargs="+", default=[2000, 8000])
    p.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--seeds", type=int, default=2)
    p.set_defaults(fn=_cmd_needles)

    p = sub.add_parser("export-blind", help="make anonymised rating sheets for open-ended tasks")
    p.add_argument("--tasks", default="data/tasks_sample.jsonl")
    p.add_argument("--out", default="results/run1")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--csv", default="results/rating_sheet.csv")
    p.add_argument("--key", default="results/rating_key.json")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_cmd_blind)

    p = sub.add_parser("agreement", help="inter-rater agreement (and optional unblinding)")
    p.add_argument("raters", nargs="+", help="filled rating CSVs, one per rater")
    p.add_argument("--key")
    p.set_defaults(fn=_cmd_agree)

    p = sub.add_parser("power", help="Monte Carlo power / type-I error of the McNemar test")
    p.add_argument("--n", type=int, nargs="+", default=[150, 300])
    p.add_argument("--delta", type=float, nargs="+", default=[0.0, 0.05])
    p.add_argument("--acc", type=float, default=0.8)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_cmd_power)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
