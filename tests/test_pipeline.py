import json
from pathlib import Path

from llmbench import rubric, runner
from llmbench.report import compare, to_markdown
from llmbench.tasks import make_needle_suite, save_tasks

SAMPLE = str(Path(__file__).resolve().parents[1] / "data" / "tasks_sample.jsonl")


def cfg(acc_a, acc_b):
    return {"retries": 0, "workers": 2, "providers": [
        {"label": "a", "type": "mock", "model": "mock-a", "seed": 1, "accuracy": {"default": acc_a}},
        {"label": "b", "type": "mock", "model": "mock-b", "seed": 2, "accuracy": {"default": acc_b,},
         "price_per_mtok": {"input": 1.0, "output": 2.0}}]}


def test_end_to_end_perfect_vs_zero(tmp_path):
    runner.run(cfg(1.0, 0.0), SAMPLE, str(tmp_path), log=lambda *_: None)
    res = compare(SAMPLE, str(tmp_path), "a", "b", bootstrap_reps=300)
    assert res["overall"]["acc_a"] == 1.0 and res["overall"]["acc_b"] == 0.0
    assert res["overall"]["p"] < 1e-6
    assert all(r["acc_a"] == 1.0 for r in res["rows"])
    assert "writing" not in {r["dimension"] for r in res["rows"]}      # rubric tasks excluded
    md = to_markdown(res)
    assert "| **Overall** |" in md and "Holm" in md
    assert res["operations"]["b"]["cost_usd"] is not None


def test_resume_does_not_duplicate(tmp_path):
    runner.run(cfg(0.8, 0.8), SAMPLE, str(tmp_path), log=lambda *_: None)
    n1 = len((tmp_path / "a.jsonl").read_text().splitlines())
    runner.run(cfg(0.8, 0.8), SAMPLE, str(tmp_path), log=lambda *_: None)
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == n1 == 28


def test_manifest_records_task_hash_and_hides_env_names(tmp_path):
    runner.run(cfg(1, 1), SAMPLE, str(tmp_path), limit=3, log=lambda *_: None)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert len(m["tasks_sha256"]) == 64 and m["n_tasks"] == 3
    assert all("api_key_env" not in p for p in m["providers"])


def test_needle_suite_runs_through_mock(tmp_path):
    tasks = tmp_path / "needles.jsonl"
    save_tasks(make_needle_suite((300,), (0.2, 0.8), (1,)), tasks)
    runner.run(cfg(1.0, 0.0), str(tasks), str(tmp_path / "r"), log=lambda *_: None)
    res = compare(str(tasks), str(tmp_path / "r"), "a", "b", bootstrap_reps=100)
    assert res["rows"][0]["dimension"] == "long_context" and res["rows"][0]["acc_a"] == 1.0


def test_blind_export_agreement_and_unblind(tmp_path):
    runner.run(cfg(1, 1), SAMPLE, str(tmp_path), log=lambda *_: None)
    csv_path, key = tmp_path / "sheet.csv", tmp_path / "key.json"
    assert rubric.export_blind(SAMPLE, str(tmp_path), "a", "b", str(csv_path), str(key)) == 4
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert all(r["item_id"].endswith(("|X", "|Y")) for r in rows)
    assert not any(l in r["response"] for r in rows for l in ("model_a", "model_b"))
    paths = []
    for k, offset in enumerate((0, 0, 1)):                      # three raters
        p = tmp_path / f"r{k}.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            import csv as c
            w = c.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader()
            for i, r in enumerate(rows):
                w.writerow({**r, "score": (i % 3 + offset) % 3 if k == 2 else i % 3})
        paths.append(str(p))
    ag = rubric.agreement(paths)
    assert ag["items"] == 4 and ag["raters"] == 3 and -1 <= ag["fleiss_kappa"] <= 1
    assert set(rubric.unblind(paths, str(key))) == {"a", "b"}
