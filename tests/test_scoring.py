import json
from pathlib import Path

import pytest

from llmbench import scoring
from llmbench.tasks import Task, load_tasks, make_needle_task

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "tasks_sample.jsonl"


def T(scorer, ref, **meta):
    return Task("t", "d", "p", scorer, ref, meta)


def test_mcq():
    t = T("mcq", "B")
    assert scoring.score(t, "Reasoning...\nAnswer: B")[0] == 1
    assert scoring.score(t, "Answer: (b)")[0] == 1
    assert scoring.score(t, "Answer: A\nAnswer: B")[0] == 1   # last wins
    assert scoring.score(t, "Answer: C")[0] == 0
    assert scoring.score(t, "I think B")[0] == 0


def test_numeric_with_arabic_digits_and_commas():
    assert scoring.score(T("numeric", 56), "Final answer: ٥٦")[0] == 1
    assert scoring.score(T("numeric", 1210), "so total is 1,210\nFinal answer: 1,210")[0] == 1
    assert scoring.score(T("numeric", 0.25, tolerance=1e-6), "Final answer: 0.25")[0] == 1
    assert scoring.score(T("numeric", 21), "Final answer: 22")[0] == 0
    assert scoring.score(T("numeric", 21), "no digits here")[0] == 0


def test_exact_arabic_normalisation():
    t = T("exact", ["بغداد"])
    assert scoring.score(t, "العاصمة هي بَغْدَاد")[0] == 1
    assert scoring.score(T("exact", ["Au"]), "The symbol is Au.")[0] == 1
    assert scoring.score(T("exact", ["1969"]), "It was 1968")[0] == 0


def test_abstain_heuristic():
    t = T("abstain", "ABSTAIN")
    assert scoring.score(t, "I cannot verify that this book exists.")[0] == 1
    assert scoring.score(t, "لا يوجد كاتب بهذا الاسم")[0] == 1
    assert scoring.score(t, "It is 'The Silent Orchard' (1987).")[0] == 0


def test_needle_task_generation_is_deterministic_and_scored():
    a, b = make_needle_task(7, 500, 0.5), make_needle_task(7, 500, 0.5)
    assert a.prompt == b.prompt and a.reference == b.reference
    assert a.reference in a.prompt
    assert scoring.score(a, f"The code is {a.reference}.")[0] == 1
    assert scoring.score(a, "The code is 123456.")[0] == 0


def test_rubric_is_unscored():
    assert scoring.score(T("rubric", None), "anything")[0] is None


def test_none_response_scores_zero():
    assert scoring.score(T("mcq", "A"), None)[0] == 0


def test_every_sample_code_task_solution_passes_and_wrong_fails():
    code_tasks = [t for t in load_tasks(SAMPLE) if t.scorer == "code"]
    assert len(code_tasks) == 4
    for t in code_tasks:
        good = f"```python\n{t.meta['solution']}\n```"
        bad = f"```python\n{t.meta['wrong_solution']}\n```"
        assert scoring.score(t, good)[0] == 1, t.id
        assert scoring.score(t, bad)[0] == 0, t.id


def test_code_timeout_and_infinite_loop():
    t = T("code", None, tests="f()")
    assert scoring.run_python_tests("def f():\n    while True: pass", "f()", timeout=2)[0] is False


def test_sample_set_is_well_formed():
    tasks = load_tasks(SAMPLE)
    assert len(tasks) == len({t.id for t in tasks}) == 28
    assert {t.dimension for t in tasks} >= {"knowledge", "reasoning", "coding", "factuality", "arabic", "writing"}
    for t in tasks:
        if t.scorer == "mcq":
            assert str(t.reference) in "ABCD"
