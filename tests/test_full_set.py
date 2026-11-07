from pathlib import Path

import pytest

from llmbench.scoring import score as _score


def score_task(t, r):
    return _score(t, r)[0]
from llmbench.tasks import load_tasks

FULL = Path(__file__).resolve().parents[1] / "data" / "tasks_full.jsonl"


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(FULL)


def test_counts_and_unique_ids(tasks):
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))
    assert len(tasks) >= 600
    assert len({t.prompt for t in tasks}) == len(tasks)


def test_mcq_keys_valid_and_balanced(tasks):
    keys = [t.reference for t in tasks if t.scorer == "mcq"]
    assert set(keys) <= set("ABCD")
    assert max(keys.count(k) for k in "ABCD") / len(keys) < 0.4


def test_code_reference_passes_and_wrong_fails(tasks):
    code = [t for t in tasks if t.scorer == "code"]
    assert len(code) == 30
    for t in code:
        good = score_task(t, "```python\n" + t.meta["solution"] + "\n```")
        bad = score_task(t, "```python\n" + t.meta["wrong_solution"] + "\n```")
        assert good == 1, t.id
        assert bad == 0, t.id


def test_needles_present_in_prompt(tasks):
    for t in [t for t in tasks if t.scorer == "needle"]:
        assert str(t.reference) in t.prompt, t.id


def test_mock_end_to_end_scores(tasks):
    # a response that echoes the reference must score 1 on every auto-scored non-code task
    for t in tasks:
        if t.scorer in ("mcq", "numeric", "exact", "needle"):
            ref = t.reference[0] if isinstance(t.reference, list) else t.reference
            assert score_task(t, f"Answer: {ref}") == 1, t.id
