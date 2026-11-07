"""Automatic scorers. Each returns (score, note); score is 1, 0, or None.

``None`` means "needs human rating" (rubric tasks) and is excluded from the
paired tests.

Security note: the ``code`` scorer executes model-generated code in a
subprocess with CPU/memory limits. That is NOT a security sandbox. Run the
harness inside a container or VM when evaluating untrusted output.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from typing import Optional

from .tasks import Task

_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_ARABIC_DIACRITICS = re.compile("[ً-ٰٟـ]")

_ABSTAIN_RE = re.compile(
    r"(does not exist|doesn't exist|do not exist|no such|no (?:record|evidence|information|reliable)|"
    r"not aware of|cannot verify|can't verify|unable to (?:verify|find)|i don't have|i do not have|"
    r"fictional|made[- ]up|appears to be (?:fictional|invented)|has(?:n't| not) (?:happened|occurred)|"
    r"not (?:yet )?(?:happened|occurred|been held)|in the future|no (?:world cup|such event)|"
    r"لا أملك|لا اعرف|لا أعرف|لا يوجد|لا توجد|غير موجود|خيالي|لم يحدث|لا يمكنني التحقق|لم تقم)",
    re.IGNORECASE,
)


def normalize_text(s: str) -> str:
    s = s.translate(_DIGIT_MAP)
    s = _ARABIC_DIACRITICS.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"[^\w\s.\-]", " ", s)


def score_mcq(task: Task, response: str):
    m = re.findall(r"answer\s*[:：]\s*\(?\s*([A-Ea-e])\b", response, flags=re.IGNORECASE)
    if not m:
        return 0, "no 'Answer: X' line"
    return int(m[-1].upper() == str(task.reference).upper()), f"parsed={m[-1].upper()}"


def score_numeric(task: Task, response: str):
    text = response.translate(_DIGIT_MAP).replace(",", "")
    tail = re.split(r"final answer\s*[:：]", text, flags=re.IGNORECASE)
    segment = tail[-1] if len(tail) > 1 else text
    nums = re.findall(r"-?\d+(?:\.\d+)?", segment)
    if not nums:
        return 0, "no number found"
    got = float(nums[0] if len(tail) > 1 else nums[-1])
    tol = float(task.meta.get("tolerance", 1e-6))
    return int(abs(got - float(task.reference)) <= tol), f"parsed={got}"


def score_exact(task: Task, response: str):
    refs = task.reference if isinstance(task.reference, list) else [task.reference]
    body = normalize_text(response)
    hit = any(normalize_text(str(r)) in body for r in refs)
    return int(hit), "contains reference" if hit else "reference not found"


def score_needle(task: Task, response: str):
    return int(str(task.reference) in response.translate(_DIGIT_MAP)), "needle"


def score_abstain(task: Task, response: str):
    hit = bool(_ABSTAIN_RE.search(response))
    return int(hit), "heuristic abstention match" if hit else "no abstention phrase (heuristic)"


def _extract_code(response: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    return blocks[0] if blocks else response


def _limits() -> None:  # pragma: no cover - runs in the child process
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 29, 1 << 29))
    except Exception:
        pass


def run_python_tests(code: str, tests: str, timeout: float = 10.0) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code + "\n\n" + tests + "\n")
        try:
            kwargs = {"preexec_fn": _limits} if os.name == "posix" else {}
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True,
                               text=True, timeout=timeout, env={"PATH": os.environ.get("PATH", "")},
                               **kwargs)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if p.returncode == 0:
            return True, "tests passed"
        return False, (p.stderr.strip().splitlines() or ["failed"])[-1][:200]


def score_code(task: Task, response: str):
    ok, note = run_python_tests(_extract_code(response), task.meta["tests"])
    return int(ok), note


def score_rubric(task: Task, response: str):
    return None, "needs human rating"


_DISPATCH = {"mcq": score_mcq, "numeric": score_numeric, "exact": score_exact,
             "needle": score_needle, "abstain": score_abstain, "code": score_code,
             "rubric": score_rubric}


def score(task: Task, response: Optional[str]):
    if response is None:
        return 0, "no response"
    return _DISPATCH[task.scorer](task, response)
