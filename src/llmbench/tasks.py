"""Task definition, loading, and generators."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCORERS = {"mcq", "numeric", "exact", "code", "abstain", "needle", "rubric"}

FORMAT_HINTS = {
    "mcq": "\n\nEnd your reply with a final line of the form: Answer: <letter>",
    "numeric": "\n\nEnd your reply with a final line of the form: Final answer: <number>",
    "code": "\n\nReturn the complete function in a single ```python code block.",
}


@dataclass
class Task:
    id: str
    dimension: str
    prompt: str
    scorer: str
    reference: Any = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scorer not in SCORERS:
            raise ValueError(f"task {self.id}: unknown scorer {self.scorer!r}")

    @property
    def full_prompt(self) -> str:
        return self.prompt + FORMAT_HINTS.get(self.scorer, "")


def load_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            t = Task(id=obj["id"], dimension=obj["dimension"], prompt=obj["prompt"],
                     scorer=obj["scorer"], reference=obj.get("reference"),
                     meta=obj.get("meta", {}))
            if t.id in seen:
                raise ValueError(f"{path}:{ln}: duplicate task id {t.id}")
            seen.add(t.id)
            tasks.append(t)
    return tasks


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps({"id": t.id, "dimension": t.dimension, "prompt": t.prompt,
                                 "scorer": t.scorer, "reference": t.reference,
                                 "meta": t.meta}, ensure_ascii=False) + "\n")


_FILLER = [
    "The committee reviewed the quarterly logistics report and noted a steady rise in regional shipments.",
    "Rainfall across the northern plateau remained below the seasonal average for the third consecutive month.",
    "A small workshop on spreadsheet hygiene concluded that consistent column naming prevents many errors.",
    "The library extended its evening hours to accommodate students preparing for final examinations.",
    "Engineers inspected the bridge supports and recorded only minor surface corrosion near the eastern pier.",
    "The orchestra rehearsed the second movement twice before moving on to the closing section.",
    "Local farmers reported that the new irrigation schedule reduced water use without lowering yields.",
    "An internal memo reminded staff that invoices must be archived for at least five years.",
]


def make_needle_task(seed: int, n_words: int = 2000, depth: float = 0.5) -> Task:
    """Long-context retrieval task: find a code hidden at a relative depth.

    The filler is deterministic given ``seed``. The code is a random 6-digit
    number so the answer cannot be guessed from world knowledge.
    """
    rng = random.Random(seed)
    code = f"{rng.randint(100000, 999999)}"
    needle = f"The maintenance override code for vault {rng.randint(10, 99)} is {code}."
    words: list[str] = []
    while len(words) < n_words:
        words.extend(rng.choice(_FILLER).split())
    words = words[:n_words]
    pos = int(depth * len(words))
    text = " ".join(words[:pos]) + " " + needle + " " + " ".join(words[pos:])
    prompt = (text + "\n\nQuestion: What is the maintenance override code mentioned in the "
              "text above? Reply with the number only.")
    return Task(id=f"needle-{n_words}-{int(depth * 100):03d}-{seed}", dimension="long_context",
                prompt=prompt, scorer="needle", reference=code,
                meta={"n_words": n_words, "depth": depth})


def make_needle_suite(lengths=(2000, 8000), depths=(0.1, 0.5, 0.9), seeds=(1, 2)) -> list[Task]:
    return [make_needle_task(s, n, d) for n in lengths for d in depths for s in seeds]
