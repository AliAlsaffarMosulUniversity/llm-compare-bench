"""Run tasks against providers and store raw responses (resumable JSONL)."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .providers import ProviderError, build_provider
from .tasks import Task, load_tasks


def _load_done(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                if not o.get("error"):
                    done.add((o["task_id"], o["rep"]))
    return done


def run(config: dict, tasks_path: str, out_dir: str, repeats: int = 1, limit: int | None = None,
        log=print) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(tasks_path)
    if limit:
        tasks = tasks[:limit]
    retries = int(config.get("retries", 4))
    system_prompt = config.get("system_prompt", "")
    providers = [build_provider(s, retries, system_prompt) for s in config["providers"]]

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "tasks_file": str(tasks_path),
        "tasks_sha256": hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest(),
        "n_tasks": len(tasks), "repeats": repeats, "system_prompt": system_prompt,
        "providers": [{k: v for k, v in s.items() if k != "api_key_env"} for s in config["providers"]],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    workers = max(1, int(config.get("workers", 1)))
    for prov in providers:
        path = out / f"{prov.label}.jsonl"
        done = _load_done(path)
        todo = [(t, r) for r in range(repeats) for t in tasks if (t.id, r) not in done]
        log(f"[{prov.label}] {len(todo)} to run, {len(done)} already done")

        def one(item):
            task, rep = item
            rec = {"task_id": task.id, "rep": rep, "provider": prov.label, "model": prov.model,
                   "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            try:
                r = prov.generate(task.full_prompt, task)
                rec.update(response=r.text, latency_s=round(r.latency_s, 4),
                           input_tokens=r.input_tokens, output_tokens=r.output_tokens, error=None)
            except ProviderError as e:
                rec.update(response=None, latency_s=None, input_tokens=None, output_tokens=None, error=str(e))
            return rec

        with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
            for n, rec in enumerate(ex.map(one, todo), 1):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if rec["error"]:
                    log(f"  ! {rec['task_id']}: {rec['error']}")
                if n % 25 == 0:
                    log(f"  {prov.label}: {n}/{len(todo)}")
    return out
