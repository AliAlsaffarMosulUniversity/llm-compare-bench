# llmbench — reproducible paired evaluation of two LLM chat systems

Companion code for the paper **"A Comprehensive Practical Comparison between Google Gemini and ChatGPT:
Architecture, Algorithms, Performance, and Productivity"** (Ali A. Alsaffar, University of Mosul).

The toolkit runs the *same* task set against two systems, scores the answers, and reports paired
statistics (exact McNemar test, paired bootstrap CIs, Holm correction, inter-rater agreement). It is
standard-library only at runtime.

> **Status.** The statistics, scorers, runner, blind-rating workflow and the OpenAI/Gemini request
> builders are unit-tested (32 tests; the HTTP adapters against a *local stub* that mimics the documented
> response shapes). The adapters have **not** been exercised against the live services in this repository's
> history, and the bundled 28-task set is a smoke test, not a benchmark. No result for any real model is
> included. Model names and prices are never hard-coded: you supply them in `config.json`.

## Install

```bash
git clone https://github.com/AliAlsaffarMosulUniversity/llm-compare-bench
cd llm-compare-bench
pip install -e ".[dev]"      # dev extras only needed for tests/figures
python -m pytest -q
```

## Run a real comparison

```bash
cp config.example.json config.json         # put exact model IDs; config.json is git-ignored
export OPENAI_API_KEY=...  GEMINI_API_KEY=...

llmbench gen-needles --output data/tasks_needles.jsonl     # optional long-context tasks
# or use the ready-made extended set (637 items: 607 auto-scored + 30 human-rated writing):
#   data/tasks_full.jsonl   (built by scripts/build_full_tasks.py; answers are computed, not typed)
cp data/tasks_full.jsonl data/my_tasks.jsonl

llmbench run    --config config.json --tasks data/my_tasks.jsonl --out results/run1
llmbench report --tasks data/my_tasks.jsonl --out results/run1 --a chatgpt --b gemini
```

`report` writes `results/run1/report.md` and `report.json`. Runs are resumable; re-running skips
finished items. `manifest.json` records the task-file SHA-256, providers, parameters and UTC start time.

### Open-ended (human-rated) tasks

```bash
llmbench export-blind --tasks data/my_tasks.jsonl --out results/run1 --a chatgpt --b gemini
# give results/rating_sheet.csv to >= 3 raters; each fills the `score` column (0/1/2)
llmbench agreement rater1.csv rater2.csv rater3.csv --key results/rating_key.json
```

Slots X/Y are randomised per item, so raters never see which system wrote what.

### Task format (`JSONL`)

```json
{"id":"know-01","dimension":"knowledge","scorer":"mcq","prompt":"…A) … B) …","reference":"B"}
```

Scorers: `mcq`, `numeric` (±`meta.tolerance`), `exact` (Arabic-aware normalisation), `code`
(`meta.tests` are appended to the model's code and run), `needle`, `abstain` (keyword heuristic —
audit manually), `rubric` (human rated).

## Validate the method (no API keys needed)

```bash
python scripts/validate_pipeline.py        # ~15 s; writes results/example_validation/
python scripts/make_figures.py
llmbench power --n 150 300 600 --delta 0 0.05      # quick power check
```

Using synthetic responders with a known accuracy gap, the full pipeline (runner → scorer → report)
recovers the gap, and the 95 % CI covers it in 96 % of 200 replicates. The Monte Carlo grid shows the exact
McNemar test keeps its size (≤ 5 %) and that **300 paired tasks detect a 5-point gap only about half
the time** (≈ 600 are needed for 80 % power at 80 % accuracy, ρ = 0.5). See `results/example_validation/`.

## Safety notes

* The `code` scorer runs model-generated code in a subprocess with CPU/memory limits. That is **not** a
  sandbox — run the harness in a container or VM.
* API keys are read from environment variables only and travel in HTTP headers, never in URLs or logs.
* Respect each provider's terms of service and rate limits; set `workers` conservatively.

## Layout

```
src/llmbench/   stats.py scoring.py providers.py runner.py report.py rubric.py power.py tasks.py cli.py
data/           tasks_sample.jsonl (28 smoke-test items), tasks_full.jsonl (637 items; scripts/build_full_tasks.py)
scripts/        validate_pipeline.py  make_figures.py  build_sample_tasks.py
tests/          37 unit/integration tests
results/example_validation/   synthetic-responder validation output (CSV/JSON)
docs/           figures used in the paper
```

## Citation

See `CITATION.cff`. MIT licence.
