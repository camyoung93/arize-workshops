# Media Agent — Experiment Pipeline

End-to-end experiment harness for the Media Agent NL-to-SQL pipeline.
Tests SQL generation quality and final answer synthesis across 145 diverse
questions, scores results in Arize using LLM-as-judge evaluators.

---

## Quick start

```bash
# From the media-agent directory
cd media-agent

# 0. Make sure the app is set up (DB seeded, .env configured)
python seed_db.py

# 1. Generate the experiment dataset (one-time, already checked in)
python -m experiments.build_experiment_dataset

# 2. Upload dataset to Arize
python -m experiments.arize_dataset_setup

# 3. Run the pipeline on all 145 questions (traces go to Arize automatically)
python -m experiments.run_experiment_batch

# 4. Upload experiment results to Arize
python -m experiments.arize_experiment_ops --runs .scratch/experiment_runs_*.json

# 5. Create evaluators and run scoring (see EVALUATORS.md or use AX skills)
```

---

## What's in the dataset

145 questions across 5 categories, each with golden SQL and a golden answer:

| Category | Count | What it tests |
|---|---|---|
| `simple` | 35 | Single-table lookups, counts, filters |
| `aggregation` | 33 | GROUP BY, ranking, temporal comparisons |
| `multi_hop` | 32 | Joins across 2+ tables, sequential reasoning |
| `constraint_following` | 25 | Formatting, exclusion, persona instructions |
| `edge_case` | 20 | Missing data, schema mismatch, out-of-range |

Difficulty spread: 49 easy, 67 medium, 29 hard.

Some golden answers are **intentionally less efficient** (correct but verbose
SQL, or acceptable but not optimal prose) so that experiment scoring can surface
cases where the model actually outperforms the baseline.

---

## Architecture

```
One-time setup                        Reusable per experiment
─────────────                         ──────────────────────
build_experiment_dataset.py           run_experiment_batch.py
  └─▸ data/experiment_dataset.json      └─▸ .scratch/experiment_runs_*.json
                                              (traces sent to Arize live)
arize_dataset_setup.py                arize_experiment_ops.py
  └─▸ Arize dataset                     └─▸ Arize experiment
       "media-agent-experiment"

Then: create evaluators + run scoring tasks via AX skills / ax CLI
```

---

## File reference

| File | Run | Purpose |
|---|---|---|
| `build_experiment_dataset.py` | One-time | Generates 145-question dataset to `data/experiment_dataset.json` |
| `arize_dataset_setup.py` | One-time | Uploads dataset to Arize (SDK or ax CLI) |
| `run_experiment_batch.py` | Per experiment | Runs pipeline, captures SQL + answers, writes run records |
| `arize_experiment_ops.py` | Per experiment | Creates Arize experiment from run records |
| `schemas.py` | — | Typed `ExperimentExample` and `ExperimentRun` dataclasses |
| `EVALUATORS.md` | Reference | Evaluator templates, column mappings, task setup |
| `data/experiment_dataset.json` | Artifact | The 145-row dataset (checked in) |

---

## Scripts in detail

### build_experiment_dataset.py

Deterministic generator — always produces the same 145 rows. Categories and
golden answers are hardcoded against the app's SQLite schema (articles, authors,
revenue, traffic).

```bash
python -m experiments.build_experiment_dataset
# Wrote 145 examples to experiments/data/experiment_dataset.json
#   Categories: {'simple': 35, 'aggregation': 33, ...}
```

### arize_dataset_setup.py

Uploads the dataset JSON to Arize as a GENERATIVE dataset. Requires
`ARIZE_API_KEY` and `ARIZE_SPACE_ID` in `.env` or environment.

```bash
# SDK path
python -m experiments.arize_dataset_setup

# ax CLI path
ax datasets create --name "media-agent-experiment" \
  --space-id $ARIZE_SPACE_ID --file experiments/data/experiment_dataset.json
```

### run_experiment_batch.py

Runs each dataset question through the live app pipeline (`run_pipeline`),
captures the final answer, planner SQL, plan text, latency, and metadata.
All traces are sent to Arize automatically via the app's existing instrumentation.

```bash
# Full batch
python -m experiments.run_experiment_batch

# Dry-run (first 10)
python -m experiments.run_experiment_batch --count 10

# Force a specific synthesizer prompt version
python -m experiments.run_experiment_batch --prompt-version v2

# Custom output path
python -m experiments.run_experiment_batch --output my_runs.json
```

Output: `.scratch/experiment_runs_<timestamp>.json`

### arize_experiment_ops.py

Creates an Arize experiment from collected run records, linked to the
experiment dataset.

```bash
# SDK path
python -m experiments.arize_experiment_ops --runs .scratch/experiment_runs_20260330_120000.json

# With a custom experiment name
python -m experiments.arize_experiment_ops --runs .scratch/experiment_runs_*.json --name "v1-baseline"

# ax CLI path
ax experiments create --name "media-experiment-v1" \
  --dataset-id <DATASET_ID> --file .scratch/experiment_runs_*.json
```

---

## Evaluator setup

Three evaluators are designed for this experiment. Full templates, column
mappings, and `ax` commands are in [`EVALUATORS.md`](EVALUATORS.md):

1. **SQL Quality** — compares generated SQL vs golden SQL (correct / partially_correct / incorrect)
2. **Response Quality** — scores final answer vs golden answer, rubric-based (excellent / good / acceptable / poor)
3. **Constraint Following** — checks formatting/exclusion compliance (follows_all / follows_most / violates)

Use the **arize-evaluator** AX skill to create these interactively, or copy
the `ax evaluators create` commands from EVALUATORS.md.

---

## Dataset row shape

Each example in `experiment_dataset.json`:

```json
{
  "id": "eval_001",
  "question": "How many articles were published in total?",
  "category": "simple",
  "difficulty": "easy",
  "expected_sql": "SELECT COUNT(*) AS article_count FROM articles",
  "expected_answer": "There are 200 articles in total.",
  "must_include": ["200"],
  "must_not_include": [],
  "notes": "",
  "role": "finance"
}
```

## Experiment run shape

Each run in the output JSON:

```json
{
  "example_id": "eval_001",
  "output": "The database contains 200 articles in total.",
  "sql_generated": "SELECT COUNT(*) FROM articles",
  "sql_plan": "Count all rows in the articles table.",
  "tables_referenced": "[\"articles\"]",
  "prompt_version": "v1",
  "latency_ms": 2340.5,
  "category": "simple",
  "difficulty": "easy",
  "error": null
}
```

---

## Typical workflow

```
1. python -m experiments.build_experiment_dataset     # generate dataset (one-time)
2. python -m experiments.arize_dataset_setup          # upload to Arize (one-time)
3. python -m experiments.run_experiment_batch         # run pipeline on all 145 Qs
4. python -m experiments.arize_experiment_ops \       # upload experiment
     --runs .scratch/experiment_runs_*.json
5. Create evaluators (arize-evaluator skill)          # SQL Quality, Response Quality, etc.
6. ax tasks create ... && ax tasks trigger-run        # score the experiment
7. Arize UI → Experiments → view scores               # compare, filter, analyze
```

---

## Notes

- All run artifacts go to `.scratch/` (gitignored). Only the dataset source
  (`data/experiment_dataset.json`) is checked in.
- The batch runner uses role `finance` for all questions so guardrail behavior
  doesn't interfere with SQL/synthesis scoring.
- Traces are sent to the Arize project configured in `.env`
  (default: `media-agent-demo`).
