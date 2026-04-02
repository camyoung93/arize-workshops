# Media Agent — Evaluator Setup Guide

Reference for creating Arize evaluators and tasks against this experiment.

Use the **arize-evaluator** AX skill to create these, or copy the `ax` commands below.

---

## Prerequisites

- Experiment created via `arize_experiment_ops.py` or `ax experiments create`
- An AI Integration configured in your Arize space (e.g., OpenAI or Anthropic)
- Dataset ID and Experiment ID from the steps above

---

## Evaluator 1: SQL Quality

Scores the generated SQL against the golden expected SQL.

**Template variables:** `{input}`, `{output}`, `{expected_sql}`

**Column mappings (experiment task):**
- `input` → dataset example field `question`
- `output` → experiment run field `sql_generated`
- `expected_sql` → dataset example field `expected_sql`

```bash
ax evaluators create \
  --name "SQL Quality" \
  --space-id $ARIZE_SPACE_ID \
  --template-name "sql_quality" \
  --commit-message "Initial version" \
  --ai-integration-id $AI_INTEGRATION_ID \
  --model-name "gpt-4o" \
  --include-explanations \
  --use-function-calling \
  --template 'You are a SQL evaluation expert. Compare the generated SQL query against the expected SQL query for the given question.

Question: {input}

Generated SQL: {output}

Expected SQL: {expected_sql}

Evaluate on these criteria:
1. Does the generated SQL answer the same question as the expected SQL?
2. Is the generated SQL syntactically valid?
3. Does it reference the correct tables and columns?
4. Is it reasonably efficient (no unnecessary subqueries or redundant joins)?

Note: The generated SQL does NOT need to be identical to the expected SQL. Different valid approaches to the same question are acceptable. The expected SQL may itself be intentionally less efficient.

Respond with exactly one of these labels: correct, partially_correct, incorrect'
```

---

## Evaluator 2: Response Quality

Scores the final synthesized answer for relevance, accuracy, and completeness.

**Template variables:** `{input}`, `{output}`, `{expected_answer}`

**Column mappings (experiment task):**
- `input` → dataset example field `question`
- `output` → experiment run field `output` (final answer)
- `expected_answer` → dataset example field `expected_answer`

```bash
ax evaluators create \
  --name "Response Quality" \
  --space-id $ARIZE_SPACE_ID \
  --template-name "response_quality" \
  --commit-message "Initial version" \
  --ai-integration-id $AI_INTEGRATION_ID \
  --model-name "gpt-4o" \
  --include-explanations \
  --use-function-calling \
  --template 'You are evaluating a data analyst assistant response. Given the question and the expected answer (which may be intentionally less optimal), judge the actual response.

Question: {input}

Actual Response: {output}

Reference Answer: {expected_answer}

Evaluate:
1. Does the response correctly answer the question?
2. Is the response grounded in data (not hallucinating)?
3. Is the response clear, concise, and well-structured?
4. Does the response match or exceed the quality of the reference answer?

A response that is MORE accurate, more concise, or better structured than the reference answer should still be rated positively — the reference is a baseline, not a ceiling.

Respond with exactly one of these labels: excellent, good, acceptable, poor'
```

---

## Evaluator 3: Constraint Following (for constraint_following category)

Scores whether the response follows explicit formatting/exclusion instructions.

**Template variables:** `{input}`, `{output}`

**Column mappings (experiment task — with query filter):**
- `input` → dataset example field `question`
- `output` → experiment run field `output`
- **Query filter:** `category = 'constraint_following'`

```bash
ax evaluators create \
  --name "Constraint Following" \
  --space-id $ARIZE_SPACE_ID \
  --template-name "constraint_following" \
  --commit-message "Initial version" \
  --ai-integration-id $AI_INTEGRATION_ID \
  --model-name "gpt-4o" \
  --include-explanations \
  --use-function-calling \
  --template 'You are evaluating whether a response follows explicit formatting and content constraints specified in the question.

Question (contains the constraints): {input}

Response: {output}

Check:
1. Does the response follow the exact format requested (sentence count, bullet structure, table format, etc.)?
2. Does it respect exclusion instructions (topics to avoid, words not to use)?
3. Does it follow persona/style constraints (non-technical, metaphorical, JSON-only, etc.)?

Respond with exactly one of these labels: follows_all, follows_most, violates'
```

---

## Creating a Task to Run Evaluators

After creating evaluators, create a task to run them against the experiment:

```bash
# Run SQL Quality + Response Quality on all experiment runs
ax tasks create \
  --name "Media Eval Scoring" \
  --task-type template_evaluation \
  --dataset-id $DATASET_ID \
  --experiment-ids "$EXPERIMENT_ID" \
  --evaluators '[
    {
      "evaluator_id": "'$SQL_QUALITY_EVAL_ID'",
      "column_mappings": {
        "input": "question",
        "output": "sql_generated",
        "expected_sql": "expected_sql"
      }
    },
    {
      "evaluator_id": "'$RESPONSE_QUALITY_EVAL_ID'",
      "column_mappings": {
        "input": "question",
        "output": "output",
        "expected_answer": "expected_answer"
      }
    }
  ]' \
  --no-continuous

# Trigger the scoring run
ax tasks trigger-run $TASK_ID \
  --experiment-ids "$EXPERIMENT_ID" \
  --wait
```

---

## Span-Level Evaluators (on project traces)

The batch runner sends all traces to Arize. You can also create span-level
evaluators on the project for ongoing monitoring:

**SQL Planner spans** — filter: `openinference.span.kind = 'CHAIN'` on
planner_stage spans. Map `input.value` → `{input}`, `output.value` → `{output}`.

**Synthesizer spans** — filter: `openinference.span.kind = 'CHAIN'` on
synthesizer_stage spans. Map `input.value` → `{input}`, `output.value` → `{output}`.

Use `ax tasks create --project-id <ID>` with appropriate column mappings
and query filters for span-level evaluation.

---

## Workflow Summary

```
1. Build dataset:     python -m experiments.build_experiment_dataset
2. Upload dataset:    python -m experiments.arize_dataset_setup
3. Run batch:         python -m experiments.run_experiment_batch
4. Upload experiment: python -m experiments.arize_experiment_ops --runs .scratch/experiment_runs_*.json
5. Create evaluators: Use arize-evaluator skill or ax commands above
6. Run scoring task:  ax tasks create ... && ax tasks trigger-run ...
7. View results:      Arize UI → Experiments → compare runs
```
