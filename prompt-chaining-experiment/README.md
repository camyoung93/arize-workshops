# Prompt Chaining Experiment

A quickstart that walks through the Arize **Prompt Lifecycle** end to end: create two prompts in Prompt Hub, chain them together via OpenAI, and log the results as an experiment — all in a single TypeScript script.

```
Dataset (topics)
  │
  ▼
Prompt Hub: "content-drafter"         Prompt Hub: "content-refiner"
  │                                     │
  │  {topic} ──► OpenAI ──► draft       │  {draft} ──► OpenAI ──► refined output
  │                          │          │                          │
  └──────────────────────────┘          └──────────────────────────┘
                                                    │
                                                    ▼
                                          Arize Experiment
                                        (view + evaluate in UI)
```

## Prerequisites

- **Node.js** 20+
- An **Arize** account with an API key ([app.arize.com](https://app.arize.com))
- An **OpenAI** API key

## Setup

```bash
cd prompt-chaining-experiment

# 1. Set up the ax CLI (if you haven't already)
pip install arize-ax-cli
ax profiles create          # interactive — enters your API key and region

# 2. Copy and fill in credentials
cp .env.example .env
#    ARIZE_API_KEY  — from app.arize.com → Settings → API Keys
#    ARIZE_SPACE_ID — your space name or ID
#    OPENAI_API_KEY — from platform.openai.com

# 3. Install dependencies
npm install
```

## Run

```bash
npm start          # runs: tsx src/main.ts
```

The script executes six steps in order:

### Step 1 — Create Prompt 1 ("content-drafter")

Registers a prompt in Prompt Hub with a `{topic}` variable.

<details>
<summary>ax CLI equivalent</summary>

```bash
ax prompts create \
  --name "content-drafter" \
  --space "$ARIZE_SPACE_ID" \
  --provider openAI \
  --model gpt-4o-mini \
  --input-variable-format f_string \
  --description "Generates a raw blog introduction from a topic" \
  --messages '[
    {"role":"system","content":"You are a skilled content writer. Write a short, engaging blog introduction (2-3 paragraphs) for the given topic."},
    {"role":"user","content":"Topic: {topic}"}
  ]'
```

</details>

### Step 2 — Create Prompt 2 ("content-refiner")

Registers a second prompt with a `{draft}` variable that accepts the output of Prompt 1.

<details>
<summary>ax CLI equivalent</summary>

```bash
ax prompts create \
  --name "content-refiner" \
  --space "$ARIZE_SPACE_ID" \
  --provider openAI \
  --model gpt-4o-mini \
  --input-variable-format f_string \
  --description "Polishes a draft for tone, clarity, and conciseness" \
  --messages '[
    {"role":"system","content":"You are an expert editor. Refine the following draft for clarity, conciseness, and a professional tone. Return only the improved text."},
    {"role":"user","content":"{draft}"}
  ]'
```

</details>

### Step 3 — Create dataset

Uploads five inline topics as a dataset to Arize.

<details>
<summary>ax CLI equivalent</summary>

```bash
ax datasets create \
  --name "prompt-chaining-topics" \
  --space "$ARIZE_SPACE_ID" \
  --json '[
    {"topic":"Why observability matters for LLM applications"},
    {"topic":"Getting started with prompt engineering"},
    {"topic":"Best practices for evaluating AI output quality"},
    {"topic":"How prompt chaining improves complex workflows"},
    {"topic":"The role of experiments in iterating on LLM systems"}
  ]'
```

</details>

### Step 4 — Pull prompts from Prompt Hub

Fetches the latest version of each prompt so the script has the message templates.

<details>
<summary>ax CLI equivalent</summary>

```bash
ax prompts get <drafter-prompt-id>
ax prompts get <refiner-prompt-id>
```

</details>

### Step 5 — Run the prompt chain

For each topic in the dataset:

1. Substitute `{topic}` into the drafter messages and call OpenAI → **draft**
2. Substitute `{draft}` into the refiner messages and call OpenAI → **refined output**

### Step 6 — Log experiment

Creates an experiment in Arize with one run per topic, each containing the refined output (and the intermediate draft as metadata).

<details>
<summary>ax CLI equivalent</summary>

```bash
# After writing runs to a JSON file with example_id + output columns:
ax experiments create \
  --name "prompt-chaining-v1" \
  --dataset <dataset-id> \
  --file runs.json
```

</details>

## What to look for in Arize

Go to **app.arize.com → Datasets & Experiments**:

1. **Dataset**: `prompt-chaining-topics` — the five topics used as input.
2. **Experiment**: `prompt-chaining-v1` — one run per topic with both the intermediate draft and the final refined output.
3. **Evaluate**: Create evaluators (LLM-as-judge or code-based) to score output quality, then run them against the experiment.

## Project structure

```
prompt-chaining-experiment/
├── src/
│   └── main.ts          # Single script: setup + run, top to bottom
├── .env.example
├── package.json
├── tsconfig.json
└── README.md
```
