# Synthetic Spans

Notebooks and scripts for generating synthetic OpenInference traces and sending them to Arize (AX) or Phoenix (PX). Each subfolder targets a specific industry use case with a realistic but anonymized domain.

## Layout

```
synthetic-data/
├── financial-data/    # Financial data / research agent traces (NL-to-SQL, briefing, research augment)
├── telecom/           # Telecom call center agent traces
├── banking/           # Retail banking agent traces (stock, transactions, risk)
├── streaming/         # Video streaming search augment traces + feature store monitoring
├── generic/           # Industry-agnostic demos (AE triage, NL-to-filter)
├── config/
│   └── template.yaml  # Configuration template for the span generator
├── .env.example       # Required env vars
├── requirements.txt
└── requirements-v7.txt
```

## Notebooks and scripts

| Subfolder | File | What it generates |
|-----------|------|-------------------|
| financial-data | `synthetic_spans_findata_media_agent.py` | Data Analytics team NL-to-SQL agent traces (coordinator + planning/engineer/response agents, subscription domain) |
| financial-data | `synthetic_spans_findata_briefing.ipynb` | Personalized briefing traces (wires, aggregation, personas) |
| financial-data | `synthetic_spans_findata_research_augment.ipynb` | Research report augmentation traces (Vertex/Gemini metadata) |
| telecom | `synthetic_spans_for_telco.ipynb` | Call center agent traces (greeting, verification, account lookup) |
| banking | `synthetic_spans_for_retail_bank.ipynb` | Financial agent traces (stock analysis, transactions, risk) |
| streaming | `synthetic_spans_streamflix_search_augment.ipynb` | Search augment traces (catalog, reranking, personalized response) |
| streaming | `streamflix_feature_store_ingest.py` | Feature store monitoring: 30 days offline/online snapshots, drift on genre_affinity_score and device_type |
| generic | `synthetic_spans_ae_triage.ipynb` | AE triage agent (Scenario A read-only, Scenario B blocked tool) |
| generic | `synthetic_spans_ae_triage_scenario_c.ipynb` | AE triage v2 with dataset experiments and OTLP traces |
| generic | `synthetic_spans_nl_to_filter.ipynb` | NL-to-filter / NLP-to-SQL synthetic traces |

## Setup

**Python:** Arize SDK v8 requires **Python 3.10+**. Use a 3.10+ interpreter when creating the venv.

1. Create a venv in this directory:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Arize credentials.
3. Export the env vars (or use `python-dotenv`):
   ```bash
   export $(grep -v '^#' .env | xargs)
   ```
4. Open the notebook for your target scenario and run it.

## Config

`config/template.yaml` is a full configuration template for the span generator (scenarios, tools, poisoned data, agent mapping). Copy and customize for your use case if you need YAML-based config instead of inline notebook config.
