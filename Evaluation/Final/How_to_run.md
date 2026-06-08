# SDRF Benchmark — LLM Judge Evaluation Pipeline

## Overview

This pipeline evaluates the quality of metadata extracted from proteomics papers by LLM based agents (Claude, GPT, Gemini, LLaMA). It uses **Gemma-4-31B-IT** as an LLM judge (via OpenRouter) to assess each extracted value against the original paper text, producing verdicts of `high`, `medium` or `low`.

Three specialised extraction agents are evaluated per model:

- **BiologicalAgent** — sample level metadata (species, tissue, disease, sex, age, etc.)
- **TechnicalAgent** — MS instrument and protocol metadata (instrument, cleavage agent, labeling, fragmentation, etc.)
- **ExperimentalDesignAgent** — study design metadata (replicates, samples, fractions, experimental design, etc.)

No golden SDRF files are used. Correctness is judged entirely by the LLM judge reading the original manuscript text.

---

## 1. Requirements

### Python Packages

Install all dependencies before running:

```bash
pip install pandas numpy matplotlib openai deepeval
```

### Environment Variable

You **must** set your OpenRouter API key before running:

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

---

## 2. Required Input Files

### Directory Structure

```
/Users/fateme/Desktop/Hari_results/
├── test_set/
│   └── <PXD_ID>/
│       └── manuscript.txt          # Title + Abstract + Methods sections
└── media/volume/.../Final_results/
    └── test_claude/                # One extraction dir per model
        ├── BiologicalAgent/
│       │   └── <PXD_ID>_biological.json
        ├── TechnicalAgent/
│       │   └── <PXD_ID>_technical.json
        └── ExperimentalDesignAgent/
            └── <PXD_ID>_experimental.json
```

### Manuscript Format

Each `manuscript.txt` should contain clearly delimited sections. The pipeline extracts only the title, abstract and methods — longer results/discussion text is intentionally excluded to focus the judge:

```
=== TITLE ===
...

=== ABSTRACT ===
...

=== METHODS ===
...
```

If no section markers are found, the full file is used.

### Extraction JSON Format

Each agent JSON file maps raw field names to extracted values. Values may be plain strings, or alternating `[value, evidence_sentence, value, evidence_sentence, ...]` lists. Evidence sentences (strings longer than 120 characters) and sentinel strings (`"unknown"`, `"n/a"`, `"none"`, `"null"`, `""`) are automatically stripped.

```json
{
  "species": "Homo sapiens",
  "tissue": ["liver", "Liver tissue was collected...", "plasma", "Plasma samples were..."],
  "disease_state": "unknown"
}
```

---

## 3. How to Run

### Basic Usage

```bash
python LLm_as_judge.py --model claude
python LLm_as_judge.py --model gpt
python LLm_as_judge.py --model gemini
python LLm_as_judge.py --model llama
```

### All Options

```bash
python LLm_as_judge.py \
  --model   claude        # Required. One of: claude | gpt | gemini | llama
  --no-judge              # Skip LLM judge calls (coverage CSV only)
  --limit   10            # Evaluate only the first N papers
  --workers 4             # Max parallel LLM judge threads (default: 4)
```

### Example Runs

```bash
# Evaluate GPT extractions, first 5 papers only
python LLm_as_judge.py --model gpt --limit 5

# Generate coverage CSV without calling the LLM judge
python LLm_as_judge.py --model gemini --no-judge

# Full run with 8 parallel judge threads
python LLm_as_judge.py --model llama --workers 8
```

---

## 4. Internal Pipeline Flow

For each paper the pipeline runs these steps in order:

```
1. Discover PXD IDs from agent output directories
2. Load manuscript.txt -> extract title + abstract + methods
3. Load per-agent extraction JSONs
   └─ Flatten alternating value/evidence lists
   └─ Strip sentinels and evidence sentences
   └─ Map raw field names to canonical names (FIELD_NAME_MAP)
4. Build coverage rows   -> one row per agent * pipeline field
5. Build eval rows       -> deduplicated canonical field * value pairs
6. Two-pass LLM judge evaluation
   ├─ Pass 1: judge all values with raw sibling context
   └─ Pass 2: rejudge "medium" verdicts with clean sibling context
              (bad values pruned from sibling sets after Pass 1)
7. Write per paper JSON append to CSVs
8. Compute per paper accuracy statistics
```

All CSV outputs are written **incrementally** after each paper, so results are available during a long run even if interrupted.

---

## 5. Field Name Mapping

Raw agent field names are mapped to canonical evaluation names before judging:

| Raw Field (in JSON) | Canonical Name (evaluated) |
|---|---|
| `tissue` | `organ` |
| `disease_state` | `disease` |
| `sample_source` | `material_type` |
| `labeling` | `label` |
| `fragmentation_method` | `fragmentation` |
| `fractionation_method` | `fractionation` |
| `number_of_biological_replicates` / `biological_replicate` | `replicates` |
| `number_of_technical_replicates` / `technical_replicate` | `technical_replicates` |
| `number_of_fractions` | `fractions` |

Fields not in the map are kept under their original name.

The following fields are **excluded from evaluation** entirely:
`phenotype`, `ethnicity`, `mass_analyzer`, `technology_type`

---

## 6. Verdict Definitions

The LLM judge assigns one of three verdicts to each extracted value:

| Verdict | Meaning |
|---|---|
| **high** | Correct type, present in or inferable from the paper, factually accurate, and the full set of extracted values is complete |
| **medium** | Correct type and factually valid, but the full set of extracted values does not fully cover what the paper describes. `corrected value` suggestion is provided. |
| **low** | Hallucinated (absent from paper and not a safe default), type mismatch (value belongs to a different field) or factually incorrect |

> **Note:** The `missing` verdict seen in earlier versions of this README is no longer produced. This pipeline does not use golden SDRF files, so missed fields are tracked only in the coverage CSV (`was_extracted: false`), not as judge verdicts.

### Match Type Labels (derived from verdict + flags)

| Label | Condition |
|---|---|
| `CORRECT` | verdict = high |
| `PARTIAL` | verdict = medium |
| `HALLUCINATED` | `HALLUCINATED: yes` in judge response |
| `TYPE_MISMATCH` | `TYPE_CORRECT: no` in judge response |
| `NO_MATCH` | verdict = low, not hallucinated, not a type mismatch |
| `ERROR` | API call or parsing failure |

---

## 7. Evaluation Logic

### The Sibling Value Rule (Completeness)

Each value is judged for completeness against `all extracted values`, the **full set** of values extracted for that field across all agents not in isolation. If the set collectively covers what the paper describes, every individual value in it is marked complete (`VALUE_COMPLETE: yes`, verdict = `high`).

This prevents penalising correct partial extractions when a field legitimately has multiple values (e.g. `material_type = ["cell line", "primary cells"]`).

### Two-Pass Strategy

**Pass 1** judges all values using the raw (unfiltered) sibling set. This establishes initial verdicts and identifies bad values (hallucinations, type mismatches, wrong values).

**Pass 2** re judges any `medium` verdicts using a *clean* sibling set from which bad values identified in Pass 1 have been pruned. This prevents incorrect sibling values from inflating completeness scores for otherwise good extractions.

### Safe Defaults

Certain values are accepted as correct without needing explicit mention in the paper text:

| Field | Accepted defaults |
|---|---|
| `disease` / `disease_state` | `"normal"`, `"healthy"`, `"no disease"`, `"none"` |
| `label` / `labeling` | `"label-free"`, `"lfq"`, `"none"` |
| `fractions` | `"1"`, `"none"`, `"not applicable"` |
| `replicates` | `"1"` |
| `developmental_stage` | `"adult"`, `"embryonic"`, `"neonatal"`, `"fetal"`, etc. |

### Concentration Fields

For `reduction_concentration` and `alkylation concentration`, a value containing only a numeric quantity and unit (e.g. `"10 mM"`) is considered **complete** even without the reagent name, since the reagent is extracted separately in `reduction reagent` / `alkylation reagent`.

### Always-LLM Fields

`material_type` and `acquisition_method` always invoke the LLM judge (never short-circuited), because their values are frequently inferable from context rather than stated explicitly.

---

## 8. Output Files

### CSV Files (written incrementally — available during execution)

| File | Description |
|---|---|
| `llm_judge_annotation_review.csv` | One row per extracted value with full judge verdicts (main result file) |
| `llm_judge_per_paper.csv` | Per paper counts of correct / hallucinated / mismatched / wrong / incomplete values and overall accuracy |
| `llm_judge_coverage.csv` | One row per agent × pipeline field showing extracted value and whether the field was extracted at all |

### JSON Files (written per paper)

```
json_outputs/<PXD_ID>.json
```

Contains structured per paper results with parsed judge check narratives (TYPE CHECK, SOURCE CHECK, TRUTH CHECK, COMPLETENESS CHECK) split into separate fields.

### Plot Files (generated after all papers complete)

| File | Description |
|---|---|
| `llm_judge_annotation_quality_counts.png` | Stacked bar chart — quality breakdown (correct / hallucinated / mismatch / wrong / incomplete) per paper |
| `llm_judge_accuracy.png` | Per paper accuracy bars with overall mean line and threshold marker |
| `llm_judge_aggregate.png` | Aggregate quality counts and percentages summed across all papers |

---

## 9. Concurrency & Reliability

- Judge calls run in parallel using a `ThreadPoolExecutor` with `MAX_WORKERS=4` threads (overridable via `--workers`).
- Each thread holds its own `GEval` metric instance to avoid shared state.
- API calls retry up to 3 times with a delay of `RETRY_DELAY 8 attempt` seconds between attempts.
- The disk cache and internal call counters are protected by a `threading.Lock`.
