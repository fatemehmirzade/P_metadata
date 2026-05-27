# SDRF Benchmark — LLM Judge Evaluation Pipeline

## Overview

This pipeline evaluates the quality of metadata extracted from proteomics papers by LLM based agents (Claude, GPT, Gemini, LLaMA). It uses **Gemma-4-31B-IT** as an LLM judge (via OpenRouter) to assess each extracted value against the original paper text, producing verdicts of `high`, `medium`, or `low`.

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
<BASE_DIR>/                                    # e.g. /Users/fateme/Desktop/Hari_results
│
├── test_set/                                  # One subfolder per dataset (PXD ID)
│   ├── PXD000001/
│   │   ├── manuscript.txt                     # Paper text (title + abstract + methods)
│   │   └── PXD000001.sdrf.tsv                # SDRF golden reference
│   ├── PXD000002/
│   │   ├── manuscript.txt
│   │   └── some_name_sdrf.tsv
│   └── ...
│
├── reports_test_set_claude/                   # Pre-computed tier-match results
│   └── sdrf_benchmark_detailed.csv
├── reports_test_set_gpt/
│   └── sdrf_benchmark_detailed.csv
├── reports_test_set_gemini/
│   └── sdrf_benchmark_detailed.csv
├── reports_test_set_llama/
│   └── sdrf_benchmark_detailed.csv
│
├── <extraction_dir>/                          # Per-model agent extraction outputs
│   ├── BiologicalAgent/
│   │   ├── PXD000001_biological.json
│   │   ├── PXD000002_biological.json
│   │   └── ...
│   ├── TechnicalAgent/
│   │   ├── PXD000001_technical.json
│   │   └── ...
│   └── ExperimentalDesignAgent/
│       ├── PXD000001_experimental.json
│       └── ...
│
└── .prompt_cache_google_gemma-4-31b-it/       # Auto-created by the code (disk cache)
```

---

## 3. How to Run

```bash
python H_Pipeline_Evaluation.py --model claude
```

---

## 4. Output Files

### CSV Files (written incrementally — available during execution)

| File | Description |
|------|-------------|
| `llm_judge_annotation_review.csv` | Per-annotation verdicts (main result file) |
| `llm_judge_per_paper.csv` | Per-paper aggregated statistics |
| `llm_judge_coverage.csv` | Field extraction coverage per agent per paper |

### Plot Files (generated after all papers complete)

| File | Description |
|------|-------------|
| `llm_judge_tier_distribution.png` | 5-tier match distribution bar chart |
| `llm_judge_annotation_quality_counts.png` | Stacked bar — quality breakdown per paper |
| `llm_judge_accuracy.png` | Per-paper accuracy with mean and threshold |
| `llm_judge_aggregate.png` | Aggregate quality across all papers |

---

## 5. Verdict Definitions

The LLM judge assigns one of three verdicts to each extracted value:

| Verdict | Meaning |
|---------|---------|
| **high** | Correct type, present in paper, factually accurate, and complete |
| **medium** | Correct but partially incomplete (e.g., supertype drift, single channel of a multi-channel scheme). A `corrected_value` is provided. |
| **low** | Hallucinated (absent from paper), type mismatch (belongs to a different field), or factually incorrect |
| **missing** | Golden value exists in SDRF but was never extracted by the pipeline |
