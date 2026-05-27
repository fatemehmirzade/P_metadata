# Proteomics Metadata Extraction Pipeline

Automatically extracts standarized proteomics metadata from scientific papers using LLMs and [DocETL](https://github.com/ucbepic/docetl)

## Pipeline Overview

```
papers (*.txt)
      │
      ▼
 prepare_data.py          ← Split papers into abstract / methods / supplementary
      │
      ▼
 DocETL pipelines (×7)    ← LLM extraction per category (Gemma 4)
      │
      ▼
 merge_and_generate_ann.py ← Merge all outputs into per paper .ann files
      │
      ▼
 filtering.py              ← LLM based annotation filtering
      │
      ▼
 Judge_validation_V_final.py ← GEval / DeepEval LLM as a judge validation
```

The seven DocETL pipelines each extract a different category of proteomics metadata (biological info, MS instruments, sample prep, separation, data analysis, clinical/experimental, factor values, 60+ entity types total). All LLM calls use Gemma 4.

## Input

Place your full text papers as `.txt` files in a folder (default: `text_files_Papers_Ian/`).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install docetl "pyrate-limiter>=3.0,<4.0" deepeval openai matplotlib pandas numpy
```

Set your API keys as environment variables:

```bash
export API_KEY="your-key"
```

## Run

```bash
#1. Prepare input data
python prepare_data.py

#2. Run extraction pipelines
docetl run 01_biological_info.yaml
docetl run 02_ms_instruments.yaml
docetl run 03_sample_prep.yaml
docetl run 04_separation.yaml
docetl run 05_data_analysis.yaml
docetl run 06_clinical_experimental.yaml
docetl run 07_factor_values.yaml

#3. Merge outputs into .ann files
python merge_and_generate_ann.py

#4. Filter annotations
python filtering.py

#5. Validate with LLM as judge
python Judge_validation_V_final.py
```

