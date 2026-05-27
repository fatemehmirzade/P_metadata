import os
import re
import json
import argparse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openai

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

BASE_DIR   = "/Users/fateme/Desktop/Hari_results"
TEST_SET   = os.path.join(BASE_DIR, "test_set")
GOLDEN_DIR = os.path.join(BASE_DIR, "sdrf_golden")

MODEL_RESULT_DIRS = {
    "claude": os.path.join(BASE_DIR, "reports_test_set_claude"),
    "gpt":    os.path.join(BASE_DIR, "reports_test_set_gpt"),
    "gemini": os.path.join(BASE_DIR, "reports_test_set_gemini"),
    "llama":  os.path.join(BASE_DIR, "reports_test_set_llama"),
}

MODEL_EXTRACTION_DIRS = {
    "claude": "/Users/fateme/Desktop/Hari_results/media/volume/bert_training_data_models/llama/codes/extraction_framework/benchmark_data/Final_results/test_claude",
    "gpt":    "/Users/fateme/Desktop/Hari_results/media 3/volume/bert_training_data_models/llama/codes/extraction_framework/benchmark_data/Final_results/test_gpt",
    "gemini": "/Users/fateme/Desktop/Hari_results/media 2/volume/bert_training_data_models/llama/codes/extraction_framework/benchmark_data/Final_results/test_gemini",
    "llama":  "/Users/fateme/Desktop/Hari_results/media 4/volume/bert_training_data_models/llama/codes/extraction_framework/benchmark_data/Final_results/test",
}

AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]

USE_LLM_JUDGE    = True
EVALUATION_MODEL = "gpt-5.2"
LIMIT            = None

AGENT_FILE_SUFFIX = {
    "BiologicalAgent":         "biological",
    "TechnicalAgent":          "technical",
    "ExperimentalDesignAgent": "experimental",
}

SKIP_MISSED_FIELDS = {
    "pxd_id", "dataset_id", "accession", "pride_id", "proteomexchange_id",
    "repository", "doi", "pubmed_id", "pmid",
}

FIELD_TO_AGENT = {
    "species": "BiologicalAgent", "organ": "BiologicalAgent",
    "tissue": "BiologicalAgent", "cell_type": "BiologicalAgent",
    "cell_line": "BiologicalAgent", "disease": "BiologicalAgent",
    "disease_state": "BiologicalAgent", "sex": "BiologicalAgent",
    "age": "BiologicalAgent", "developmental_stage": "BiologicalAgent",
    "ethnicity": "BiologicalAgent", "material_type": "BiologicalAgent",
    "sample_source": "BiologicalAgent", "strain": "BiologicalAgent",
    "BMI": "BiologicalAgent", "anatomic_site_tumor": "BiologicalAgent",
    "instrument": "TechnicalAgent", "cleavage_agent": "TechnicalAgent",
    "label": "TechnicalAgent", "labeling": "TechnicalAgent",
    "fragmentation": "TechnicalAgent", "fragmentation_method": "TechnicalAgent",
    "ptm": "TechnicalAgent", "reduction_reagent": "TechnicalAgent",
    "collision_energy": "TechnicalAgent", "fractionation": "TechnicalAgent",
    "fractionation_method": "TechnicalAgent", "enrichment_method": "TechnicalAgent",
    "acquisition_method": "TechnicalAgent", "alkylation_reagent": "TechnicalAgent",
    "alkylation_concentration": "TechnicalAgent",
    "reduction_concentration": "TechnicalAgent",
    "ionization_type": "TechnicalAgent", "mass_analyzer": "TechnicalAgent",
    "replicates": "ExperimentalDesignAgent",
    "technical_replicates": "ExperimentalDesignAgent",
    "number_of_samples": "ExperimentalDesignAgent",
    "fractions": "ExperimentalDesignAgent",
    "technology_type": "ExperimentalDesignAgent",
    "factor_value": "ExperimentalDesignAgent",
    "experimental_design": "ExperimentalDesignAgent",
    "number_of_biological_replicates": "ExperimentalDesignAgent",
    "number_of_technical_replicates": "ExperimentalDesignAgent",
    "number_of_fractions": "ExperimentalDesignAgent",
    "biological_replicate": "ExperimentalDesignAgent",
    "technical_replicate": "ExperimentalDesignAgent",
}

PIPELINE_FIELDS = {
    "BiologicalAgent": [
        "species", "tissue", "cell_type", "disease_state", "sample_source",
        "age", "anatomic_site_tumor", "BMI", "cell_line", "sex", "strain",
        "material_type", "developmental_stage", "ethnicity",
    ],
    "TechnicalAgent": [
        "acquisition_method", "alkylation_reagent", "alkylation_concentration",
        "cleavage_agent", "collision_energy", "enrichment_method",
        "fractionation_method", "fragmentation_method", "instrument",
        "ionization_type", "labeling", "reduction_reagent", "reduction_concentration",
    ],
    "ExperimentalDesignAgent": [
        "biological_replicate", "technical_replicate", "experimental_design",
        "factor_value", "number_of_fractions", "number_of_technical_replicates",
        "number_of_biological_replicates", "number_of_samples",
    ],
}

FIELD_NAME_MAP = {
    "tissue":                          "organ",
    "disease_state":                   "disease",
    "sample_source":                   "material_type",
    "labeling":                        "label",
    "fragmentation_method":            "fragmentation",
    "fractionation_method":            "fractionation",
    "acquisition_method":              "acquisition_method",
    "alkylation_reagent":              "alkylation_reagent",
    "alkylation_concentration":        "alkylation_concentration",
    "reduction_concentration":         "reduction_concentration",
    "enrichment_method":               "enrichment_method",
    "ionization_type":                 "ionization_type",
    "number_of_biological_replicates": "replicates",
    "biological_replicate":            "replicates",
    "number_of_technical_replicates":  "technical_replicates",
    "technical_replicate":             "technical_replicates",
    "number_of_fractions":             "fractions",
    "number_of_samples":               "number_of_samples",
    "experimental_design":             "experimental_design",
}

FIELD_DEFINITIONS = {
    "species":              "Source organism (e.g. Homo sapiens, Mus musculus).",
    "organ":                "Tissue or organ of origin (e.g. liver, brain cortex, plasma).",
    "cell_type":            "Primary cell type or lineage (e.g. neurons, fibroblasts).",
    "cell_line":            "Name of immortalized cell line (e.g. HEK293T, HeLa).",
    "disease":              "Disease state or diagnosis (e.g. breast cancer, Type 2 diabetes). 'normal' or 'healthy' is correct when subjects are healthy.",
    "sex":                  "Donor sex (e.g. male, female).",
    "age":                  "Age of donor (e.g. 45 years, E14.5 embryo). Ranges like '20-40 years' are acceptable if the paper reports a range.",
    "developmental_stage":  "Developmental stage (e.g. adult, embryonic, early seed development).",
    "ethnicity":            "Donor ancestry or ethnicity (e.g. European, East Asian).",
    "material_type":        "Broad material class: 'tissue', 'cell line', 'primary cells', 'biofluid', etc. Must match what was actually used.",
    "strain":               "Animal/plant strain (e.g. BALB/c, C57BL/6J, Nipponbare).",
    "BMI":                  "Body-Mass Index of donor (kg/m^2).",
    "anatomic_site_tumor":  "Anatomic site of tumor if applicable.",
    "instrument":           "Mass spectrometer make and model (e.g. Thermo Q-Exactive Plus).",
    "cleavage_agent":       "Protease used for digestion (e.g. trypsin, Lys-C).",
    "label":                "Isobaric or metabolic label (e.g. TMT, SILAC, label-free). For SILAC, the golden may list specific isotope channels.",
    "fragmentation":        "Fragmentation method (e.g. HCD, CID, ETD).",
    "ptm":                  "Post-translational modification studied (e.g. phosphorylation).",
    "reduction_reagent":    "Chemical used to reduce disulfides (e.g. DTT, TCEP).",
    "collision_energy":     "Collision energy setting (e.g. normalized collision energy 25).",
    "fractionation":        "Offline fractionation method (e.g. strong anion exchange chromatography).",
    "enrichment_method":    "Enrichment protocol (e.g. TiO2 phosphopeptide enrichment).",
    "acquisition_method":   "MS acquisition scheme (e.g. DDA, DIA, data dependent).",
    "alkylation_reagent":   "Chemical used for alkylation (e.g. iodoacetamide, IAA).",
    "alkylation_concentration": "Concentration of alkylation reagent (e.g. 10 mM).",
    "reduction_concentration":  "Concentration of reduction reagent (e.g. 5 mM).",
    "ionization_type":      "Ionization method (e.g. electrospray, nano-ESI, MALDI).",
    "mass_analyzer":        "Mass analyzer type (e.g. Orbitrap, TOF, quadrupole).",
    "replicates":           "Number of biological replicates (e.g. 3).",
    "technical_replicates": "Number of technical replicates (e.g. 2).",
    "number_of_samples":    "Total sample count (e.g. 12).",
    "fractions":            "Number of fractions generated (e.g. 12).",
    "technology_type":      "Broad technology type (e.g. proteomics).",
    "factor_value":         "Experimental factor or variable being studied (e.g. drug treatment, cell type).",
    "experimental_design":  "Study design type (e.g. time course, cross-sectional, case-control).",
}

ANNOTATION_CRITERIA = """You are an expert in proteomics and mass-spectrometry experimental metadata.

Evaluate ONE predicted annotation extracted from a proteomics paper.

INPUT LAYOUT:
  - 'actual output'    = the field name and the extracted value.
  - 'expected output'  = the golden (ground truth) value if one exists,
                         OR "NO GOLDEN - evaluate against source text only."
  - 'context'          = the source paper text (abstract + methods) and the
                         definition of this specific field type.

EVALUATION RULES:

1. VALUE CORRECTNESS:
   - If golden EXISTS: prediction is CORRECT if it matches the golden value.
     Synonyms, abbreviations, partial matches that unambiguously identify the
     same entity = CORRECT. Digit vs word ("3" vs "three") = CORRECT.
     "SILAC" vs "Lys8/Arg10 (heavy); Lys4/Arg6 (medium)" = CORRECT
     (SILAC is the method; specific channels are detail level differences).
   - If NO golden: prediction is CORRECT if supported by the source text
     and fits the field definition. Do NOT penalise just because no golden exists.
   - "unknown" is correct when the field is genuinely absent from source text.
   - "unknown" when source text DOES mention the field = WRONG.

2. TYPE MISMATCH:
   Does the value fit this field's definition, or does it clearly belong to a
   different metadata field? Only flag if clearly wrong field.

3. COMPLETENESS (lenient):
   Flag as incomplete ONLY when the value is too generic to identify the entity,
   a critical qualifier is missing that changes identity, or it is ambiguous.
   Extracting "seed" when golden says "pistil; seed" = INCOMPLETE, not wrong.
   Extracting "Nipponbare" when golden says "Nipponbare (Oryza sativa L.)" = COMPLETE.

4. HALLUCINATION:
   The value is a hallucination ONLY if it cannot be found in or reasonably
   inferred from the source text. These are NOT hallucinations:
   - Valid abbreviation expansions (HCD for higher energy collisional dissociation)
   - Morphological variants (tryptic for trypsin)
   - Standard synonyms (IAA for iodoacetamide)
   - "normal" or "healthy" for disease field when subjects are not diseased

SCORING:
  10 = Correct, complete, non-hallucinated, matches golden if one exists.
   8 = Correct value but minor completeness issue (missing secondary detail).
   5 = Partially correct - captures some truth but golden mismatch or too vague.
   3 = Wrong value or clear type mismatch, but at least exists in source text.
   0 = Hallucinated - value NOT found in source text at all."""


ANNOTATION_STEPS = [
    "Read the source text in 'context' and locate any mention of this field.",
    "Compare the extracted value (in 'actual output') against the golden value "
    "(in 'expected output') if one exists - synonyms, abbreviations, "
    "and equivalent representations count as matching.",
    "If no golden exists, check whether the extracted value is supported by "
    "the source text and fits the field definition.",
    "Check for hallucination: can this value be found in or reasonably inferred "
    "from the source text? If not, score 0.",
    "Assign a score 0-10 based on correctness, completeness, and hallucination status.",
]


class GPT52MediumReasoning(DeepEvalBaseLLM):

    def __init__(self):
        self._model_name = "gpt-5.2"
        self._client = None

    def load_model(self):
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def generate(self, prompt: str, schema=None) -> str:
        client = self.load_model()
        call_kwargs = {
            "model":                self._model_name,
            "messages":             [{"role": "user", "content": prompt}],
            "reasoning_effort":     "medium",
            "max_completion_tokens": 16000,
        }
        response = client.chat.completions.create(**call_kwargs)
        content  = response.choices[0].message.content or ""
        if schema is not None:
            try:
                raw = content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw.strip())
                return schema.model_validate_json(raw)
            except Exception:
                return content
        return content

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return "gpt-5.2-medium-reasoning"


MISSED_CRITERIA = """You are an expert in proteomics and mass-spectrometry experimental metadata.

Determine whether the source text contains extractable information for a
specific metadata field that was NOT extracted by the pipeline.

INPUT LAYOUT:
  - 'actual output'   = states which field was NOT extracted.
  - 'expected output'  = "Check if source text contains this field."
  - 'context'          = the source paper text and the field definition.

RULES:
  - Score 10 if the source text CLEARLY and EXPLICITLY mentions a value for
    this field (e.g. source says "trypsin digestion" -> cleavage_agent is present).
  - Score 7 if the value can be REASONABLY INFERRED from context
    (e.g. "Q Exactive" implies electrospray ionization).
  - Score 3 if there is only a vague or ambiguous mention.
  - Score 0 if the field is genuinely NOT mentioned or applicable to this paper."""


MISSED_STEPS = [
    "Read the source text in 'context' carefully.",
    "Search for any mention of the metadata field described in 'actual output'.",
    "Determine if a concrete, extractable value exists for this field in the source text.",
    "Score 10 if clearly present, 7 if inferable, 3 if vague, 0 if absent.",
]


_gpt52_model = None
_geval_metric = None
_missed_metric = None

def get_judge_model():
    global _gpt52_model
    if _gpt52_model is None:
        _gpt52_model = GPT52MediumReasoning()
    return _gpt52_model


def get_missed_metric():
    global _missed_metric
    if _missed_metric is None:
        _missed_metric = GEval(
            name="MissedFieldDetector",
            criteria=MISSED_CRITERIA,
            evaluation_steps=MISSED_STEPS,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=get_judge_model(),
            threshold=0.5,
        )
    return _missed_metric


def get_annotation_metric():
    global _geval_metric
    if _geval_metric is None:
        _geval_metric = GEval(
            name="AnnotationJudge",
            criteria=ANNOTATION_CRITERIA,
            evaluation_steps=ANNOTATION_STEPS,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=get_judge_model(),
            threshold=0.5,
        )
    return _geval_metric


def judge_single_annotation(source_text: str,
                             field_name: str,
                             extracted_value: str,
                             golden_value: str) -> dict:
    field_def = FIELD_DEFINITIONS.get(field_name, f"Metadata field: {field_name}")

    actual_output = f"Field: {field_name}\nExtracted value: {extracted_value}"

    if golden_value and golden_value.strip():
        expected_output = f"Golden value: {golden_value}"
    else:
        expected_output = "NO GOLDEN - evaluate against source text only."

    context = [
        source_text,
        f"FIELD DEFINITION for '{field_name}': {field_def}",
    ]

    test_case = LLMTestCase(
        input=f"Evaluate extraction of '{field_name}' = '{extracted_value}'",
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
    )

    metric = get_annotation_metric()

    try:
        metric.measure(test_case)
        score  = metric.score
        reason = metric.reason or ""
    except Exception as e:
        print(f"          [GEval] Error on {field_name}: {type(e).__name__}: {e}")
        return {
            "value_correct": None, "value_complete": None,
            "hallucination": None, "type_mismatch": None,
            "issue_summary": f"GEval error: {e}", "geval_score": None,
        }

    if score >= 0.8:
        correct, complete, halluc = True, True, False
    elif score >= 0.5:
        correct, complete, halluc = True, False, False
    elif score >= 0.2:
        correct, complete, halluc = False, False, False
    else:
        correct, complete, halluc = False, False, True

    type_mismatch = False
    reason_lower = reason.lower()
    if "type mismatch" in reason_lower or "wrong field" in reason_lower:
        type_mismatch = True

    return {
        "value_correct":  correct,
        "value_complete":  complete,
        "hallucination":   halluc,
        "type_mismatch":   type_mismatch,
        "issue_summary":   f"GEval={score:.2f}. {reason[:250]}",
        "geval_score":     score,
    }


def check_missed_field(source_text: str, field_name: str) -> float:
    field_def = FIELD_DEFINITIONS.get(field_name, f"Metadata field: {field_name}")

    test_case = LLMTestCase(
        input=f"Is '{field_name}' present in the source text?",
        actual_output=f"Field '{field_name}' was NOT extracted by the pipeline.",
        expected_output="Check if source text contains this field.",
        context=[
            source_text,
            f"FIELD DEFINITION for '{field_name}': {field_def}",
        ],
    )

    metric = get_missed_metric()
    try:
        metric.measure(test_case)
        return metric.score
    except Exception as e:
        print(f"          [missed] Error on {field_name}: {e}")
        return 0.0


def load_manuscript(pxd_id: str) -> str:
    path = os.path.join(TEST_SET, pxd_id, "manuscript.txt")
    if not os.path.exists(path):
        print(f"    WARNING: manuscript not found: {path}")
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
        kept_parts = []
        for section in ["TITLE", "ABSTRACT", "METHODS"]:
            pattern = rf"===\s*{section}\s*===(.*?)(?===\s*[A-Z]|\Z)"
            match   = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            if match:
                kept_parts.append(f"=== {section} ===\n{match.group(1).strip()}")
        if kept_parts:
            result = "\n\n".join(kept_parts)
            print(f"    Manuscript: {len(full_text):,} chars total, "
                  f"using {len(result):,} chars (title+abstract+methods)")
            return result
        else:
            print(f"    Manuscript: {len(full_text):,} chars (full text, no sections found)")
            return full_text
    except Exception as e:
        print(f"    WARNING: could not read manuscript {path}: {e}")
        return ""


def load_llm_output_json(extraction_dir: str, pxd_id: str, agent: str) -> dict:
    suffix    = AGENT_FILE_SUFFIX.get(agent, agent.lower())
    agent_dir = os.path.join(extraction_dir, agent)
    if not os.path.exists(agent_dir):
        print(f"    WARNING: agent dir not found: {agent_dir}")
        return {}

    raw_data = None
    candidates = [f"{pxd_id}_{suffix}.json", f"{pxd_id}.json"]
    for cname in candidates:
        path = os.path.join(agent_dir, cname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                print(f"    [{agent}] loaded: {cname}")
                break
            except Exception as e:
                print(f"    WARNING: could not read {path}: {e}")
                return {}

    if raw_data is None:
        for fname in sorted(os.listdir(agent_dir)):
            if fname.startswith(pxd_id) and fname.endswith(".json"):
                path = os.path.join(agent_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    print(f"    [{agent}] loaded (fallback): {fname}")
                    break
                except Exception as e:
                    print(f"    WARNING: could not read {path}: {e}")
                    return {}

    if raw_data is None:
        print(f"    [{agent}] WARNING: no JSON found for {pxd_id}")
        return {}

    allowed_fields = set()
    for fields_list in PIPELINE_FIELDS.values():
        for f in fields_list:
            allowed_fields.add(f)
            allowed_fields.add(FIELD_NAME_MAP.get(f, f))

    parsed = {}
    for field, val in raw_data.items():
        if field.startswith("_"):
            continue
        if field not in allowed_fields:
            continue
        if isinstance(val, list) and len(val) >= 1:
            extracted = str(val[0]).strip()
        elif isinstance(val, str):
            extracted = val.strip()
        else:
            extracted = str(val).strip()
        if not extracted:
            continue
        mapped_field = FIELD_NAME_MAP.get(field, field)
        parsed[mapped_field] = extracted
    return parsed


def load_golden_json(pxd_id: str, agent: str) -> dict:
    fname = f"{pxd_id}_{agent}_golden.json"
    path  = os.path.join(GOLDEN_DIR, fname)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "fields" in raw:
            fields = raw["fields"]
        else:
            fields = raw
        result = {}
        for k, v in fields.items():
            if v is None:
                continue
            v_str = str(v).strip()
            if v_str and v_str.lower() not in ("nan", "none", "null", ""):
                result[k] = v_str
        return result
    except Exception as e:
        print(f"    WARNING: could not read golden {path}: {e}")
        return {}


def expand_match_type(match_type: str) -> dict:
    mt           = (match_type or "").upper().strip()
    exact        = (mt == "EXACT")
    normalized   = (mt == "NORMALIZED")
    ontology     = (mt == "ONTOLOGY")
    hierarchical = (mt == "HIERARCHICAL")
    semantic     = (mt == "SEMANTIC")
    matched      = any([exact, normalized, ontology, hierarchical, semantic])
    return {
        "tier1_exact":        exact,
        "tier2_normalized":   normalized,
        "tier3_ontology":     ontology,
        "tier4_hierarchical": hierarchical,
        "tier5_semantic":     semantic,
        "any_match":          matched,
        "no_match":           not matched,
    }


def load_detailed_csv(model_result_dir: str) -> pd.DataFrame:
    path = os.path.join(model_result_dir, "sdrf_benchmark_detailed.csv")
    if not os.path.exists(path):
        print(f"  ERROR: not found - {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"agent", "pxd_id", "field", "golden", "llm", "match_type", "score"}
    missing  = required - set(df.columns)
    if missing:
        print(f"  ERROR: missing columns {missing}")
        return pd.DataFrame()
    df["llm"]        = df["llm"].fillna("").astype(str)
    df["golden"]     = df["golden"].fillna("").astype(str)
    df["match_type"] = df["match_type"].fillna("NO_MATCH").astype(str)
    df["score"]      = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df


def process_model(model_name: str,
                  model_result_dir: str,
                  extraction_dir: str) -> pd.DataFrame:
    detailed_df = load_detailed_csv(model_result_dir)
    if detailed_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Loaded {len(detailed_df)} rows from sdrf_benchmark_detailed.csv")

    pxd_ids = sorted(detailed_df["pxd_id"].unique())
    print(f"  Papers: {len(pxd_ids)}")

    if LIMIT and LIMIT < len(pxd_ids):
        pxd_ids = pxd_ids[:LIMIT]
        print(f"  Limit applied: running only first {LIMIT} papers: {pxd_ids}")

    all_rows = []
    all_coverage_rows = []

    for pxd_id in pxd_ids:
        print(f"\n  --- {pxd_id} ---")

        source_text = load_manuscript(pxd_id)

        all_predicted = {}
        all_golden    = {}
        for agent in AGENTS:
            pred = load_llm_output_json(extraction_dir, pxd_id, agent)
            gold = load_golden_json(pxd_id, agent)
            if pred:
                all_predicted[agent] = pred
            if gold:
                all_golden[agent] = gold

        pxd_df = detailed_df[detailed_df["pxd_id"] == pxd_id].copy()

        flat_golden_for_coverage = {}
        for agent in AGENTS:
            g = load_golden_json(pxd_id, agent)
            flat_golden_for_coverage.update(g)

        coverage_rows = []
        for agent in AGENTS:
            pred = all_predicted.get(agent, {})
            for raw_field in PIPELINE_FIELDS.get(agent, []):
                mapped = FIELD_NAME_MAP.get(raw_field, raw_field)
                extracted = pred.get(mapped) or pred.get(raw_field)
                golden_val = (flat_golden_for_coverage.get(mapped)
                              or flat_golden_for_coverage.get(raw_field))
                coverage_rows.append({
                    "paper_id":         pxd_id,
                    "agent":            agent,
                    "pipeline_field":   raw_field,
                    "mapped_field":     mapped,
                    "extracted_value":  extracted or "",
                    "was_extracted":    bool(extracted and str(extracted).strip()
                                            and str(extracted).strip().lower() != "unknown"),
                    "has_golden":       bool(golden_val),
                    "golden_value":     golden_val or "",
                })
        all_coverage_rows.extend(coverage_rows)

        n_extracted  = sum(1 for r in coverage_rows if r["was_extracted"])
        n_has_golden = sum(1 for r in coverage_rows if r["has_golden"])
        n_both       = sum(1 for r in coverage_rows if r["was_extracted"] and r["has_golden"])
        total_fields = len(coverage_rows)
        print(f"    Coverage: {n_extracted}/{total_fields} fields extracted  "
              f"| {n_has_golden}/{total_fields} fields have golden  "
              f"| {n_both}/{total_fields} overlap")

        tier_rows = []

        for _, row in pxd_df.iterrows():
            tier = expand_match_type(row["match_type"])
            tier_rows.append({
                "paper_id":           pxd_id,
                "agent":              row["agent"],
                "annotation_type":    row["field"],
                "extracted_value":    row["llm"],
                "golden_value":       row["golden"],
                "match_score":        round(float(row["score"]), 4),
                "match_type":         row["match_type"],
                "tier1_exact":        tier["tier1_exact"],
                "tier2_normalized":   tier["tier2_normalized"],
                "tier3_ontology":     tier["tier3_ontology"],
                "tier4_hierarchical": tier["tier4_hierarchical"],
                "tier5_semantic":     tier["tier5_semantic"],
                "any_match":          tier["any_match"],
                "no_match":           tier["no_match"],
                "has_golden":         True,
                "type_mismatch":      None,
                "correct_type":       None,
                "value_correct":      None,
                "value_complete":     None,
                "hallucination":      None,
                "source_evidence":    "",
                "issue_summary":      "",
                "corrected_value":    None,
                "row_type":           "sdrf_match",
            })

        sdrf_fields_this_paper = set(pxd_df["field"].tolist())
        for agent, pred_dict in all_predicted.items():
            for field, val in pred_dict.items():
                if field in sdrf_fields_this_paper:
                    continue
                if str(val).strip().lower() in ("unknown", ""):
                    continue
                golden_val = flat_golden_for_coverage.get(field, "")
                tier_rows.append({
                    "paper_id":           pxd_id,
                    "agent":              agent,
                    "annotation_type":    field,
                    "extracted_value":    val,
                    "golden_value":       golden_val,
                    "match_score":        None,
                    "match_type":         "NOT_IN_SDRF",
                    "tier1_exact":        None,
                    "tier2_normalized":   None,
                    "tier3_ontology":     None,
                    "tier4_hierarchical": None,
                    "tier5_semantic":     None,
                    "any_match":          None,
                    "no_match":           None,
                    "has_golden":         bool(golden_val),
                    "type_mismatch":      None,
                    "correct_type":       None,
                    "value_correct":      None,
                    "value_complete":     None,
                    "hallucination":      None,
                    "source_evidence":    "",
                    "issue_summary":      "",
                    "corrected_value":    None,
                    "row_type":           "judge_only",
                })

        if USE_LLM_JUDGE and source_text:
            n_annotations = len(tier_rows)
            print(f"    [judge] Running GEval on {n_annotations} annotations...")

            for idx, tr in enumerate(tier_rows):
                field_name      = tr["annotation_type"]
                extracted_value = tr["extracted_value"]
                golden_value    = tr["golden_value"] or ""

                if not extracted_value or extracted_value.strip().lower() == "unknown":
                    tr["value_correct"]  = None
                    tr["value_complete"] = None
                    tr["hallucination"]  = None
                    tr["type_mismatch"]  = None
                    tr["issue_summary"]  = "Skipped (unknown/empty)"
                    continue

                if tr.get("tier1_exact"):
                    tr["value_correct"]  = True
                    tr["value_complete"] = True
                    tr["hallucination"]  = False
                    tr["type_mismatch"]  = False
                    tr["issue_summary"]  = "Exact match - skipped GEval"
                    print(f"        [{idx+1}/{n_annotations}] {field_name}: "
                          f"EXACT match - skip")
                    continue

                print(f"        [{idx+1}/{n_annotations}] {field_name}: "
                      f"'{extracted_value[:40]}' vs golden='{golden_value[:40]}'",
                      end="  ")

                result = judge_single_annotation(
                    source_text, field_name, extracted_value, golden_value)

                tr["value_correct"]  = result["value_correct"]
                tr["value_complete"] = result["value_complete"]
                tr["hallucination"]  = result["hallucination"]
                tr["type_mismatch"]  = result["type_mismatch"]
                tr["issue_summary"]  = result["issue_summary"]

                geval_s = result.get("geval_score")
                if geval_s is not None:
                    print(f"-> score={geval_s:.2f}  "
                          f"correct={result['value_correct']}  "
                          f"halluc={result['hallucination']}")
                else:
                    print(f"-> ERROR")

        if USE_LLM_JUDGE and source_text:
            extracted_fields = set()
            for tr in tier_rows:
                ev = tr["extracted_value"]
                if ev and str(ev).strip() and str(ev).strip().lower() != "unknown":
                    extracted_fields.add(tr["annotation_type"].lower())

            all_golden_fields = {}
            for agent in AGENTS:
                g = load_golden_json(pxd_id, agent)
                for k, v in g.items():
                    all_golden_fields[k] = (v, agent)

            missed_rows = []

            for field, (golden_val, agent) in all_golden_fields.items():
                if field.lower() in extracted_fields:
                    continue
                if field.lower() in SKIP_MISSED_FIELDS:
                    continue
                print(f"        [missed] {field}: golden exists but not extracted - MISSING")
                missed_rows.append({
                    "paper_id":           pxd_id,
                    "agent":              agent,
                    "annotation_type":    field,
                    "extracted_value":    "MISSING",
                    "golden_value":       golden_val,
                    "match_score":        0.0,
                    "match_type":         "NO_MATCH",
                    "tier1_exact":        False,
                    "tier2_normalized":   False,
                    "tier3_ontology":     False,
                    "tier4_hierarchical": False,
                    "tier5_semantic":     False,
                    "any_match":          False,
                    "no_match":           True,
                    "has_golden":         True,
                    "type_mismatch":      False,
                    "correct_type":       None,
                    "value_correct":      False,
                    "value_complete":     False,
                    "hallucination":      False,
                    "source_evidence":    "",
                    "issue_summary":      "Not extracted - golden value exists",
                    "corrected_value":    golden_val,
                    "row_type":           "missed",
                })

            golden_field_set = set(k.lower() for k in all_golden_fields)
            for agent in AGENTS:
                for raw_field in PIPELINE_FIELDS.get(agent, []):
                    mapped = FIELD_NAME_MAP.get(raw_field, raw_field)
                    if mapped.lower() in extracted_fields:
                        continue
                    if raw_field.lower() in extracted_fields:
                        continue
                    if mapped.lower() in golden_field_set:
                        continue
                    if mapped.lower() in SKIP_MISSED_FIELDS:
                        continue

                    score = check_missed_field(source_text, mapped)
                    if score >= 0.5:
                        sdrf_agent = FIELD_TO_AGENT.get(mapped, agent)
                        print(f"        [missed] {mapped}: GEval={score:.2f} "
                              f"- present in source but not extracted")
                        missed_rows.append({
                            "paper_id":           pxd_id,
                            "agent":              sdrf_agent,
                            "annotation_type":    mapped,
                            "extracted_value":    "MISSING",
                            "golden_value":       "",
                            "match_score":        0.0,
                            "match_type":         "NO_MATCH",
                            "tier1_exact":        False,
                            "tier2_normalized":   False,
                            "tier3_ontology":     False,
                            "tier4_hierarchical": False,
                            "tier5_semantic":     False,
                            "any_match":          False,
                            "no_match":           True,
                            "has_golden":         False,
                            "type_mismatch":      False,
                            "correct_type":       None,
                            "value_correct":      False,
                            "value_complete":     False,
                            "hallucination":      False,
                            "source_evidence":    "",
                            "issue_summary":      f"Not extracted - GEval={score:.2f} "
                                                  f"confirms present in source text",
                            "corrected_value":    None,
                            "row_type":           "missed",
                        })
                    else:
                        print(f"        [missed] {mapped}: GEval={score:.2f} "
                              f"- not in source (OK to skip)")

            if missed_rows:
                print(f"    [missed] Added {len(missed_rows)} MISSING rows")
                tier_rows.extend(missed_rows)

        all_rows.extend(tier_rows)

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    coverage_df = pd.DataFrame(all_coverage_rows) if all_coverage_rows else pd.DataFrame()
    df = pd.DataFrame(all_rows)

    final_cols = [
        "paper_id", "agent", "annotation_type", "extracted_value",
        "golden_value", "match_score", "match_type",
        "tier1_exact", "tier2_normalized", "tier3_ontology",
        "tier4_hierarchical", "tier5_semantic", "any_match", "no_match",
        "has_golden",
        "type_mismatch", "correct_type", "value_correct", "value_complete",
        "hallucination", "source_evidence", "issue_summary", "corrected_value",
    ]
    for c in final_cols:
        if c not in df.columns:
            df[c] = None
    return df[final_cols], coverage_df


def compute_per_paper_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    records = []
    for paper_id in sorted(df["paper_id"].unique()):
        pxd_df  = df[df["paper_id"] == paper_id]
        pred_df = pxd_df[pxd_df["extracted_value"] != "MISSING"]
        miss_df = pxd_df[pxd_df["extracted_value"] == "MISSING"]
        total   = len(pred_df)

        rec = {
            "paper_id":        paper_id,
            "total_predicted": total,
            "total_missed":    len(miss_df),
            "exact":           int(pred_df["tier1_exact"].sum()),
            "normalized":      int(pred_df["tier2_normalized"].sum()),
            "ontology":        int(pred_df["tier3_ontology"].sum()),
            "hierarchical":    int(pred_df["tier4_hierarchical"].sum()),
            "semantic":        int(pred_df["tier5_semantic"].sum()),
            "no_match":        int(pred_df["no_match"].sum()),
            "match_rate":      round(float(pred_df["any_match"].mean()), 4) if total > 0 else 0.0,
            "avg_score":       round(float(pred_df["match_score"].dropna().mean()), 4)
                               if pred_df["match_score"].notna().any() else 0.0,
        }

        if USE_LLM_JUDGE and pred_df["value_correct"].notna().any():
            n_correct = int(pred_df[
                (pred_df["type_mismatch"] == False) &
                (pred_df["value_correct"]  == True)  &
                (pred_df["value_complete"] == True)  &
                (pred_df["hallucination"]  == False)
            ].shape[0])
            n_halluc     = int(pred_df["hallucination"].fillna(False).sum())
            n_mismatch   = int(pred_df["type_mismatch"].fillna(False).sum())
            n_wrong      = int(pred_df[
                (pred_df["value_correct"] == False) &
                (pred_df["hallucination"] == False)
            ].shape[0])
            n_incomplete = int(pred_df[
                (pred_df["value_correct"]  == True) &
                (pred_df["value_complete"] == False)
            ].shape[0])

            rec["judge_n_correct"]      = n_correct
            rec["judge_n_hallucinated"] = n_halluc
            rec["judge_n_mismatch"]     = n_mismatch
            rec["judge_n_wrong"]        = n_wrong
            rec["judge_n_incomplete"]   = n_incomplete
            rec["judge_n_missed"]       = len(miss_df)
            rec["judge_accuracy"]       = round(n_correct / total, 4) if total > 0 else 0.0

        records.append(rec)

    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, per_paper_df: pd.DataFrame,
                 model_name: str, out_dir: str):
    BG       = "white"
    GRID_C   = "#E5E7EB"
    TEXT_C   = "#111827"
    SUB_C    = "#6B7280"
    SPINE_C  = "#D1D5DB"

    CLR_CORRECT    = "#27ae60"
    CLR_HALL       = "#e74c3c"
    CLR_MISMATCH   = "#9b59b6"
    CLR_WRONG      = "#e67e22"
    CLR_INCOMPLETE = "#3498db"
    CLR_MISSED     = "#95a5a6"

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.yaxis.grid(True, color=GRID_C, lw=0.8, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(SPINE_C)
        ax.tick_params(colors=TEXT_C, labelsize=9)

    pred_df     = df[df["extracted_value"] != "MISSING"]
    tier_cols   = ["tier1_exact", "tier2_normalized", "tier3_ontology",
                   "tier4_hierarchical", "tier5_semantic", "no_match"]
    tier_labels = ["Exact", "Normalized", "Ontology", "Hierarchical", "Semantic", "No Match"]
    tier_colors = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899", "#EF4444"]
    tier_totals = [int(pred_df[c].sum()) for c in tier_cols]
    grand       = sum(tier_totals)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    style_ax(ax)
    bars = ax.bar(tier_labels, tier_totals, color=tier_colors,
                  edgecolor="white", linewidth=1.0, alpha=0.9, zorder=3)
    for bar, val in zip(bars, tier_totals):
        pct = val / grand * 100 if grand > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f"{val}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT_C)
    ax.set_ylabel("Count", fontsize=11, color=SUB_C)
    ax.set_title(f"{model_name.upper()} - 5-Tier Match Distribution "
                 f"({grand} total annotations)",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_tier_distribution.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: llm_judge_tier_distribution.png")

    if per_paper_df.empty or "judge_accuracy" not in per_paper_df.columns:
        return

    required = ["judge_n_correct", "judge_n_hallucinated", "judge_n_mismatch",
                "judge_n_wrong", "judge_n_incomplete", "judge_n_missed"]
    if not all(c in per_paper_df.columns for c in required):
        return

    names        = per_paper_df["paper_id"].tolist()
    x            = np.arange(len(names))
    w            = 0.6
    n_correct    = per_paper_df["judge_n_correct"].fillna(0).values
    n_hall       = per_paper_df["judge_n_hallucinated"].fillna(0).values
    n_mismatch   = per_paper_df["judge_n_mismatch"].fillna(0).values
    n_wrong      = per_paper_df["judge_n_wrong"].fillna(0).values
    n_incomplete = per_paper_df["judge_n_incomplete"].fillna(0).values
    n_missed     = per_paper_df["judge_n_missed"].fillna(0).values
    totals       = per_paper_df["total_predicted"].fillna(0).values

    fig, ax = plt.subplots(figsize=(max(16, len(names) * 0.7), 7), facecolor=BG)
    style_ax(ax)
    ax.bar(x, n_correct, w, label="Correct", color=CLR_CORRECT, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, n_hall, w, bottom=n_correct,
           label="Hallucinated", color=CLR_HALL, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, n_mismatch, w, bottom=n_correct+n_hall,
           label="Type Mismatch", color=CLR_MISMATCH, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, n_wrong, w, bottom=n_correct+n_hall+n_mismatch,
           label="Wrong Value", color=CLR_WRONG, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, n_incomplete, w, bottom=n_correct+n_hall+n_mismatch+n_wrong,
           label="Incomplete", color=CLR_INCOMPLETE, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, n_missed, w, bottom=n_correct+n_hall+n_mismatch+n_wrong+n_incomplete,
           label="Missed", color=CLR_MISSED, edgecolor="white", lw=0.5, alpha=0.92)
    grand_per_bar = n_correct + n_hall + n_mismatch + n_wrong + n_incomplete + n_missed
    for i, (gpb, cor) in enumerate(zip(grand_per_bar, n_correct)):
        ax.text(i, gpb + 0.3,
                f"predicted={int(totals[i])}  missed={int(n_missed[i])}",
                ha="center", va="bottom", fontsize=7, fontweight="bold", color=TEXT_C)
        if cor >= 2:
            ax.text(i, cor / 2, str(int(cor)), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TEXT_C)
    ax.set_ylabel("Number of Annotations", fontsize=11, color=SUB_C)
    ax.set_title(f"{model_name.upper()} - Annotation Quality per Paper (raw counts)",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=SPINE_C, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_annotation_quality_counts.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: llm_judge_annotation_quality_counts.png")

    grand_pp = n_correct + n_hall + n_mismatch + n_wrong + n_incomplete + n_missed
    denom    = np.where(grand_pp == 0, 1, grand_pp)
    p_correct    = n_correct    / denom * 100
    p_hall       = n_hall       / denom * 100
    p_mismatch   = n_mismatch   / denom * 100
    p_wrong      = n_wrong      / denom * 100
    p_incomplete = n_incomplete / denom * 100
    p_missed     = n_missed     / denom * 100

    fig, ax = plt.subplots(figsize=(max(16, len(names) * 0.7), 7), facecolor=BG)
    style_ax(ax)
    ax.bar(x, p_correct, w, label="Correct", color=CLR_CORRECT, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, p_hall, w, bottom=p_correct,
           label="Hallucinated", color=CLR_HALL, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, p_mismatch, w, bottom=p_correct+p_hall,
           label="Type Mismatch", color=CLR_MISMATCH, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, p_wrong, w, bottom=p_correct+p_hall+p_mismatch,
           label="Wrong Value", color=CLR_WRONG, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, p_incomplete, w, bottom=p_correct+p_hall+p_mismatch+p_wrong,
           label="Incomplete", color=CLR_INCOMPLETE, edgecolor="white", lw=0.5, alpha=0.92)
    ax.bar(x, p_missed, w, bottom=p_correct+p_hall+p_mismatch+p_wrong+p_incomplete,
           label="Missed", color=CLR_MISSED, edgecolor="white", lw=0.5, alpha=0.92)
    for i, pct in enumerate(p_correct):
        if pct >= 6:
            ax.text(i, pct / 2, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TEXT_C)
    ax.set_ylabel("% of Annotations", fontsize=11, color=SUB_C)
    ax.set_ylim(0, 115)
    ax.axhline(100, color=SPINE_C, lw=0.8, linestyle="--")
    ax.set_title(f"{model_name.upper()} - Annotation Quality per Paper (normalised %)",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=SPINE_C, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_annotation_quality_pct.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: llm_judge_annotation_quality_pct.png")

    acc_vals = per_paper_df["judge_accuracy"].tolist()
    overall  = float(np.mean(acc_vals))

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.55), 6), facecolor=BG)
    style_ax(ax)
    bar_colors = ["#27ae60" if v >= 0.7 else "#e67e22" if v >= 0.5 else "#e74c3c"
                  for v in acc_vals]
    bars = ax.bar(x, acc_vals, color=bar_colors, alpha=0.88,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.012,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=TEXT_C)
    ax.axhline(overall, color="#2471a3", lw=2, linestyle="--",
               label=f"Mean = {overall:.3f}")
    ax.axhline(0.7, color="#e74c3c", lw=1.5, linestyle=":", alpha=0.7,
               label="Threshold = 0.70")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TEXT_C)
    ax.set_ylabel("Accuracy (Correct / Total Predicted)", fontsize=11, color=SUB_C)
    ax.set_ylim([0, 1.18])
    ax.set_title(f"{model_name.upper()} - LLM Judge Accuracy per Paper (GEval)",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=SPINE_C)
    ax.text(0.98, 0.97, f"Overall: {overall:.1%}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color="#27ae60",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                      edgecolor="#27ae60", alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_accuracy.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: llm_judge_accuracy.png")

    agg_labels = ["Correct", "Hallucinated", "Type Mismatch",
                  "Wrong Value", "Incomplete", "Missed"]
    agg_vals   = [int(n_correct.sum()), int(n_hall.sum()), int(n_mismatch.sum()),
                  int(n_wrong.sum()),   int(n_incomplete.sum()), int(n_missed.sum())]
    agg_colors = [CLR_CORRECT, CLR_HALL, CLR_MISMATCH,
                  CLR_WRONG, CLR_INCOMPLETE, CLR_MISSED]
    grand_total = sum(agg_vals)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    style_ax(ax)
    bars5 = ax.bar(agg_labels, agg_vals, color=agg_colors,
                   edgecolor="white", linewidth=1.0, alpha=0.92, zorder=3)
    for bar, val in zip(bars5, agg_vals):
        pct = val / grand_total * 100 if grand_total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_C)
    overall_acc = agg_vals[0] / grand_total * 100 if grand_total > 0 else 0
    ax.set_ylabel("Total Count (all papers)", fontsize=11, color=SUB_C)
    ax.set_title(f"{model_name.upper()} - Aggregate Annotation Quality "
                 f"(total = {grand_total})",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.text(0.5, 0.97, f"Overall accuracy: {overall_acc:.1f}%",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold", color=CLR_CORRECT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                      edgecolor=CLR_CORRECT, alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_aggregate.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: llm_judge_aggregate.png")


def main():
    parser = argparse.ArgumentParser(
        description="5-tier SDRF matching + per-annotation LLM-as-judge via DeepEval GEval")
    parser.add_argument("--model",    required=True,
                        choices=list(MODEL_RESULT_DIRS.keys()),
                        help="Model: claude / gpt / gemini / llama")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge (5-tier columns only)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Limit to first N papers (for testing)")
    args = parser.parse_args()

    global USE_LLM_JUDGE, LIMIT
    if args.no_judge:
        USE_LLM_JUDGE = False
    if args.limit:
        LIMIT = args.limit

    model_name       = args.model
    model_result_dir = MODEL_RESULT_DIRS[model_name]
    extraction_dir   = MODEL_EXTRACTION_DIRS[model_name]
    out_dir          = os.path.join(BASE_DIR, f"evaluation_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"  MODEL          : {model_name.upper()}")
    print(f"  Results dir    : {model_result_dir}")
    print(f"  Extraction dir : {extraction_dir}")
    print(f"  Manuscripts    : {TEST_SET}/<PXD>/manuscript.txt")
    print(f"  Golden dir     : {GOLDEN_DIR}")
    print(f"  Output folder  : {out_dir}")
    print(f"  LLM judge      : {'ENABLED via DeepEval GEval (per-annotation)  model=gpt-5.2 reasoning=medium' if USE_LLM_JUDGE else 'DISABLED'}")
    print(f"  Limit          : {LIMIT if LIMIT else 'ALL papers'}")

    df, coverage_df = process_model(model_name, model_result_dir, extraction_dir)
    if df.empty:
        print("ERROR: No data processed.")
        return

    if not coverage_df.empty:
        cov_path = os.path.join(out_dir, "llm_judge_coverage.csv")
        coverage_df.to_csv(cov_path, index=False, escapechar="\\", doublequote=True)
        print(f"  Coverage report   : {cov_path}")
        print("\n  Coverage summary (across all papers):")
        for agent in AGENTS:
            ag = coverage_df[coverage_df["agent"] == agent]
            n_ext  = ag["was_extracted"].sum()
            n_gold = ag["has_golden"].sum()
            n_both = (ag["was_extracted"] & ag["has_golden"]).sum()
            total  = len(ag)
            print(f"    {agent:<28}: "
                  f"extracted={n_ext}/{total}  "
                  f"has_golden={n_gold}/{total}  "
                  f"overlap={n_both}/{total}")

    review_path = os.path.join(out_dir, "llm_judge_annotation_review.csv")
    df.to_csv(review_path, index=False, escapechar="\\", doublequote=True)
    print(f"\n  Annotation review : {review_path}")
    print(f"  Total rows        : {len(df)}")

    per_paper_df = compute_per_paper_stats(df)
    if not per_paper_df.empty:
        per_paper_path = os.path.join(out_dir, "llm_judge_per_paper.csv")
        per_paper_df.to_csv(per_paper_path, index=False, escapechar="\\", doublequote=True)
        print(f"  Per-paper stats   : {per_paper_path}")

    print("\nGenerating plots...")
    plot_results(df, per_paper_df, model_name, out_dir)

    pred_df = df[df["extracted_value"] != "MISSING"]
    print("\nTier distribution:")
    for col, label in [("tier1_exact", "Exact"), ("tier2_normalized", "Normalized"),
                       ("tier3_ontology", "Ontology"), ("tier4_hierarchical", "Hierarchical"),
                       ("tier5_semantic", "Semantic"), ("no_match", "No Match")]:
        n   = int(pred_df[col].sum())
        pct = n / len(pred_df) * 100 if len(pred_df) > 0 else 0
        print(f"  {label:<15}: {n:>4}  ({pct:.1f}%)")

    if USE_LLM_JUDGE and not per_paper_df.empty and "judge_accuracy" in per_paper_df.columns:
        print(f"\nLLM Judge mean accuracy (GEval per-annotation): "
              f"{float(per_paper_df['judge_accuracy'].mean()):.1%}")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()