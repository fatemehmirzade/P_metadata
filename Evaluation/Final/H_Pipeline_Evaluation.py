import os
import re
import json
import hashlib
import time
import argparse
import threading
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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

EVALUATION_MODEL      = "google/gemma-4-31b-it"
OPENROUTER_BASE_URL   = "https://openrouter.ai/api/v1"
MODEL_TEMPERATURE     = 0
MODEL_ENABLE_THINKING = True

USE_LLM_JUDGE = True
LIMIT          = None

_CACHE_MODEL_SLUG = EVALUATION_MODEL.replace("/", "_").replace(" ", "_")
CACHE_DIR         = os.path.join(BASE_DIR, f".prompt_cache_{_CACHE_MODEL_SLUG}")
CACHE_ENABLED     = True

MAX_WORKERS = 4

MAX_RETRIES = 3
RETRY_DELAY = 2

_thread_local = threading.local()

AGENT_FILE_SUFFIX = {
    "BiologicalAgent":         "biological",
    "TechnicalAgent":          "technical",
    "ExperimentalDesignAgent": "experimental",
}

SKIP_MISSED_FIELDS = {
    "pxd_id", "dataset_id", "accession", "pride_id", "proteomexchange_id",
    "repository", "doi", "pubmed_id", "pmid",
    "technology_type", "phenotype", "mass_analyzer", "ethnicity",
}

SKIP_ANNOTATION_FIELDS = {"phenotype", "ethnicity", "mass_analyzer", "technology_type"}

_ALWAYS_LLM_FIELDS = {"material_type", "acquisition_method"}

_NOT_EXTRACTED_VALUES = {"unknown", "n/a", "not available", "none", "null", "na", ""}

def _is_not_extracted(value: str) -> bool:
    return value.strip().lower() in _NOT_EXTRACTED_VALUES


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

SAFE_DEFAULTS = {
    "disease":              {"normal", "healthy", "no disease", "none", "not applicable", "not diseased"},
    "disease_state":        {"normal", "healthy", "no disease", "none", "not applicable", "not diseased"},
    "developmental_stage":  {"adult", "mature", "neonatal", "embryonic", "fetal", "postnatal",
                             "juvenile", "larval", "seedling", "pupal", "aged"},
    "label":                {"label-free", "lfq", "none", "label free"},
    "labeling":             {"label-free", "lfq", "none", "label free"},
    "fractions":            {"1", "none", "not applicable"},
    "number_of_fractions":  {"1", "none", "not applicable"},
    "replicates":           {"1"},
    "technical_replicates": {"1"},
    "experimental_design":  {"treated vs control", "case vs control", "time course",
                             "dose response", "cross-sectional", "longitudinal"},
}

CONCENTRATION_FIELDS = {"reduction_concentration", "alkylation_concentration"}


SDRF_COLUMN_TO_FIELD: dict[str, str] = {
    "characteristics[organism]":            "species",
    "characteristics[organism part]":       "organ",
    "characteristics[age]":                 "age",
    "characteristics[developmental stage]": "developmental_stage",
    "characteristics[sex]":                 "sex",
    "characteristics[ancestry category]":   "ethnicity",
    "characteristics[cell type]":           "cell_type",
    "characteristics[disease]":             "disease",
    "characteristics[cell line]":           "cell_line",
    "characteristics[strain]":              "strain",
    "characteristics[phenotype]":           "phenotype",
    "characteristics[individual]":          "individual",
    "comment[label]":                       "label",
    "comment[fraction identifier]":         "_fraction_identifier",
    "comment[ms2 mass analyzer]":           "mass_analyzer",
    "comment[instrument]":                  "instrument",
    "comment[dissociation method]":         "fragmentation",
    "comment[cleavage agent details]":      "cleavage_agent",
    "comment[enrichment process]":          "enrichment_method",
    "comment[collision energy]":            "collision_energy",
    "comment[acquisition method]":          "acquisition_method",
    "comment[reduction reagent]":           "reduction_reagent",
    "comment[alkylation reagent]":          "alkylation_reagent",
    "comment[fractionation method]":        "fractionation",
    "comment[technical replicate]":         "technical_replicates",
    "material type":                        "material_type",
    "technology type":                      "technology_type",
}

_SDRF_SKIP_VALUES = {
    "not available", "not applicable", "not aplicable",
    "n/a", "na", "none", "null", "unknown", "",
}


def _parse_sdrf_value(raw: str) -> str | None:

    raw = str(raw).strip()
    if raw.lower() in _SDRF_SKIP_VALUES:
        return None

    nt_match = re.search(r"\bNT=([^;]+)", raw)
    if nt_match:
        name = nt_match.group(1).strip()
        if name.lower() not in _SDRF_SKIP_VALUES:
            return name
        return None

    return raw if raw else None


def _collapse_technical_replicates(golden: dict[str, list[str]]) -> None:
    if "technical_replicates" not in golden:
        return
    vals = golden["technical_replicates"]
    numeric_vals = []
    for v in vals:
        try:
            numeric_vals.append(int(float(v)))
        except (ValueError, TypeError):
            pass
    if numeric_vals:
        max_val = max(numeric_vals)
        golden["technical_replicates"] = [str(max_val)]
        print(f"      [FIX] technical_replicates: collapsed {vals} → ['{max_val}']")


def _deduplicate_cell_type_vs_organ(golden: dict[str, list[str]]) -> None:
    if "cell_type" not in golden or "organ" not in golden:
        return
    organ_lower = {v.lower() for v in golden["organ"]}
    remaining = [v for v in golden["cell_type"] if v.lower() not in organ_lower]
    removed = len(golden["cell_type"]) - len(remaining)
    if removed > 0:
        print(f"      [FIX] cell_type: removed {removed} values that duplicate organ entries")
    if remaining:
        golden["cell_type"] = remaining
    else:
        del golden["cell_type"]
        print(f"      [FIX] cell_type: fully subsumed by organ — removed from golden")


def load_golden_from_sdrf(pxd_id: str) -> dict[str, list[str]]:
    dataset_dir = os.path.join(TEST_SET, pxd_id)
    sdrf_path   = None
    if os.path.isdir(dataset_dir):
        for fname in os.listdir(dataset_dir):
            if fname.lower().endswith("sdrf.tsv") or fname.lower() == "sdrf.tsv":
                sdrf_path = os.path.join(dataset_dir, fname)
                break
            if "sdrf" in fname.lower() and fname.lower().endswith(".tsv"):
                sdrf_path = os.path.join(dataset_dir, fname)
                break
    if sdrf_path is None or not os.path.exists(sdrf_path):
        print(f"    WARNING: SDRF not found in {dataset_dir}")
        return {}
    try:
        df = pd.read_csv(sdrf_path, sep="\t", dtype=str, na_values=[])
        df = df.fillna("")
    except Exception as e:
        print(f"    WARNING: could not read SDRF {sdrf_path}: {e}")
        return {}
    print(f"    SDRF loaded: {os.path.basename(sdrf_path)}  "
          f"({len(df)} rows × {len(df.columns)} cols)")
    golden: dict[str, list[str]] = {}
    col_map: dict[str, str] = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        col_base  = re.sub(r"\.\d+$", "", col_lower)
        if col_base in SDRF_COLUMN_TO_FIELD:
            col_map[col] = SDRF_COLUMN_TO_FIELD[col_base]
    for col, field in col_map.items():
        if field == "_fraction_identifier":
            unique_fracs = set()
            for v in df[col]:
                parsed = _parse_sdrf_value(v)
                if parsed is not None:
                    unique_fracs.add(parsed)
            if unique_fracs:
                n_fracs = str(len(unique_fracs))
                golden.setdefault("fractions", [])
                if n_fracs not in golden["fractions"]:
                    golden["fractions"].append(n_fracs)
            continue
        unique_vals: list[str] = []
        seen_lower: set[str]   = set()
        for v in df[col]:
            parsed = _parse_sdrf_value(v)
            if parsed is not None and parsed.lower() not in seen_lower:
                seen_lower.add(parsed.lower())
                unique_vals.append(parsed)
        if unique_vals:
            existing = golden.setdefault(field, [])
            for uv in unique_vals:
                if uv.lower() not in {e.lower() for e in existing}:
                    existing.append(uv)
    if "source name" in {c.lower() for c in df.columns}:
        source_col = [c for c in df.columns if c.lower() == "source name"]
        if source_col:
            unique_sources = df[source_col[0]].dropna().unique()
            meaningful = [s for s in unique_sources
                          if str(s).strip().lower() not in _SDRF_SKIP_VALUES]
            if meaningful:
                golden.setdefault("number_of_samples", [str(len(meaningful))])
    _collapse_technical_replicates(golden)
    _deduplicate_cell_type_vs_organ(golden)
    for skip_field in SKIP_ANNOTATION_FIELDS:
        if skip_field in golden:
            del golden[skip_field]
    n_fields = len(golden)
    n_vals   = sum(len(v) for v in golden.values())
    print(f"    SDRF golden: {n_fields} fields, {n_vals} total values")
    for fld, vals in sorted(golden.items()):
        preview = "; ".join(vals[:5])
        if len(vals) > 5:
            preview += f" ... (+{len(vals)-5} more)"
        print(f"      {fld}: [{preview}]")
    return golden


def _flatten_golden_lists(golden_lists: dict[str, list[str]]) -> dict[str, str]:
    flat: dict[str, str] = {}
    for field, vals in golden_lists.items():
        if vals:
            flat[field] = vals[0]
    return flat


_FIELD_DEFINITIONS_TEXT = """\
BIOLOGICAL / SAMPLE FIELD TYPES:
species              : Source organism (e.g. "Homo sapiens", "Mus musculus"). Common names ("human", "mouse") are equivalent to scientific names.
organ                : Tissue or organ of origin (e.g. "liver", "brain cortex", "plasma"). Equivalent to the 'tissue' pipeline field.
cell_type            : Primary cell type or lineage (e.g. "neurons", "fibroblasts").
cell_line            : Name of immortalized cell line (e.g. "HEK293T", "HeLa").
disease              : Disease state or diagnosis (e.g. "breast cancer", "Type 2 diabetes"). "normal" or "healthy" is correct when subjects are healthy controls. Equivalent to the 'disease_state' pipeline field.
sex                  : Donor sex (e.g. "male", "female").
age                  : Age of donor or developmental time point (e.g. "45 years", "E14.5 embryo"). Ranges are acceptable when reported as a range.
developmental_stage  : Developmental stage of source material (e.g. "adult", "embryonic", "early seed development"). Inferable from subject description (e.g. "adult patients" -> "adult").
ethnicity            : Donor ancestry or ethnicity (e.g. "European", "East Asian").
material_type        : Broad material class: "tissue", "cell line", "primary cells", "biofluid", "whole organism", "plasma", "serum", "organoid". Equivalent to the 'sample_source' pipeline field. Must match what was actually used.
strain               : Animal or plant strain (e.g. "BALB/c", "C57BL/6J", "Nipponbare").
BMI                  : Body-Mass Index of donor (kg/m^2).
anatomic_site_tumor  : Anatomical location of the tumor, if applicable (e.g. "left lung lobe", "colon").

TECHNICAL / MS FIELD TYPES:
instrument           : Mass spectrometer make and model (e.g. "Thermo Q-Exactive Plus", "Orbitrap Fusion Lumos"). LC instruments alone are NOT this type.
cleavage_agent       : Protease used for protein digestion (e.g. "trypsin", "Lys-C"). "Trypsin/P" equals "trypsin".
label                : Isobaric or metabolic labeling method applied (e.g. "TMT", "SILAC", "label-free", "iTRAQ"). "label-free" is correct when no isobaric/metabolic labels are used. Equivalent to the 'labeling' pipeline field.
fragmentation        : Fragmentation method for MS/MS (e.g. "HCD", "CID", "ETD"). Equivalent to 'fragmentation_method'.
ptm                  : Post-translational modification studied or enriched for (e.g. "phosphorylation", "ubiquitination").
reduction_reagent    : Chemical used to reduce disulfide bonds (e.g. "DTT", "TCEP", "dithiothreitol").
reduction_concentration : Concentration of the reduction reagent (e.g. "5 mM", "2.5 mM", "10 mM"). Must be a numeric quantity with a unit. The reagent name is extracted separately in the 'reduction_reagent' field, so the concentration value alone (e.g. "2.5 mM" without the reagent name) is COMPLETE and sufficient.
collision_energy     : Collision energy used in MS/MS (e.g. "normalized collision energy 25", "27 eV").
fractionation        : Offline peptide/protein fractionation method applied before LC-MS (e.g. "high-pH reverse-phase fractionation", "strong anion exchange"). NOT biological/cellular fractionation. Equivalent to 'fractionation_method'.
enrichment_method    : Enrichment protocol applied before MS (e.g. "TiO2 phosphopeptide enrichment", "immunoprecipitation").
acquisition_method   : MS acquisition scheme (e.g. "DDA", "DIA", "data-dependent", "data-independent"). Must be a scheme name, NOT an instrument name. Equivalent to 'acquisition_method'.
alkylation_reagent   : Chemical used for cysteine alkylation (e.g. "iodoacetamide", "IAA", "NEM").
alkylation_concentration : Concentration of the alkylation reagent (e.g. "10 mM", "55 mM", "20 mM"). Must be a numeric quantity with a unit. The reagent name is extracted separately in the 'alkylation_reagent' field, so the concentration value alone (e.g. "10 mM" without the reagent name) is COMPLETE and sufficient.
ionization_type      : Ionization source type (e.g. "electrospray", "nano-ESI", "MALDI"). Can be inferred from instrument model (Q Exactive -> electrospray).
mass_analyzer        : Mass analyzer type used (e.g. "Orbitrap", "TOF", "quadrupole", "ion trap").

EXPERIMENTAL DESIGN FIELD TYPES:
replicates                    : Number of biological replicates (e.g. "3"). "1" is correct when no replication is mentioned. Equivalent to 'number_of_biological_replicates' / 'biological_replicate'.
technical_replicates          : Number of technical replicates per sample (e.g. "2"). Equivalent to 'number_of_technical_replicates' / 'technical_replicate'.
number_of_samples             : Total count of samples processed in the study (e.g. "12").
fractions                     : Number of fractions generated per sample (e.g. "12"). "1" is correct when no fractionation occurred. Equivalent to 'number_of_fractions'.
technology_type               : Broad technology type applied (e.g. "proteomics", "phosphoproteomics").
factor_value                  : Experimental factor or variable under study (e.g. "drug treatment", "cell type", "time point").
experimental_design           : Study design type (e.g. "time course", "cross-sectional", "case-control", "treated vs control").
"""

FIELD_DEFINITIONS: dict[str, str] = {}
for _line in _FIELD_DEFINITIONS_TEXT.strip().splitlines():
    _stripped = _line.strip()
    if not _stripped or _stripped.endswith(":"):
        continue
    if ":" in _stripped:
        _parts = _stripped.split(":", 1)
        FIELD_DEFINITIONS[_parts[0].strip()] = _stripped

_KNOWN_FIELD_TYPES_LOWER: set[str] = set(FIELD_DEFINITIONS.keys())



_STATIC_SYSTEM_PROMPT = """\
You are an expert in proteomics and mass spectrometry experimental metadata.
Your task is to assess the correctness of metadata field values extracted from
proteomics papers, measured against the field definitions provided below.

You will receive two inputs:
  1. A PAPER TEXT message (title + abstract + methods), prepended to the conversation.
  2. An assessment request as a JSON object with a "task" key.

Three task types are supported.

=================================================================
TASK TYPE: "verify"
=================================================================
Determine whether a candidate value is a valid extraction for the
given metadata field, by reading the paper text and applying the
field definitions below.

JSON input fields:
  field_name        -- the metadata field being assessed
  field_definition  -- precise definition of that field
  candidate_value   -- the extracted value to assess (ONE value)
  reference_values  -- list of golden/ground-truth values (may be []).
                       PROVIDED FOR MATCHED_REFERENCE TRACKING ONLY.
                       These do NOT determine the verdict. A value that
                       differs from all reference_values but is correct
                       per the paper text and definition MUST receive
                       VERDICT: high.

=================================================================
CORE EVALUATION QUESTIONS (answer in order)
=================================================================
1. TYPE CHECK   : Does the candidate value belong to this metadata field's
                  definition? Ask: "Is this the RIGHT CATEGORY OF THING
                  for this field (organism name, tissue, instrument model,
                  method name, numeric count, etc.)?"
                  -- If the value is the right CATEGORY but wrong SPECIFIC
                     ENTITY within that category: TYPE_CORRECT: yes,
                     VALUE_CORRECT: no, VERDICT: low (rule D).
                  -- Only TYPE_CORRECT: no (rule B) when the fundamental
                     nature of the value belongs to a completely different
                     field.

2. SOURCE CHECK : Is the candidate value present in or reasonably inferable
                  from the paper text?
                  -- SAFE DEFAULTS that do NOT require explicit mention:
                     "normal"/"healthy" for disease when subjects are healthy;
                     "adult"/"neonatal"/etc. for developmental_stage when
                     inferable from subject description;
                     "label-free" for label/labeling when no isobaric or
                     metabolic labels are mentioned;
                     "1" for fractions or replicates when paper is silent;
                     standard material_type values inferable from context;
                     abbreviation expansions (IAA = iodoacetamide, HCD =
                     higher-energy collisional dissociation, etc.).
                  -- INFERABLE FIELDS: For material_type and acquisition_method,
                     the value may be reasonably inferable from the experimental
                     context even when not explicitly stated in the paper text.
                     For material_type, infer from what biological material was
                     used (e.g. cell lines, tissue, biofluid, etc.).
                     For acquisition_method, infer from the described MS workflow,
                     instrument settings, or analysis software (e.g. MaxQuant DDA,
                     Spectronaut DIA, etc.).
                     Do NOT mark these as HALLUCINATED solely because the exact
                     term is absent. Instead, assess whether the value is a
                     reasonable inference from the described experiment.
                  -- If absent AND not a safe default AND not reasonably
                     inferable -> HALLUCINATED: yes (rule A -> VERDICT: low).
                  -- If present but in a different context -> HALLUCINATED: no,
                     VALUE_CORRECT: no (rule D).

3. TRUTH CHECK  : Is the candidate value factually correct and valid for
                  this experiment as described in the paper?
                  -- Synonyms, abbreviations, digit/word swaps ("3"/"three"),
                     common/scientific name swaps ("human"/"Homo sapiens"),
                     and equivalent representations ("SILAC" for a SILAC
                     experiment) are all CORRECT.
                  -- Differing from reference_values alone is NEVER a reason
                     to fail this check.

4. COMPLETENESS : Flag MEDIUM only for genuine incompleteness:
                  -- Subtype drift: value is more specific than what the paper
                     describes.
                  -- Supertype drift: value is broader (e.g. "cells" when paper
                     says "cell line").
                  -- Single channel of a confirmed multi-channel scheme
                     (e.g. only "heavy" when paper uses triple SILAC).
                  -- Extracting part of a multi-value field (e.g. one tissue
                     when two are used).
                  -- Do NOT flag medium for verbose phrasing, abbreviation
                     expansion, or quality qualifiers.

                  IMPORTANT — CONCENTRATION FIELDS EXCEPTION:
                  For reduction_concentration and alkylation_concentration,
                  the reagent name is extracted in a SEPARATE companion field
                  (reduction_reagent / alkylation_reagent). Therefore a
                  concentration value that contains only the numeric quantity
                  and unit (e.g. "2.5 mM", "10 mM", "55 mM") WITHOUT the
                  reagent name is COMPLETE. Do NOT flag these as incomplete
                  for lacking the reagent name.

=================================================================
VERDICT RULES
=================================================================

LOW (most severe) -- use when ANY of:
  (A) HALLUCINATED: absent from paper text and not a safe default.
  (B) TYPE MISMATCH: the value's fundamental nature belongs to a
      completely different metadata field.
  (D) CONTRADICTION / INCORRECT: present in source, type is correct,
      but the value is factually wrong or contradicts the paper.

MEDIUM -- type correct, value present, no LOW rule applies, AND any of:
  (a) SUBTYPE / SUPERTYPE DRIFT.
  (b) SINGLE CHANNEL of a multi-channel scheme.
  (c) Partial extraction from a multi-value field.
  NOTE: concentration fields missing only the reagent name are NOT incomplete.

HIGH -- correct type, present (or valid safe default), factually
accurate, and complete.
IMPORTANT: A value that is NOT in reference_values but is correct per
the paper text and definition MUST receive HIGH.

=================================================================
THINGS THAT ARE NEVER A REASON TO DOWNGRADE
=================================================================
  - Differing from reference_values.
  - Verbose or sentence-form phrasing.
  - Abbreviation expansion or contraction.
  - Synonym or equivalent representation.
  - Digit/word swaps ("3" = "three").
  - Unit spacing/case differences ("37 C" = "37C").
  - Common-name/scientific-name swaps.
  - Quality/purity qualifiers ("MS-grade trypsin" = "trypsin").
  - Enzyme configuration variant ("Trypsin/P" = "trypsin").
  - "SILAC" = "triple SILAC" when paper uses triple SILAC.
  - Inferable safe default values listed in SOURCE CHECK above.
  - Concentration fields missing the reagent name (extracted separately).

=================================================================
DOMAIN EQUIVALENCES (treat as identical)
=================================================================
  - "Homo sapiens" = "human"; "Mus musculus" = "mouse"
  - "IAA" = "iodoacetamide"; "DTT" = "dithiothreitol"; "TCEP" = tris(2-carboxyethyl)phosphine
  - "HCD" = "higher-energy collisional dissociation"
  - "DDA" = "data-dependent acquisition"; "DIA" = "data-independent acquisition"
  - "label-free" = "LFQ" = "label free" (when no isobaric labels used)
  - "SILAC" = "triple SILAC" (when paper describes triple SILAC)
  - Any abbreviation in parentheses that expands the other value.
  - "Trypsin/P" = "trypsin"

=================================================================
MATCHED_REFERENCE field (tracking only — does NOT affect verdict)
=================================================================
Identify the ONE reference value from reference_values that the candidate
most closely relates to. Output verbatim if one relates; output NONE if
reference_values is [] or no reference relates.

This field does NOT affect the verdict.

=================================================================
OUTPUT FORMAT for "verify"
=================================================================
TYPE CHECK:        <one sentence>
SOURCE CHECK:      <one sentence: is the candidate present or a valid safe default?>
TRUTH CHECK:       <one sentence: is the candidate factually correct?>
COMPLETENESS CHECK:<one sentence>
MATCHED_REFERENCE: <exact reference string, or NONE>
TYPE_CORRECT:      yes | no
CORRECT_TYPE_NAME: <the correct field name when TYPE_CORRECT is no, otherwise NONE>
VALUE_CORRECT:     yes | no
VALUE_COMPLETE:    yes | no
HALLUCINATED:      yes | no
VERDICT:           high | medium | low
CORRECTED_VALUE:   <when VERDICT is medium: the complete/correct value from the
                   paper text that this field SHOULD be. When VERDICT is
                   high or low: output NONE>

Field meanings:
  TYPE_CORRECT      = yes when the value belongs to the correct field.
                      no ONLY when fundamental nature is that of a different field.
  CORRECT_TYPE_NAME = the field name the value actually belongs to when
                      TYPE_CORRECT is no. Must be one of the defined field names.
                      NONE when TYPE_CORRECT is yes.
  VALUE_CORRECT     = yes when present/inferable, type correct, factually valid.
                      no for hallucinations, mismatches, contradictions.
  VALUE_COMPLETE    = yes when fully specified.
                      no only for partial/single-channel stand-ins.
                      IMPORTANT: For reduction_concentration and
                      alkylation_concentration, a numeric quantity + unit
                      WITHOUT the reagent name IS complete (yes).
  HALLUCINATED      = yes ONLY when absent from paper text AND not a safe default.
  VERDICT           = overall high | medium | low.
  CORRECTED_VALUE   = when VERDICT is medium, the complete or correct value
                      from the paper text. For supertype drift, give the more
                      specific term. For single-channel, give the full multi-
                      channel label. Output NONE for high and low verdicts.

Typical couplings:
  HIGH verdict              -> CORRECTED_VALUE NONE
  MEDIUM (drift)            -> CORRECTED_VALUE <corrected>
  MEDIUM (single chan)      -> CORRECTED_VALUE <corrected>
  LOW hallucination         -> CORRECTED_VALUE NONE
  LOW type mismatch         -> CORRECTED_VALUE NONE
  LOW wrong kind            -> CORRECTED_VALUE NONE

=================================================================
TASK TYPE: "check_missing"
=================================================================
Determine whether the paper text contains information for the specified
metadata field.

JSON input fields:
  field_name       -- the metadata field
  field_definition -- precise definition of that field

OUTPUT FORMAT:
SEARCH:  <one sentence>
FINDING: <one sentence>
VERDICT: high | medium | low
(high = clearly present; medium = inferable; low = absent)

=================================================================
TASK TYPE: "check_coverage"
=================================================================
Determine whether a specific reference value (gt_value) is semantically
COVERED by ANY of the candidate values provided.

JSON input fields:
  field_name        -- the metadata field
  field_definition  -- precise definition of that field
  gt_value          -- the reference value to look for
  candidate_values  -- list of candidate values to check against

OUTPUT FORMAT for "check_coverage":
SEARCH:    <one sentence: which candidate is the best match?>
REASONING: <one sentence: does that candidate cover gt_value?>
COVERED: yes | no

=================================================================
WORKED EXAMPLES (verify)
=================================================================

Example 1 -- HIGH: synonym, correct
field_name=species, candidate_value="human", reference_values=["Homo sapiens"]

TYPE CHECK:        Species/organism name -- correct field.
SOURCE CHECK:      Present in the paper text.
TRUTH CHECK:       "human" is the common-name equivalent of "Homo sapiens" -- correct.
COMPLETENESS CHECK:Complete.
MATCHED_REFERENCE: Homo sapiens
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high
CORRECTED_VALUE:   NONE

Example 2 -- LOW (rule B): instrument name labeled as acquisition_method
field_name=acquisition_method, candidate_value="Q Exactive Plus",
reference_values=["DDA"]

TYPE CHECK:        "Q Exactive Plus" is a mass spectrometer model -- its
                   fundamental nature belongs to the 'instrument' field.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      no
CORRECT_TYPE_NAME: instrument
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           low
CORRECTED_VALUE:   NONE

Example 3 -- LOW (rule A): hallucinated
field_name=enrichment_method, candidate_value="IMAC phosphopeptide enrichment",
reference_values=[]

TYPE CHECK:        Enrichment protocol -- correct field.
SOURCE CHECK:      No mention of IMAC in the paper -- HALLUCINATED.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      yes
VERDICT:           low
CORRECTED_VALUE:   NONE

Example 4 -- MEDIUM: single channel of triple SILAC
field_name=label, candidate_value="heavy",
reference_values=["triple SILAC"]

TYPE CHECK:        Isotopic label channel -- correct field.
SOURCE CHECK:      "heavy" channel present in paper.
TRUTH CHECK:       Correct that a heavy channel is used, but paper uses triple SILAC.
COMPLETENESS CHECK:Incomplete -- only one channel of a three-channel scheme.
MATCHED_REFERENCE: triple SILAC
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   triple SILAC

Example 5 -- HIGH: concentration without reagent name (extracted separately)
field_name=reduction_concentration, candidate_value="2.5 mM",
reference_values=[]

TYPE CHECK:        Numeric concentration with unit -- correct field.
SOURCE CHECK:      "2.5 mM" is present in the paper text.
TRUTH CHECK:       The concentration value is factually correct per the paper.
COMPLETENESS CHECK:Complete -- the reagent name is extracted separately.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high
CORRECTED_VALUE:   NONE

Example 6 -- HIGH: material_type inferred from context
field_name=material_type, candidate_value="cell line",
reference_values=[]

TYPE CHECK:        Broad material class -- correct field.
SOURCE CHECK:      Paper describes experiments on HEK293T cells, which is a cell line.
                   "cell line" is reasonably inferable from the experimental context.
TRUTH CHECK:       Factually correct -- HEK293T is indeed a cell line.
COMPLETENESS CHECK:Complete.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high
CORRECTED_VALUE:   NONE

Example 7 -- HIGH: acquisition_method inferred from workflow
field_name=acquisition_method, candidate_value="DDA",
reference_values=[]

TYPE CHECK:        MS acquisition scheme -- correct field.
SOURCE CHECK:      Paper describes using MaxQuant for analysis and top-N MS2 scans,
                   which is characteristic of DDA. Reasonably inferable.
TRUTH CHECK:       Factually correct based on the described workflow.
COMPLETENESS CHECK:Complete.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high
CORRECTED_VALUE:   NONE

=================================================================
METADATA FIELD DEFINITIONS
=================================================================
""" + _FIELD_DEFINITIONS_TEXT



_ANNOTATION_GEVAL_CRITERIA = (
    "Assess whether a candidate metadata value is a valid extraction for the "
    "given field, based solely on the paper text and field definitions. "
    "Evaluate in order: "
    "(1) TYPE: does it match the field definition in FUNDAMENTAL NATURE? "
    "    A value of the wrong specific entity is TYPE_CORRECT: yes but "
    "    VALUE_CORRECT: no (rule D). Only TYPE_CORRECT: no when the value "
    "    belongs to a completely different field category (rule B). "
    "(2) SOURCE: is it present in or a valid safe default for the paper text? "
    "    Absent and not a safe default -> HALLUCINATED: yes -> low. "
    "    Present but wrong context -> HALLUCINATED: no, VALUE_CORRECT: no (rule D). "
    "    IMPORTANT: For material_type and acquisition_method, the value may be "
    "    reasonably inferable from the experimental context even when not explicitly "
    "    stated. Do NOT mark as HALLUCINATED solely because the exact term is absent. "
    "    Assess whether the value is a reasonable inference from the experiment. "
    "(3) TRUTH: is it factually correct per the paper and field definition? "
    "    Synonyms, abbreviations, digit/word swaps = CORRECT. "
    "    Differing from reference_values is NEVER a reason to fail. "
    "(4) COMPLETENESS: single-channel stand-in or partial multi-value -> medium. "
    "    Subtype or supertype drift -> medium. "
    "    EXCEPTION: For reduction_concentration and alkylation_concentration, "
    "    a numeric quantity + unit WITHOUT the reagent name IS complete because "
    "    the reagent is extracted separately in a companion field. "
    "HIGH = correct type + present/inferable + factually valid + complete. "
    "MEDIUM = correct type + present + valid, but partial/drift. "
    "LOW = hallucinated, type mismatch, or factually wrong. "
    "When VERDICT is medium, also output CORRECTED_VALUE: <the complete/correct value "
    "from the paper text that this field should be>."
)

_ANNOTATION_GEVAL_STEPS = [
    "TYPE CHECK: ask 'What is the fundamental nature of this value (organism name, "
    "tissue/organ, instrument model, method name, numeric count, etc.)?' "
    "If its nature belongs to a completely different field, state the correct field "
    "and set TYPE_CORRECT: no -> VERDICT: low (rule B). "
    "If it is the right category but the wrong specific entity, TYPE_CORRECT: yes "
    "and VALUE_CORRECT: no (rule D).",

    "SOURCE CHECK: confirm the candidate value (or safe default / domain-equivalent "
    "phrasing) is present in or inferable from the paper text. "
    "Safe defaults that do NOT require explicit mention: 'normal'/'healthy' for disease "
    "when subjects are controls; developmental stage terms inferable from subjects; "
    "'label-free' when no labeling is used; '1' for fractions/replicates when not discussed; "
    "standard material_type values inferable from context; abbreviation expansions. "
    "IMPORTANT: For material_type and acquisition_method, the value may be reasonably "
    "inferable from the experimental context (e.g. cell lines imply material_type='cell line', "
    "MaxQuant + top-N scans imply acquisition_method='DDA'). Do NOT mark as HALLUCINATED "
    "solely because the exact term is absent — assess whether it is a reasonable inference. "
    "If ABSENT and not a safe default and not reasonably inferable: "
    "HALLUCINATED: yes -> VERDICT: low (rule A). "
    "If PRESENT but in wrong context: HALLUCINATED: no, VALUE_CORRECT: no (rule D).",

    "TRUTH CHECK: assess whether the candidate is factually correct AND fits the field "
    "definition for this experiment. Synonyms, abbreviations, digit/word swaps, "
    "common/scientific name swaps are CORRECT. Differing from reference_values alone "
    "is NEVER a reason to fail -- a correct value not in reference_values is HIGH.",

    "COMPLETENESS CHECK: flag MEDIUM only for (a) subtype drift, (b) supertype drift, "
    "(c) single-channel stand-in for multi-channel scheme, or (d) partial extraction "
    "from a known multi-value field. Do NOT flag medium for verbose phrasing, "
    "abbreviation expansion, or quality qualifiers. "
    "CRITICAL EXCEPTION: For reduction_concentration and alkylation_concentration fields, "
    "a value containing only the numeric quantity and unit (e.g. '2.5 mM', '10 mM') "
    "WITHOUT the reagent name is COMPLETE — the reagent name is extracted in a separate "
    "companion field (reduction_reagent / alkylation_reagent). Do NOT flag as incomplete.",

    "MATCHED_REFERENCE: identify the closest reference from reference_values for "
    "tracking only -- does NOT affect verdict. Output verbatim or NONE.",

    "Output TYPE CHECK / SOURCE CHECK / TRUTH CHECK / COMPLETENESS CHECK, "
    "MATCHED_REFERENCE, TYPE_CORRECT (yes/no), CORRECT_TYPE_NAME (field_name or NONE), "
    "VALUE_CORRECT (yes/no), VALUE_COMPLETE (yes/no), "
    "HALLUCINATED (yes/no), VERDICT: high | medium | low, and "
    "CORRECTED_VALUE: <value or NONE>.",
]

_MISSED_GEVAL_CRITERIA = (
    "Determine whether the paper text contains information for the specified "
    "metadata field. Output VERDICT: high | medium | low. "
    "(high = clearly present; medium = inferable; low = absent)"
)
_MISSED_GEVAL_STEPS = [
    "Search the source text for any mention of the metadata field.",
    "Determine if a valid value is clearly present, inferable, or absent.",
    "Write SEARCH, FINDING, and VERDICT: high | medium | low.",
]

_COVERAGE_GEVAL_CRITERIA = (
    "Determine whether the reference value (gt_value) is semantically COVERED "
    "by ANY of the candidate values provided. Coverage includes: semantically "
    "equivalent (exact match, abbreviation, domain equivalence, common/scientific "
    "name swap, digit/word swap, unit reformat), a more-specific subtype/variant/"
    "configuration, a broader term that clearly subsumes gt_value, a verbose or "
    "sentence-form phrasing of the same concept, or a list/set that contains "
    "gt_value as one of its items. Output COVERED: yes or COVERED: no."
)
_COVERAGE_GEVAL_STEPS = [
    "Identify the best candidate among candidate_values that could relate to gt_value.",
    "Determine if that candidate is semantically equivalent to, a subtype/variant of, "
    "a broader-term subsumer of, a sentence-form phrasing of, or a list containing gt_value.",
    "Output SEARCH, REASONING, and COVERED: yes | no.",
]


class DiskResponseCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self._dir     = Path(cache_dir)
        self._enabled = enabled
        self._hits    = 0
        self._misses  = 0
        self._lock    = threading.Lock()
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(paper_id: str, task: str, field_name: str, primary_value: str) -> str:
        blob = f"{EVALUATION_MODEL}|{paper_id}|{task}|{field_name.lower()}|{primary_value}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, paper_id, task, field_name, primary_value):
        if not self._enabled:
            return None
        key  = self._make_key(paper_id, task, field_name, primary_value)
        path = self._dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self._hits += 1
                return data.get("response")
            except (json.JSONDecodeError, OSError):
                return None
        with self._lock:
            self._misses += 1
        return None

    def put(self, paper_id, task, field_name, primary_value, response):
        if not self._enabled:
            return
        key  = self._make_key(paper_id, task, field_name, primary_value)
        path = self._dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps({"key": key, "paper_id": paper_id, "task": task,
                    "field_name": field_name, "primary_value": primary_value,
                    "response": response, "timestamp": time.time(),
                }, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            print(f"  [cache-write-error] {exc}")

    def stats(self):
        n_files = sum(1 for _ in self._dir.glob("*.json")) if self._enabled else 0
        total   = self._hits + self._misses
        return {"enabled": self._enabled, "cache_dir": str(self._dir),
            "cached_responses_on_disk": n_files, "session_hits": self._hits,
            "session_misses": self._misses,
            "session_hit_rate": f"{self._hits/total*100:.1f}%" if total > 0 else "n/a"}


def _build_messages(paper_text: str, entity_context: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": _STATIC_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "PAPER TEXT (title + abstract + methods) — use as source of truth "
                        "for all SOURCE CHECK and TRUTH CHECK assessments:\n\n" + paper_text
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": (
                "Understood. I have read the paper text and will assess all metadata "
                "field values against it using the definitions and rules in my instructions."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(entity_context, indent=2, ensure_ascii=False),
        },
    ]


def _build_openrouter_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
    return openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _extract_text_content(raw_message) -> str:
    if raw_message is None:
        return ""
    content = getattr(raw_message, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text") or "")
            elif hasattr(block, "type") and hasattr(block, "text"):
                if block.type == "text":
                    parts.append(block.text or "")
        return " ".join(parts)
    if isinstance(content, str):
        return content
    return ""


class Gemma4Judge(DeepEvalBaseLLM):

    def __init__(self):
        self._model_name      = EVALUATION_MODEL
        self._client          = None
        self._paper_text: str = ""
        self._current_paper_id: str = ""
        self._response_cache  = DiskResponseCache(CACHE_DIR, enabled=CACHE_ENABLED)
        self._call_count      = 0
        self._api_call_count  = 0
        self._disk_hit_count  = 0
        self._lock            = threading.Lock()

    def set_paper_context(self, source_text: str, paper_id: str = "") -> None:
        self._paper_text       = source_text
        self._current_paper_id = paper_id

    def set_entity_context(self, context: dict) -> None:
        _thread_local.current_context = context

    def load_model(self) -> openai.OpenAI:
        if self._client is None:
            self._client = _build_openrouter_client()
        return self._client

    def _generate_single(self, paper_text: str, paper_id: str,
                         context: dict) -> str:
        with self._lock:
            self._call_count += 1
            call_num = self._call_count

        task       = context.get("task", "verify")
        field_name = context.get("field_name", "")
        if task == "verify":
            primary_value = context.get("candidate_value", "")
        elif task == "check_coverage":
            primary_value = context.get("gt_value", "")
        else:
            primary_value = field_name

        cached = self._response_cache.get(paper_id, task, field_name, primary_value)
        if cached is not None:
            with self._lock:
                self._disk_hit_count += 1
            print(f"  [call #{call_num}] DISK CACHE HIT  "
                  f"paper={paper_id!r}  task={task}  "
                  f"field={field_name}  value={primary_value[:40]!r}")
            return cached

        with self._lock:
            self._api_call_count += 1

        messages = _build_messages(paper_text, context)
        print(f"  [call #{call_num}] API CALL  "
              f"paper={paper_id!r}  task={task}  "
              f"field={field_name}  value={primary_value[:40]!r}  "
              f"[sys={len(_STATIC_SYSTEM_PROMPT):,}  "
              f"paper={len(paper_text):,}  "
              f"entity={len(json.dumps(context)):,} chars]")

        extra_body = {}
        if MODEL_ENABLE_THINKING:
            extra_body["thinking"] = {"type": "enabled"}

        content = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.load_model().chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=MODEL_TEMPERATURE,
                    max_tokens=16000,
                    extra_body=extra_body if extra_body else None,
                    timeout=300,
                )

                if (response is None
                        or not hasattr(response, "choices")
                        or not response.choices):
                    if attempt < MAX_RETRIES:
                        print(f"  [WARNING] API returned empty/null response "
                              f"(attempt {attempt}/{MAX_RETRIES}), retrying in "
                              f"{RETRY_DELAY * attempt}s...")
                        time.sleep(RETRY_DELAY * attempt)
                        continue
                    else:
                        print(f"  [WARNING] API returned empty/null response "
                              f"(attempt {attempt}/{MAX_RETRIES}), giving up.")
                else:
                    message = response.choices[0].message
                    content = _extract_text_content(message)

                    if hasattr(response, "usage") and response.usage:
                        usage = response.usage
                        cache_info = ""
                        if hasattr(usage, "cache_creation_input_tokens"):
                            cache_info = (
                                f"  cache_create={usage.cache_creation_input_tokens:,}  "
                                f"cache_read={getattr(usage, 'cache_read_input_tokens', 0):,}"
                            )
                        print(f"  [tokens] prompt={usage.prompt_tokens:,}  "
                              f"completion={usage.completion_tokens:,}{cache_info}")

                    if content:
                        break
                    elif attempt < MAX_RETRIES:
                        print(f"  [WARNING] API returned empty text content "
                              f"(attempt {attempt}/{MAX_RETRIES}), retrying in "
                              f"{RETRY_DELAY * attempt}s...")
                        time.sleep(RETRY_DELAY * attempt)
                        continue
                    else:
                        print(f"  [WARNING] API returned empty text content "
                              f"(attempt {attempt}/{MAX_RETRIES}), giving up.")

            except Exception as api_exc:
                if attempt < MAX_RETRIES:
                    print(f"  [API ERROR] {type(api_exc).__name__}: {api_exc} "
                          f"(attempt {attempt}/{MAX_RETRIES}), retrying in "
                          f"{RETRY_DELAY * attempt}s...")
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    print(f"  [API ERROR] {type(api_exc).__name__}: {api_exc} "
                          f"(attempt {attempt}/{MAX_RETRIES}), giving up.")

        if content:
            self._response_cache.put(paper_id, task, field_name,
                                     primary_value, content)

        print(f"  [response] {content[:200]}...")
        return content

    def generate(self, prompt: str, schema=None):
        context = getattr(_thread_local, "current_context", None) or {}
        content = self._generate_single(
            self._paper_text, self._current_paper_id, context
        )

        if not content:
            content = (
                "TYPE CHECK: Unable to evaluate (no model response).\n"
                "SOURCE CHECK: N/A\nTRUTH CHECK: N/A\nCOMPLETENESS CHECK: N/A\n"
                "MATCHED_REFERENCE: NONE\n"
                "TYPE_CORRECT: yes\nCORRECT_TYPE_NAME: NONE\n"
                "VALUE_CORRECT: no\nVALUE_COMPLETE: no\n"
                "HALLUCINATED: no\nVERDICT: low\nCORRECTED_VALUE: NONE"
            )

        if schema is not None:
            payload = {"score": 1, "reason": content}
            for attempt in (
                lambda: schema(**payload),
                lambda: schema.model_validate(payload),
                lambda: schema.model_validate_json(json.dumps(payload)),
            ):
                try:
                    return attempt()
                except Exception:
                    continue
            return json.dumps(payload)
        return content

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return self._model_name

    def get_cache_stats(self) -> dict:
        stats = self._response_cache.stats()
        stats["total_generate_calls"] = self._call_count
        stats["actual_api_calls"]     = self._api_call_count
        stats["disk_cache_hits"]      = self._disk_hit_count
        if self._call_count > 0:
            stats["overall_disk_hit_rate"] = (
                f"{self._disk_hit_count / self._call_count * 100:.1f}%"
            )
        return stats


_judge_model: Gemma4Judge | None = None

def _get_judge_model() -> Gemma4Judge:
    global _judge_model
    if _judge_model is None:
        _judge_model = Gemma4Judge()
    return _judge_model


def _get_thread_annotation_metric() -> GEval:
    if not hasattr(_thread_local, "annotation_metric"):
        _thread_local.annotation_metric = GEval(
            name="AnnotationJudge",
            criteria=_ANNOTATION_GEVAL_CRITERIA,
            evaluation_steps=_ANNOTATION_GEVAL_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=_get_judge_model(), threshold=0.5)
    return _thread_local.annotation_metric


def _get_thread_missed_metric() -> GEval:
    if not hasattr(_thread_local, "missed_metric"):
        _thread_local.missed_metric = GEval(
            name="MissedFieldDetector",
            criteria=_MISSED_GEVAL_CRITERIA,
            evaluation_steps=_MISSED_GEVAL_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=_get_judge_model(), threshold=0.5)
    return _thread_local.missed_metric


def _get_thread_coverage_metric() -> GEval:
    if not hasattr(_thread_local, "coverage_metric"):
        _thread_local.coverage_metric = GEval(
            name="GTCoverageChecker",
            criteria=_COVERAGE_GEVAL_CRITERIA,
            evaluation_steps=_COVERAGE_GEVAL_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=_get_judge_model(), threshold=0.5)
    return _thread_local.coverage_metric


def _field_def_for(field_name: str) -> str:
    fn_lower = field_name.lower()
    for k, v in FIELD_DEFINITIONS.items():
        if k.lower() == fn_lower:
            return v
    return f"Metadata field: {field_name}"


def _parse_type_correct(reason: str) -> bool | None:
    m = re.search(r"TYPE_CORRECT\s*:\s*(yes|no)\b", reason, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "yes"
    return None


def _parse_correct_type_name(reason: str, field_name: str = "") -> str | None:
    fn_lower = field_name.strip().lower()
    def _accept(name):
        n = name.strip().strip("'\"`,.()")
        if n.lower() in _KNOWN_FIELD_TYPES_LOWER and n.lower() != fn_lower:
            return n
        return None
    m = re.search(r"CORRECT_TYPE_NAME\s*:\s*(\S+)", reason, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip("'\"`,.")
        if val.upper() not in ("NONE", "N/A", "NULL", "SAME", "YES", "NO"):
            result = _accept(val)
            if result:
                return result
    for pat in [r"correct\s+(?:field|type)\s+is\s+([a-z_]+)",
                r"belongs\s+to\s+(?:the\s+)?([a-z_]+)\s+field",
                r"should\s+be\s+(?:labeled\s+as\s+|classified\s+as\s+)?([a-z_]+)"]:
        m2 = re.search(pat, reason, re.IGNORECASE)
        if m2:
            result = _accept(m2.group(1))
            if result:
                return result
    return None


def _parse_yes_no(reason: str, field: str) -> bool | None:
    m = re.search(rf"{field}\s*:\s*(yes|no)\b", reason, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower() == "yes"


def _parse_matched_reference(reason: str) -> str | None:
    m = re.search(r"MATCHED_REFERENCE\s*:\s*(.+?)(?:\n|$)", reason, re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).strip().strip("'\" `")
    if not val or val.upper() in ("NONE", "N/A", "NULL"):
        return None
    return val


def _parse_corrected_value(reason: str) -> str | None:
    m = re.search(r"CORRECTED_VALUE\s*:\s*(.+?)(?:\n|$)", reason, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip("'\" `")
        if val and val.upper() not in ("NONE", "N/A", "NULL", "SAME"):
            return val

    fallback_patterns = [
        r'(?:paper|study)\s+(?:describes?|uses?|employs?|reports?)\s+'
        r'(?:a\s+)?(?:specifically\s+)?["\']([^"\']{3,60})["\']',
        r'specifically\s+describes?\s+(?:a\s+)?["\']([^"\']{3,60})["\']',
        r'should\s+be\s+["\']([^"\']{3,60})["\']',
        r'correct\s+value\s+(?:is|should\s+be)\s+["\']([^"\']{3,60})["\']',
        r'(?:paper|study)\s+(?:describes?|uses?)\s+(?:a\s+)?(?:specifically\s+)?'
        r'([A-Za-z][A-Za-z0-9 /\-]{2,40}?)(?:\s+(?:approach|experiment|protocol|method|labeling|scheme))',
    ]

    check_blocks = []
    for block_name in ["COMPLETENESS CHECK", "TRUTH CHECK"]:
        block_match = re.search(
            rf"{block_name}\s*:\s*(.+?)(?=TYPE_CORRECT|VALUE_CORRECT|VALUE_COMPLETE|HALLUCINATED|VERDICT|CORRECTED_VALUE|MATCHED_REFERENCE|$)",
            reason, re.IGNORECASE | re.DOTALL)
        if block_match:
            check_blocks.append(block_match.group(1))

    search_text = " ".join(check_blocks) if check_blocks else reason

    for pattern in fallback_patterns:
        fm = re.search(pattern, search_text, re.IGNORECASE)
        if fm:
            val = fm.group(1).strip().strip("'\" `.,;")
            if val and len(val) >= 2 and val.upper() not in ("NONE", "N/A", "NULL"):
                print(f"  [corrected-value:fallback] Extracted '{val}' from prose")
                return val

    return None


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s.strip().lower())


def _find_matching_reference(predicted, gt_candidates):
    p = _norm(predicted)
    if not p:
        return None
    for g in gt_candidates:
        if _norm(g) == p:
            return g
    return None


def _resolve_matched_reference(raw_ref, gt_candidates):
    if not raw_ref or not gt_candidates:
        return None
    raw_norm = _norm(raw_ref)
    for g in gt_candidates:
        if _norm(g) == raw_norm:
            return g
    for g in gt_candidates:
        gn = _norm(g)
        if gn and (gn in raw_norm or raw_norm in gn):
            return g
    return None


def _exact_match_result(matched_ref):
    return {"verdict": "high", "type_mismatch": False, "correct_type": None,
        "value_correct": True, "value_complete": True, "hallucination": False,
        "matched_reference": matched_ref, "corrected_value": None,
        "issue_summary": "Exact match with a ground-truth value -- LLM skipped.",
        "match_type": "CORRECT", "any_match": True, "geval_score": None}


def _safe_default_result(field_name, extracted_value):
    return {"verdict": "high", "type_mismatch": False, "correct_type": None,
        "value_correct": True, "value_complete": True, "hallucination": False,
        "matched_reference": None, "corrected_value": None,
        "issue_summary": f"Safe default value '{extracted_value}' -- LLM skipped.",
        "match_type": "CORRECT", "any_match": True, "geval_score": None}


def _is_concentration_only(field_name, extracted_value):
    if field_name.lower() not in CONCENTRATION_FIELDS:
        return False
    val = extracted_value.strip()
    if not val:
        return False
    conc_pattern = re.compile(
        r"^\s*\d+(?:\.\d+)?\s*(?:m[Mm]|[µu]M|M|mol/[Ll]|mmol/[Ll]|%)\s*$")
    return bool(conc_pattern.match(val))


def _run_geval_semantic(field_name, extracted_value, gt_candidates):
    model = _get_judge_model()
    context = {"task": "verify", "field_name": field_name,
        "field_definition": _field_def_for(field_name),
        "candidate_value": extracted_value, "reference_values": gt_candidates}

    model.set_entity_context(context)

    expected_str = (
        "Assess the candidate_value against the paper text AND the field definition: "
        "correct metadata field in FUNDAMENTAL NATURE (TYPE_CORRECT yes/no), "
        "present/inferable in source (HALLUCINATED yes/no), "
        "factually valid (VALUE_CORRECT yes/no), complete (VALUE_COMPLETE yes/no). "
        "Synonyms, abbreviations, safe defaults = CORRECT and not hallucinated. "
        "Differing from reference_values alone is NEVER a reason to fail. "
        "When VERDICT is medium, also output CORRECTED_VALUE: <the complete/correct value>. ")
    if field_name.lower() in CONCENTRATION_FIELDS:
        expected_str += (
            "IMPORTANT: This is a concentration field. The reagent name is extracted "
            "separately in a companion field. A value with only numeric quantity + unit "
            "(e.g. '2.5 mM') WITHOUT the reagent name is COMPLETE (VALUE_COMPLETE: yes). ")
    if field_name.lower() in _ALWAYS_LLM_FIELDS:
        expected_str += (
            "IMPORTANT: This is a material_type or acquisition_method field. "
            "The value may be reasonably inferable from the experimental context "
            "even when not explicitly stated in the paper text. "
            "Do NOT mark as HALLUCINATED solely because the exact term is absent. "
            "Assess whether the value is a reasonable inference from the experiment. ")
    expected_str += (
        f"Reference values for MATCHED_REFERENCE tracking only: "
        f"{'; '.join(gt_candidates)}"
        if gt_candidates else "No reference values provided.")

    test_case = LLMTestCase(
        input=f"Assess field='{field_name}' candidate_value='{extracted_value}'",
        actual_output=f"field_name: {field_name}\ncandidate_value: {extracted_value}",
        expected_output=expected_str)

    reason = ""
    score  = None
    try:
        metric = _get_thread_annotation_metric()
        metric.measure(test_case)
        reason = metric.reason or ""
        score  = metric.score
    except RecursionError:
        print(f"  [GEval recursion] {field_name}: '{extracted_value[:40]}' "
              f"-- falling back to direct API call")
        reason = model._generate_single(
            model._paper_text, model._current_paper_id, context)
        if not reason:
            reason = (
                "TYPE CHECK: Unable to evaluate (no model response).\n"
                "SOURCE CHECK: N/A\nTRUTH CHECK: N/A\nCOMPLETENESS CHECK: N/A\n"
                "MATCHED_REFERENCE: NONE\nTYPE_CORRECT: yes\nCORRECT_TYPE_NAME: NONE\n"
                "VALUE_CORRECT: no\nVALUE_COMPLETE: no\n"
                "HALLUCINATED: no\nVERDICT: low\nCORRECTED_VALUE: NONE")
        print(f"  [direct-fallback response] {reason[:200]}...")
    except Exception as exc:
        print(f"  [GEval error] {field_name}: {type(exc).__name__}: {exc}")
        try:
            reason = model._generate_single(
                model._paper_text, model._current_paper_id, context)
            if reason:
                print(f"  [GEval->direct fallback] Got response for {field_name}: "
                      f"'{extracted_value[:40]}'")
            else:
                raise ValueError("Empty response from direct fallback")
        except Exception as fallback_exc:
            print(f"  [direct fallback also failed] {fallback_exc}")
            return {"verdict": None, "type_mismatch": None, "correct_type": None,
                "value_correct": None, "value_complete": None, "hallucination": None,
                "matched_reference": None, "corrected_value": None,
                "issue_summary": f"GEval error: {exc}",
                "match_type": "ERROR", "any_match": False, "geval_score": None}

    verdict_match = re.search(r"VERDICT:\s*(high|medium|low)", reason, re.IGNORECASE)
    verdict = verdict_match.group(1).lower() if verdict_match else None
    hall_parsed = _parse_yes_no(reason, "HALLUCINATED")
    is_hallucinated = bool(hall_parsed) if hall_parsed is not None else False

    type_correct_parsed = _parse_type_correct(reason)
    if type_correct_parsed is not None:
        is_mismatch = not type_correct_parsed
    else:
        is_mismatch = (verdict == "low") and not is_hallucinated and bool(
            re.search(r"type mismatch|wrong (?:field|type)|incorrect type|belongs to",
                      reason, re.IGNORECASE))
    correct_type = _parse_correct_type_name(reason, field_name) if is_mismatch else None

    matched_ref_raw   = _parse_matched_reference(reason)
    matched_reference = _resolve_matched_reference(matched_ref_raw, gt_candidates)

    vc_parsed    = _parse_yes_no(reason, "VALUE_CORRECT")
    vcomp_parsed = _parse_yes_no(reason, "VALUE_COMPLETE")
    value_correct  = vc_parsed if vc_parsed is not None else (
        (verdict in ("high", "medium")) and not is_hallucinated and not is_mismatch)
    value_complete = vcomp_parsed if vcomp_parsed is not None else (
        (verdict == "high") and not is_hallucinated and not is_mismatch)

    if is_hallucinated or is_mismatch:
        value_correct = False
    if is_hallucinated:
        value_complete = False

    if (field_name.lower() in CONCENTRATION_FIELDS and value_correct
            and not is_hallucinated and not is_mismatch and not value_complete):
        reagent_incomplete = bool(re.search(
            r"(?:lacks?|missing|without|does not include|omits?)\s+"
            r"(?:the\s+)?(?:reagent|DTT|TCEP|IAA|iodoacetamide|dithiothreitol|"
            r"chemical|compound)\s*(?:name)?", reason, re.IGNORECASE))
        if reagent_incomplete or _is_concentration_only(field_name, extracted_value):
            value_complete = True
            verdict = "high"
            reason += ("\n[POST-PROCESSING OVERRIDE] Concentration field: reagent name "
                "is extracted separately in companion field. Value with only "
                "numeric quantity + unit is COMPLETE. Verdict upgraded to HIGH.")

    corrected_value = _parse_corrected_value(reason) if verdict == "medium" else None

    if is_hallucinated:
        match_type = "HALLUCINATED"
    elif is_mismatch:
        match_type = "TYPE_MISMATCH"
    elif verdict == "high":
        match_type = "CORRECT"
    elif verdict == "medium":
        match_type = "PARTIAL"
    else:
        match_type = "NO_MATCH"

    return {"verdict": verdict, "type_mismatch": is_mismatch,
        "correct_type": correct_type, "value_correct": value_correct,
        "value_complete": value_complete, "hallucination": is_hallucinated,
        "matched_reference": matched_reference, "corrected_value": corrected_value,
        "issue_summary": reason, "match_type": match_type,
        "any_match": match_type in ("CORRECT", "PARTIAL"), "geval_score": score}


def check_missed_annotation(field_name):
    model = _get_judge_model()
    context = {"task": "check_missing", "field_name": field_name,
        "field_definition": _field_def_for(field_name)}
    model.set_entity_context(context)
    test_case = LLMTestCase(
        input=f"Is '{field_name}' present in the source text?",
        actual_output=f"Field '{field_name}' was NOT extracted by the pipeline.",
        expected_output="Determine if the source text contains this field.")
    metric = _get_thread_missed_metric()
    try:
        metric.measure(test_case)
        reason = metric.reason or ""
        m = re.search(r"VERDICT:\s*(high|medium|low)", reason, re.IGNORECASE)
        return m.group(1).lower() if m else "low"
    except Exception as exc:
        print(f"  [missed GEval error] {field_name}: {exc}")
        return "low"


def _substr_covers(gt_v, predictions):
    r = _norm(gt_v)
    if not r or len(r) < 4:
        return False
    return any(r in _norm(pred) for pred in predictions)


def _geval_covers(field_name, gt_v, predictions):
    model = _get_judge_model()
    context = {"task": "check_coverage", "field_name": field_name,
        "field_definition": _field_def_for(field_name),
        "gt_value": gt_v, "candidate_values": predictions[:20]}
    model.set_entity_context(context)
    test_case = LLMTestCase(
        input=f"Is '{gt_v}' covered by any candidate for {field_name}?",
        actual_output=f"Candidate values: {'; '.join(predictions[:20])}",
        expected_output=f"Determine if any candidate covers the reference: {gt_v}")
    metric = _get_thread_coverage_metric()
    try:
        metric.measure(test_case)
        reason = metric.reason or ""
        m = re.search(r"COVERED:\s*(yes|no)", reason, re.IGNORECASE)
        covered = bool(m and m.group(1).lower() == "yes")
        print(f"  [coverage-geval] {field_name}: '{gt_v[:50]}' -> COVERED={covered}")
        return covered
    except Exception as exc:
        print(f"  [coverage-geval error] {field_name}: '{gt_v[:40]}': {exc}")
        return False


def _second_pass_coverage(field_name, gt_v, predictions):
    if _substr_covers(gt_v, predictions):
        print(f"  [coverage-substr] {field_name}: '{gt_v[:50]}' covered by substring")
        return True
    return _geval_covers(field_name, gt_v, predictions)


def _is_golden_skip_value(val: str) -> bool:
    if not val:
        return True
    return val.strip().lower() in _SDRF_SKIP_VALUES


def evaluate_paper_with_geval(paper_id, tier_rows, all_golden, source_text):
    _get_judge_model().set_paper_context(source_text, paper_id=paper_id)

    gt_lists: dict[str, list[str]] = {}
    for k, v_list in all_golden.items():
        clean = [s.strip() for s in v_list if s and s.strip()]
        if clean:
            gt_lists[k] = clean

    covered_gt_refs: dict[str, set] = defaultdict(set)

    work_items = []
    shortcut_results = {}

    for idx, tr in enumerate(tier_rows):
        field_name      = tr["annotation_type"]
        extracted_value = tr["extracted_value"]
        golden_value    = tr["golden_value"] or ""
        gt_candidates   = gt_lists.get(field_name,
                                       ([golden_value] if golden_value else []))

        if not extracted_value or _is_not_extracted(str(extracted_value)):
            golden_meaningful = (golden_value
                                 and not _is_golden_skip_value(golden_value)
                                 and field_name.lower() not in SKIP_ANNOTATION_FIELDS)
            if golden_meaningful:
                tr["extracted_value"] = "MISSING"
                shortcut_results[idx] = {
                    "verdict": "missing", "value_correct": False,
                    "value_complete": False, "hallucination": False,
                    "type_mismatch": False, "correct_type": None,
                    "corrected_value": golden_value,
                    "matched_reference": None,
                    "issue_summary": "Not extracted — golden value exists",
                    "match_type": "MISSING", "any_match": False,
                    "geval_score": None}
            else:
                shortcut_results[idx] = {
                    "verdict": None, "value_correct": None, "value_complete": None,
                    "hallucination": None, "type_mismatch": None, "correct_type": None,
                    "corrected_value": None, "matched_reference": None,
                    "issue_summary": "Skipped (sentinel / empty — not extracted)",
                    "match_type": "SKIPPED", "any_match": False, "geval_score": None}
            continue

        if tr.get("tier1_exact"):
            result = _exact_match_result(golden_value or extracted_value)
            if golden_value:
                covered_gt_refs[field_name].add(golden_value)
            shortcut_results[idx] = result
            continue

        if (extracted_value.strip().lower() in SAFE_DEFAULTS.get(field_name, set())
                and field_name.lower() not in _ALWAYS_LLM_FIELDS):
            shortcut_results[idx] = _safe_default_result(field_name, extracted_value)
            continue

        matched = _find_matching_reference(extracted_value, gt_candidates)
        if matched is not None:
            result = _exact_match_result(matched)
            covered_gt_refs[field_name].add(matched)
            shortcut_results[idx] = result
            continue

        work_items.append((idx, field_name, extracted_value, gt_candidates))

    n_shortcuts  = len(shortcut_results)
    n_llm_needed = len(work_items)
    n_total      = len(tier_rows)
    print(f"    [judge] {n_total} annotations: {n_shortcuts} shortcuts, "
          f"{n_llm_needed} need LLM evaluation with {MAX_WORKERS} workers...")

    llm_results: dict[int, dict] = {}

    if work_items:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for idx, field_name, extracted_value, gt_candidates in work_items:
                future = executor.submit(
                    _run_geval_semantic, field_name, extracted_value, gt_candidates)
                futures[future] = (idx, field_name, extracted_value, gt_candidates)

            for future in as_completed(futures):
                idx, field_name, extracted_value, gt_candidates = futures[future]
                try:
                    result = future.result()
                    llm_results[idx] = result
                    mref = result.get("matched_reference")
                    if mref:
                        covered_gt_refs[field_name].add(mref)
                    if result.get("verdict") is not None:
                        cv_info = ""
                        if result.get("corrected_value"):
                            cv_info = f"  corrected='{result['corrected_value'][:30]}'"
                        print(f"      [{idx+1}/{n_total}] {field_name}: "
                              f"'{extracted_value[:40]}' "
                              f"-> verdict={result['verdict']}  "
                              f"correct={result['value_correct']}  "
                              f"halluc={result['hallucination']}  "
                              f"mismatch={result['type_mismatch']}  "
                              f"score={result.get('geval_score')}{cv_info}")
                    else:
                        print(f"      [{idx+1}/{n_total}] {field_name}: "
                              f"'{extracted_value[:40]}' -> ERROR")
                except Exception as exc:
                    print(f"      [{idx+1}/{n_total}] {field_name}: "
                          f"'{extracted_value[:40]}' -> ERROR: {exc}")
                    llm_results[idx] = {
                        "verdict": None, "type_mismatch": None,
                        "correct_type": None, "value_correct": None,
                        "value_complete": None, "hallucination": None,
                        "matched_reference": None, "corrected_value": None,
                        "issue_summary": f"Thread error: {exc}",
                        "match_type": "ERROR", "any_match": False,
                        "geval_score": None}

    all_results = {**shortcut_results, **llm_results}
    for idx, tr in enumerate(tier_rows):
        result = all_results.get(idx)
        if result is None:
            continue
        tr["verdict"]         = result.get("verdict")
        tr["value_correct"]   = result.get("value_correct")
        tr["value_complete"]  = result.get("value_complete")
        tr["hallucination"]   = result.get("hallucination")
        tr["type_mismatch"]   = result.get("type_mismatch")
        tr["correct_type"]    = result.get("correct_type")
        tr["corrected_value"] = result.get("corrected_value")
        tr["issue_summary"]   = result.get("issue_summary", "")

    return tier_rows, covered_gt_refs


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


def _flatten_extraction_value(val) -> str | None:
    if isinstance(val, str):
        s = val.strip()
        return None if _is_not_extracted(s) else s
    if isinstance(val, list):
        extracted_values = []
        for item in val:
            if isinstance(item, list):
                if len(item) >= 1:
                    extracted_values.append(str(item[0]).strip())
            elif isinstance(item, str):
                extracted_values.append(item.strip())
            else:
                extracted_values.append(str(item).strip())
        for v in extracted_values:
            if not _is_not_extracted(v):
                return v
        return None
    s = str(val).strip()
    return None if _is_not_extracted(s) else s


def load_llm_output_json(extraction_dir, pxd_id, agent):
    suffix    = AGENT_FILE_SUFFIX.get(agent, agent.lower())
    agent_dir = os.path.join(extraction_dir, agent)
    if not os.path.exists(agent_dir):
        print(f"    WARNING: agent dir not found: {agent_dir}")
        return {}
    raw_data = None
    for cname in [f"{pxd_id}_{suffix}.json", f"{pxd_id}.json"]:
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
        if field.startswith("_") or field not in allowed_fields:
            continue
        extracted = _flatten_extraction_value(val)
        if extracted is None:
            print(f"    [{agent}] {field}: all values are sentinels — skipping")
            continue
        parsed[FIELD_NAME_MAP.get(field, field)] = extracted
    return parsed


def expand_match_type(match_type):
    mt = (match_type or "").upper().strip()
    exact = (mt == "EXACT"); normalized = (mt == "NORMALIZED"); ontology = (mt == "ONTOLOGY")
    hierarchical = (mt == "HIERARCHICAL"); semantic = (mt == "SEMANTIC")
    matched = any([exact, normalized, ontology, hierarchical, semantic])
    return {"tier1_exact": exact, "tier2_normalized": normalized,
        "tier3_ontology": ontology, "tier4_hierarchical": hierarchical,
        "tier5_semantic": semantic, "any_match": matched, "no_match": not matched}


def load_detailed_csv(model_result_dir):
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


_REVIEW_CSV_FIELDS = [
    "paper_id", "agent", "annotation_type", "extracted_value",
    "golden_value", "match_score", "match_type",
    "tier1_exact", "tier2_normalized", "tier3_ontology",
    "tier4_hierarchical", "tier5_semantic", "any_match", "no_match",
    "verdict", "has_golden", "type_mismatch", "correct_type",
    "value_correct", "value_complete", "hallucination",
    "source_evidence", "issue_summary", "corrected_value",
]

_STATS_CSV_FIELDS = [
    "paper_id", "total_predicted", "total_missed",
    "exact", "normalized", "ontology", "hierarchical", "semantic", "no_match",
    "match_rate", "avg_score",
    "judge_n_correct", "judge_n_hallucinated", "judge_n_mismatch",
    "judge_n_wrong", "judge_n_incomplete", "judge_n_missed", "judge_n_corrected",
    "judge_accuracy",
]

_COVERAGE_CSV_FIELDS = [
    "paper_id", "agent", "pipeline_field", "mapped_field",
    "extracted_value", "was_extracted", "has_golden", "golden_value",
]


class _IncrementalCSV:

    def __init__(self, path: str, fieldnames: list[str]):
        self._path = path
        self._fields = fieldnames
        import csv as _csv
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = _csv.DictWriter(
            self._fh, fieldnames=fieldnames,
            quoting=_csv.QUOTE_ALL, extrasaction="ignore",
        )
        self._writer.writeheader()
        self._fh.flush()

    def append_rows(self, rows: list[dict]) -> None:
        for row in rows:
            self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _compute_single_paper_stats(paper_id: str, tier_rows: list[dict]) -> dict:
    pred_rows = [r for r in tier_rows if r.get("extracted_value") != "MISSING"]
    miss_rows = [r for r in tier_rows if r.get("extracted_value") == "MISSING"]
    total     = len(pred_rows)

    rec = {
        "paper_id": paper_id,
        "total_predicted": total,
        "total_missed": len(miss_rows),
        "exact": sum(1 for r in pred_rows if r.get("tier1_exact")),
        "normalized": sum(1 for r in pred_rows if r.get("tier2_normalized")),
        "ontology": sum(1 for r in pred_rows if r.get("tier3_ontology")),
        "hierarchical": sum(1 for r in pred_rows if r.get("tier4_hierarchical")),
        "semantic": sum(1 for r in pred_rows if r.get("tier5_semantic")),
        "no_match": sum(1 for r in pred_rows if r.get("no_match")),
        "match_rate": 0.0,
        "avg_score": 0.0,
    }

    if total > 0:
        match_count = sum(1 for r in pred_rows if r.get("any_match"))
        rec["match_rate"] = round(match_count / total, 4)
        scores = [r["match_score"] for r in pred_rows
                  if r.get("match_score") is not None]
        if scores:
            rec["avg_score"] = round(sum(scores) / len(scores), 4)

    if USE_LLM_JUDGE:
        judged_rows = [r for r in pred_rows if r.get("value_correct") is not None]
        if judged_rows:
            n_correct = sum(1 for r in pred_rows
                if r.get("type_mismatch") == False
                and r.get("value_correct") == True
                and r.get("value_complete") == True
                and r.get("hallucination") == False)
            n_halluc = sum(1 for r in pred_rows if r.get("hallucination"))
            n_mismatch = sum(1 for r in pred_rows if r.get("type_mismatch"))
            n_wrong = sum(1 for r in pred_rows
                if r.get("value_correct") == False
                and not r.get("hallucination"))
            n_incomplete = sum(1 for r in pred_rows
                if r.get("value_correct") == True
                and r.get("value_complete") == False)
            n_corrected = sum(1 for r in pred_rows if r.get("corrected_value"))

            rec["judge_n_correct"]      = n_correct
            rec["judge_n_hallucinated"] = n_halluc
            rec["judge_n_mismatch"]     = n_mismatch
            rec["judge_n_wrong"]        = n_wrong
            rec["judge_n_incomplete"]   = n_incomplete
            rec["judge_n_missed"]       = len(miss_rows)
            rec["judge_n_corrected"]    = n_corrected
            rec["judge_accuracy"]       = round(n_correct / total, 4) if total > 0 else 0.0

    return rec


def _make_missed_row(paper_id, agent, field, golden_val, summary):
    return {"paper_id": paper_id, "agent": agent, "annotation_type": field,
        "extracted_value": "MISSING", "golden_value": golden_val,
        "match_score": 0.0, "match_type": "NO_MATCH",
        "tier1_exact": False, "tier2_normalized": False,
        "tier3_ontology": False, "tier4_hierarchical": False,
        "tier5_semantic": False, "any_match": False, "no_match": True,
        "verdict": "missing", "has_golden": bool(golden_val),
        "type_mismatch": False, "correct_type": None,
        "value_correct": False, "value_complete": False,
        "hallucination": False, "source_evidence": "",
        "issue_summary": summary, "corrected_value": golden_val or None,
        "row_type": "missed"}

def process_model(model_name, model_result_dir, extraction_dir, out_dir):
    detailed_df = load_detailed_csv(model_result_dir)
    if detailed_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    print(f"  Loaded {len(detailed_df)} rows from sdrf_benchmark_detailed.csv")
    pxd_ids = sorted(detailed_df["pxd_id"].unique())
    print(f"  Papers: {len(pxd_ids)}")
    if LIMIT and LIMIT < len(pxd_ids):
        pxd_ids = pxd_ids[:LIMIT]
        print(f"  Limit applied: running only first {LIMIT} papers: {pxd_ids}")

    n_papers = len(pxd_ids)

    review_csv = _IncrementalCSV(
        os.path.join(out_dir, "llm_judge_annotation_review.csv"),
        _REVIEW_CSV_FIELDS,
    )
    stats_csv = _IncrementalCSV(
        os.path.join(out_dir, "llm_judge_per_paper.csv"),
        _STATS_CSV_FIELDS,
    )
    coverage_csv = _IncrementalCSV(
        os.path.join(out_dir, "llm_judge_coverage.csv"),
        _COVERAGE_CSV_FIELDS,
    )

    print()
    print("Output mode: INCREMENTAL")
    print("  -> CSVs are written after EACH paper completes.")
    print("  -> You can inspect results on disk while the pipeline is still running.")
    print(f"  -> Review CSV   : {os.path.join(out_dir, 'llm_judge_annotation_review.csv')}")
    print(f"  -> Per-paper CSV : {os.path.join(out_dir, 'llm_judge_per_paper.csv')}")
    print(f"  -> Coverage CSV  : {os.path.join(out_dir, 'llm_judge_coverage.csv')}")
    print()

    all_rows = []
    all_coverage_rows = []
    per_file_stats = []

    for paper_idx, pxd_id in enumerate(pxd_ids, 1):
        t_start = time.time()
        print(f"  PAPER [{paper_idx}/{n_papers}]: {pxd_id}")

        source_text = load_manuscript(pxd_id)
        all_predicted: dict[str, dict] = {}
        for agent in AGENTS:
            pred = load_llm_output_json(extraction_dir, pxd_id, agent)
            if pred:
                all_predicted[agent] = pred
        pxd_df = detailed_df[detailed_df["pxd_id"] == pxd_id].copy()
        golden_lists: dict[str, list[str]] = load_golden_from_sdrf(pxd_id)
        flat_golden: dict[str, str] = _flatten_golden_lists(golden_lists)

        coverage_rows = []
        for agent in AGENTS:
            pred = all_predicted.get(agent, {})
            for raw_field in PIPELINE_FIELDS.get(agent, []):
                mapped     = FIELD_NAME_MAP.get(raw_field, raw_field)
                extracted  = pred.get(mapped) or pred.get(raw_field)
                golden_val = flat_golden.get(mapped) or flat_golden.get(raw_field)
                was_extracted = bool(extracted and not _is_not_extracted(str(extracted)))
                coverage_rows.append({"paper_id": pxd_id, "agent": agent,
                    "pipeline_field": raw_field, "mapped_field": mapped,
                    "extracted_value": extracted or "", "was_extracted": was_extracted,
                    "has_golden": bool(golden_val), "golden_value": golden_val or ""})
        all_coverage_rows.extend(coverage_rows)

        coverage_csv.append_rows(coverage_rows)

        n_extracted  = sum(1 for r in coverage_rows if r["was_extracted"])
        n_has_golden = sum(1 for r in coverage_rows if r["has_golden"])
        n_both       = sum(1 for r in coverage_rows if r["was_extracted"] and r["has_golden"])
        total_fields = len(coverage_rows)
        print(f"    Coverage: {n_extracted}/{total_fields} extracted  "
              f"| {n_has_golden}/{total_fields} have golden  "
              f"| {n_both}/{total_fields} overlap")

        tier_rows = []
        for _, row in pxd_df.iterrows():
            if row["field"].lower() in SKIP_ANNOTATION_FIELDS:
                continue
            golden_val_raw = row["golden"]
            if _is_golden_skip_value(golden_val_raw):
                golden_val_raw = ""
            tier = expand_match_type(row["match_type"])
            tier_rows.append({"paper_id": pxd_id, "agent": row["agent"],
                "annotation_type": row["field"], "extracted_value": row["llm"],
                "golden_value": golden_val_raw,
                "match_score": round(float(row["score"]), 4),
                "match_type": row["match_type"],
                "tier1_exact": tier["tier1_exact"],
                "tier2_normalized": tier["tier2_normalized"],
                "tier3_ontology": tier["tier3_ontology"],
                "tier4_hierarchical": tier["tier4_hierarchical"],
                "tier5_semantic": tier["tier5_semantic"],
                "any_match": tier["any_match"], "no_match": tier["no_match"],
                "verdict": None, "has_golden": bool(golden_val_raw),
                "type_mismatch": None, "correct_type": None,
                "value_correct": None, "value_complete": None,
                "hallucination": None, "source_evidence": "",
                "issue_summary": "", "corrected_value": None,
                "row_type": "sdrf_match"})
        sdrf_fields_this_paper = set(pxd_df["field"].tolist())
        for agent, pred_dict in all_predicted.items():
            for field, val in pred_dict.items():
                if field in sdrf_fields_this_paper or _is_not_extracted(str(val)):
                    continue
                if field.lower() in SKIP_ANNOTATION_FIELDS:
                    continue
                golden_val = flat_golden.get(field, "")
                if _is_golden_skip_value(golden_val):
                    golden_val = ""
                tier_rows.append({"paper_id": pxd_id, "agent": agent,
                    "annotation_type": field, "extracted_value": val,
                    "golden_value": golden_val, "match_score": None,
                    "match_type": "NOT_IN_SDRF",
                    "tier1_exact": None, "tier2_normalized": None,
                    "tier3_ontology": None, "tier4_hierarchical": None,
                    "tier5_semantic": None, "any_match": None, "no_match": None,
                    "verdict": None, "has_golden": bool(golden_val),
                    "type_mismatch": None, "correct_type": None,
                    "value_correct": None, "value_complete": None,
                    "hallucination": None, "source_evidence": "",
                    "issue_summary": "", "corrected_value": None,
                    "row_type": "judge_only"})

        if USE_LLM_JUDGE and source_text:
            tier_rows, covered_gt_refs = evaluate_paper_with_geval(
                pxd_id, tier_rows, golden_lists, source_text)

            extracted_fields = {
                tr["annotation_type"].lower()
                for tr in tier_rows
                if tr["extracted_value"]
                and tr["extracted_value"] != "MISSING"
                and not _is_not_extracted(str(tr["extracted_value"]))}

            missed_rows = []
            for field, golden_vals in golden_lists.items():
                if field.lower() in extracted_fields:
                    continue
                if field.lower() in SKIP_MISSED_FIELDS:
                    continue
                golden_vals_filtered = [
                    gv for gv in golden_vals
                    if gv.strip().lower() not in _SDRF_SKIP_VALUES]
                if not golden_vals_filtered:
                    print(f"        [skip] {field}: all golden values are "
                          f"'not applicable' variants — not missing")
                    continue
                if field.lower() == "technical_replicates":
                    numeric_golden = []
                    for gv in golden_vals_filtered:
                        try:
                            numeric_golden.append(int(float(gv)))
                        except (ValueError, TypeError):
                            pass
                    if numeric_golden:
                        golden_vals_filtered = [str(max(numeric_golden))]
                preds_for_field = [
                    tr["extracted_value"]
                    for tr in tier_rows
                    if tr["annotation_type"].lower() == field.lower()
                    and tr["extracted_value"]
                    and tr["extracted_value"] != "MISSING"
                    and not _is_not_extracted(str(tr["extracted_value"]))]
                for golden_val in golden_vals_filtered:
                    if not golden_val:
                        continue
                    if preds_for_field:
                        if _second_pass_coverage(field, golden_val, preds_for_field):
                            continue
                    if field in covered_gt_refs and golden_val in covered_gt_refs[field]:
                        continue
                    already_missing = any(
                        tr["annotation_type"].lower() == field.lower()
                        and tr["extracted_value"] == "MISSING"
                        and tr.get("golden_value", "").strip().lower() == golden_val.strip().lower()
                        for tr in tier_rows)
                    if already_missing:
                        continue
                    print(f"        [missed] {field}: golden='{golden_val}' not extracted - MISSING")
                    sdrf_agent = FIELD_TO_AGENT.get(field, "BiologicalAgent")
                    missed_rows.append(_make_missed_row(
                        pxd_id, sdrf_agent, field, golden_val,
                        f"Not extracted — golden value exists in SDRF"))
            if missed_rows:
                print(f"    [missed] Added {len(missed_rows)} MISSING rows")
                tier_rows.extend(missed_rows)

        review_csv.append_rows(tier_rows)

        paper_stat = _compute_single_paper_stats(pxd_id, tier_rows)
        per_file_stats.append(paper_stat)
        stats_csv.append_rows([paper_stat])

        all_rows.extend(tier_rows)

        elapsed = time.time() - t_start
        n_pred = paper_stat.get("total_predicted", 0)
        n_miss = paper_stat.get("total_missed", 0)
        print(f"\n  ✓ PAPER COMPLETE: {pxd_id}  ({elapsed:.1f}s)")
        print(f"    predicted={n_pred}  missed={n_miss}")
        if USE_LLM_JUDGE and "judge_n_correct" in paper_stat:
            print(f"    verdicts: correct={paper_stat.get('judge_n_correct', 0)}  "
                  f"halluc={paper_stat.get('judge_n_hallucinated', 0)}  "
                  f"mismatch={paper_stat.get('judge_n_mismatch', 0)}  "
                  f"wrong={paper_stat.get('judge_n_wrong', 0)}  "
                  f"incomplete={paper_stat.get('judge_n_incomplete', 0)}  "
                  f"corrected={paper_stat.get('judge_n_corrected', 0)}")
            print(f"    accuracy={paper_stat.get('judge_accuracy', 0):.1%}")
        print(f"    results written to disk — you can inspect them now.")

    review_csv.close()
    stats_csv.close()
    coverage_csv.close()

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    coverage_df = pd.DataFrame(all_coverage_rows) if all_coverage_rows else pd.DataFrame()
    df = pd.DataFrame(all_rows)
    final_cols = _REVIEW_CSV_FIELDS
    for c in final_cols:
        if c not in df.columns:
            df[c] = None

    per_paper_df = pd.DataFrame(per_file_stats) if per_file_stats else pd.DataFrame()

    return df[final_cols], coverage_df, per_paper_df


def compute_per_paper_stats(df):
    if df.empty:
        return pd.DataFrame()
    records = []
    for paper_id in sorted(df["paper_id"].unique()):
        pxd_df  = df[df["paper_id"] == paper_id]
        pred_df = pxd_df[pxd_df["extracted_value"] != "MISSING"]
        miss_df = pxd_df[pxd_df["extracted_value"] == "MISSING"]
        total   = len(pred_df)
        rec = {"paper_id": paper_id, "total_predicted": total,
            "total_missed": len(miss_df),
            "exact": int(pred_df["tier1_exact"].sum()),
            "normalized": int(pred_df["tier2_normalized"].sum()),
            "ontology": int(pred_df["tier3_ontology"].sum()),
            "hierarchical": int(pred_df["tier4_hierarchical"].sum()),
            "semantic": int(pred_df["tier5_semantic"].sum()),
            "no_match": int(pred_df["no_match"].sum()),
            "match_rate": round(float(pred_df["any_match"].mean()), 4) if total > 0 else 0.0,
            "avg_score": round(float(pred_df["match_score"].dropna().mean()), 4)
                if pred_df["match_score"].notna().any() else 0.0}
        if USE_LLM_JUDGE and pred_df["value_correct"].notna().any():
            n_correct = int(pred_df[
                (pred_df["type_mismatch"] == False) &
                (pred_df["value_correct"]  == True)  &
                (pred_df["value_complete"] == True)  &
                (pred_df["hallucination"]  == False)].shape[0])
            n_halluc     = int(pred_df["hallucination"].fillna(False).sum())
            n_mismatch   = int(pred_df["type_mismatch"].fillna(False).sum())
            n_wrong      = int(pred_df[
                (pred_df["value_correct"] == False) &
                (pred_df["hallucination"] == False)].shape[0])
            n_incomplete = int(pred_df[
                (pred_df["value_correct"]  == True) &
                (pred_df["value_complete"] == False)].shape[0])
            n_corrected  = int(pred_df["corrected_value"].notna().sum())
            rec["judge_n_correct"]      = n_correct
            rec["judge_n_hallucinated"] = n_halluc
            rec["judge_n_mismatch"]     = n_mismatch
            rec["judge_n_wrong"]        = n_wrong
            rec["judge_n_incomplete"]   = n_incomplete
            rec["judge_n_missed"]       = len(miss_df)
            rec["judge_n_corrected"]    = n_corrected
            rec["judge_accuracy"]       = round(n_correct / total, 4) if total > 0 else 0.0
        records.append(rec)
    return pd.DataFrame(records)


def plot_results(df, per_paper_df, model_name, out_dir):
    BG, GRID_C, TEXT_C, SUB_C, SPINE_C = "white", "#E5E7EB", "#111827", "#6B7280", "#D1D5DB"
    CLR = dict(correct="#27ae60", hall="#e74c3c", mismatch="#9b59b6",
               wrong="#e67e22", incomplete="#3498db", missed="#95a5a6")
    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.yaxis.grid(True, color=GRID_C, lw=0.8, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: ax.spines[sp].set_color(SPINE_C)
        ax.tick_params(colors=TEXT_C, labelsize=9)

    pred_df     = df[df["extracted_value"] != "MISSING"]
    tier_cols   = ["tier1_exact", "tier2_normalized", "tier3_ontology",
                   "tier4_hierarchical", "tier5_semantic", "no_match"]
    tier_labels = ["Exact", "Normalized", "Ontology", "Hierarchical", "Semantic", "No Match"]
    tier_colors = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899", "#EF4444"]
    tier_totals = [int(pred_df[c].sum()) for c in tier_cols]
    grand       = sum(tier_totals)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG); style_ax(ax)
    bars = ax.bar(tier_labels, tier_totals, color=tier_colors,
                  edgecolor="white", linewidth=1.0, alpha=0.9, zorder=3)
    for bar, val in zip(bars, tier_totals):
        pct = val / grand * 100 if grand > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f"{val}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT_C)
    ax.set_ylabel("Count", fontsize=11, color=SUB_C)
    ax.set_title(f"{model_name.upper()} — 5-Tier Match Distribution "
                 f"({grand} total annotations)",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_tier_distribution.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved: llm_judge_tier_distribution.png")

    if per_paper_df.empty or "judge_accuracy" not in per_paper_df.columns:
        return
    required = ["judge_n_correct", "judge_n_hallucinated", "judge_n_mismatch",
                "judge_n_wrong", "judge_n_incomplete", "judge_n_missed"]
    if not all(c in per_paper_df.columns for c in required):
        return

    names = per_paper_df["paper_id"].tolist(); x = np.arange(len(names)); w = 0.6
    n_correct    = per_paper_df["judge_n_correct"].fillna(0).values
    n_hall       = per_paper_df["judge_n_hallucinated"].fillna(0).values
    n_mismatch   = per_paper_df["judge_n_mismatch"].fillna(0).values
    n_wrong      = per_paper_df["judge_n_wrong"].fillna(0).values
    n_incomplete = per_paper_df["judge_n_incomplete"].fillna(0).values
    n_missed     = per_paper_df["judge_n_missed"].fillna(0).values
    totals       = per_paper_df["total_predicted"].fillna(0).values

    fig, ax = plt.subplots(figsize=(max(16, len(names) * 0.7), 7), facecolor=BG); style_ax(ax)
    b = np.zeros(len(names))
    for arr, col, lbl in [
        (n_correct, CLR["correct"], "Correct"), (n_hall, CLR["hall"], "Hallucinated"),
        (n_mismatch, CLR["mismatch"], "Type Mismatch"), (n_wrong, CLR["wrong"], "Wrong Value"),
        (n_incomplete, CLR["incomplete"], "Incomplete"), (n_missed, CLR["missed"], "Missed")]:
        ax.bar(x, arr, w, bottom=b, label=lbl, color=col, edgecolor="white", lw=0.5, alpha=0.92)
        b += arr
    for i, (tot, cor) in enumerate(zip(totals, n_correct)):
        ax.text(i, b[i] + 0.3, f"predicted={int(tot)}  missed={int(n_missed[i])}",
                ha="center", va="bottom", fontsize=7, fontweight="bold", color=TEXT_C)
        if cor >= 2:
            ax.text(i, cor / 2, str(int(cor)), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TEXT_C)
    ax.set_ylabel("Number of Annotations", fontsize=11, color=SUB_C)
    ax.set_title(f"{model_name.upper()} — Annotation Quality per Paper (counts)",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=SPINE_C, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_annotation_quality_counts.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved: llm_judge_annotation_quality_counts.png")

    acc_vals = per_paper_df["judge_accuracy"].tolist(); overall = float(np.mean(acc_vals))
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.55), 6), facecolor=BG); style_ax(ax)
    bar_colors = ["#27ae60" if v >= 0.7 else "#e67e22" if v >= 0.5 else "#e74c3c"
                  for v in acc_vals]
    bars = ax.bar(x, acc_vals, color=bar_colors, alpha=0.88,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.012,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=TEXT_C)
    ax.axhline(overall, color="#2471a3", lw=2, linestyle="--", label=f"Mean = {overall:.3f}")
    ax.axhline(0.7, color="#e74c3c", lw=1.5, linestyle=":", alpha=0.7, label="Threshold = 0.70")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TEXT_C)
    ax.set_ylabel("Accuracy (Correct / Total Predicted)", fontsize=11, color=SUB_C)
    ax.set_ylim([0, 1.18])
    ax.set_title(f"{model_name.upper()} — LLM Judge Accuracy per Paper",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=SPINE_C)
    ax.text(0.98, 0.97, f"Overall: {overall:.1%}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold", color="#27ae60",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                      edgecolor="#27ae60", alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_accuracy.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved: llm_judge_accuracy.png")

    agg_labels  = ["Correct", "Hallucinated", "Type Mismatch",
                   "Wrong Value", "Incomplete", "Missed"]
    agg_vals    = [int(n_correct.sum()), int(n_hall.sum()), int(n_mismatch.sum()),
                   int(n_wrong.sum()), int(n_incomplete.sum()), int(n_missed.sum())]
    agg_colors  = [CLR["correct"], CLR["hall"], CLR["mismatch"],
                   CLR["wrong"], CLR["incomplete"], CLR["missed"]]
    grand_total = sum(agg_vals)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG); style_ax(ax)
    bars5 = ax.bar(agg_labels, agg_vals, color=agg_colors,
                   edgecolor="white", linewidth=1.0, alpha=0.92, zorder=3)
    for bar, val in zip(bars5, agg_vals):
        pct = val / grand_total * 100 if grand_total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_C)
    overall_acc = agg_vals[0] / grand_total * 100 if grand_total > 0 else 0
    ax.set_ylabel("Total Count (all papers)", fontsize=11, color=SUB_C)
    ax.set_title(f"{model_name.upper()} — Aggregate Annotation Quality "
                 f"(total = {grand_total})",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.text(0.5, 0.97, f"Overall accuracy: {overall_acc:.1f}%",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold", color=CLR["correct"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                      edgecolor=CLR["correct"], alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_aggregate.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved: llm_judge_aggregate.png")


def main():
    global USE_LLM_JUDGE, LIMIT, MAX_WORKERS

    parser = argparse.ArgumentParser(
        description="SDRF benchmark + Gemma-4-31B LLM judge (high/medium/low verdicts)")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_RESULT_DIRS.keys()),
                        help="Model: claude / gpt / gemini / llama")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N papers")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Max parallel LLM calls per paper (default: {MAX_WORKERS})")
    args = parser.parse_args()

    if args.no_judge:
        USE_LLM_JUDGE = False
    if args.limit:
        LIMIT = args.limit
    if args.workers:
        MAX_WORKERS = args.workers

    model_name       = args.model
    model_result_dir = MODEL_RESULT_DIRS[model_name]
    extraction_dir   = MODEL_EXTRACTION_DIRS[model_name]
    out_dir          = os.path.join(BASE_DIR, "deepeval_SDRF_results_Final_V_repo",
                                    f"evaluation_Gemma_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"  MODEL          : {model_name.upper()}")
    print(f"  Results dir    : {model_result_dir}")
    print(f"  Extraction dir : {extraction_dir}")
    print(f"  Manuscripts    : {TEST_SET}/<PXD>/manuscript.txt")
    print(f"  Golden source  : {TEST_SET}/<PXD>/*sdrf.tsv")
    print(f"  Output folder  : {out_dir}")
    print(f"  Judge model    : {EVALUATION_MODEL}  (via OpenRouter)")
    print(f"  Temperature    : {MODEL_TEMPERATURE}")
    print(f"  Thinking       : {'ENABLED' if MODEL_ENABLE_THINKING else 'DISABLED'}")
    print(f"  Disk cache     : {'ENABLED' if CACHE_ENABLED else 'DISABLED'}  dir={CACHE_DIR}")
    print(f"  LLM judge      : {'ENABLED  verdicts: high/medium/low' if USE_LLM_JUDGE else 'DISABLED'}")
    print(f"  Limit          : {LIMIT if LIMIT else 'ALL papers'}")
    print(f"  Concurrency    : {MAX_WORKERS} workers per paper")
    print(f"  Retries        : {MAX_RETRIES} (with {RETRY_DELAY}s progressive backoff)")
    print()
    print("Caching architecture:")
    print(f"  Layer 1 (disk):      key = SHA-256(model|paper_id|task|field_name|value)")
    print("                       Static system prompt excluded → cross-paper hits possible.")
    print(f"  Layer 2 (provider):  cache_control on system prompt + paper text")
    print("                       KV prefix computed ONCE per paper, reused across annotations.")
    print()
    print("Medium verdict correction:")
    print("  CORRECTED_VALUE field parsed from judge response for medium verdicts.")
    print("  Corrected values stored in output CSV for downstream use.")
    print()

    result = process_model(model_name, model_result_dir, extraction_dir, out_dir)
    df, coverage_df, per_paper_df = result

    if df.empty:
        print("ERROR: No data processed.")
        return

    if not coverage_df.empty:
        print(f"\n  Coverage report : {os.path.join(out_dir, 'llm_judge_coverage.csv')}")
        for agent in AGENTS:
            ag     = coverage_df[coverage_df["agent"] == agent]
            n_ext  = ag["was_extracted"].sum()
            n_gold = ag["has_golden"].sum()
            n_both = (ag["was_extracted"] & ag["has_golden"]).sum()
            total  = len(ag)
            print(f"    {agent:<28}: extracted={n_ext}/{total}  "
                  f"has_golden={n_gold}/{total}  overlap={n_both}/{total}")

    print(f"\n  Annotation review : {os.path.join(out_dir, 'llm_judge_annotation_review.csv')}")
    print(f"  Total rows        : {len(df)}")

    medium_df = df[(df["verdict"] == "medium") & (df["corrected_value"].notna())]
    if not medium_df.empty:
        n_med = len(medium_df)
        print(f"\n  Medium verdicts with corrections: {n_med}")
        for _, row in medium_df.head(10).iterrows():
            print(f"    {row['annotation_type']}: '{str(row['extracted_value'])[:35]}' "
                  f"→ '{str(row['corrected_value'])[:35]}'")
        if n_med > 10:
            print(f"    ... and {n_med - 10} more")

    if not per_paper_df.empty:
        print(f"  Per-paper stats   : {os.path.join(out_dir, 'llm_judge_per_paper.csv')}")

    cache_stats = _get_judge_model().get_cache_stats()
    print("CACHE STATISTICS")
    for k, v in cache_stats.items():
        print(f"  {k:<35} {v}")

    print("Generating summary plots...")
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
        print(f"\nLLM Judge mean accuracy (high verdicts): "
              f"{float(per_paper_df['judge_accuracy'].mean()):.1%}")

    print(f"FINAL SUMMARY  ({len(per_paper_df)} papers)")
    if not per_paper_df.empty and "judge_accuracy" in per_paper_df.columns:
        for _, stat in per_paper_df.iterrows():
            print(f"  {str(stat['paper_id']):<30} "
                  f"predicted={stat.get('total_predicted', 0)}  "
                  f"missed={stat.get('total_missed', 0)}  "
                  f"correct={stat.get('judge_n_correct', 0)}  "
                  f"halluc={stat.get('judge_n_hallucinated', 0)}  "
                  f"accuracy={stat.get('judge_accuracy', 0):.2f}")

    print(f"\nOutput files (available since each paper completed):")
    print(f"  Review CSV    : {os.path.join(out_dir, 'llm_judge_annotation_review.csv')}")
    print(f"  Per-paper CSV : {os.path.join(out_dir, 'llm_judge_per_paper.csv')}")
    print(f"  Coverage CSV  : {os.path.join(out_dir, 'llm_judge_coverage.csv')}")
    print(f"  Plots         : {out_dir}/llm_judge_*.png")
    print(f"\nAll outputs saved to: {out_dir}")
    print("\nPIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
