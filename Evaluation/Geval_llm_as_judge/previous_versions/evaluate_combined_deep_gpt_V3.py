import os
import re
import json
import hashlib
import time
import argparse
from collections import defaultdict
from pathlib import Path

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

EVALUATION_MODEL = "gpt-5.2"
REASONING_EFFORT = "medium"   
USE_LLM_JUDGE    = True
LIMIT            = None

_CACHE_MODEL_SLUG = EVALUATION_MODEL.replace("/", "_").replace(" ", "_")
CACHE_DIR         = os.path.join(BASE_DIR, f".prompt_cache_{_CACHE_MODEL_SLUG}")
CACHE_ENABLED     = True

AGENT_FILE_SUFFIX = {
    "BiologicalAgent":         "biological",
    "TechnicalAgent":          "technical",
    "ExperimentalDesignAgent": "experimental",
}

SKIP_MISSED_FIELDS = {
    "pxd_id", "dataset_id", "accession", "pride_id", "proteomexchange_id",
    "repository", "doi", "pubmed_id", "pmid",
}

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
    "material_type":        {"tissue", "cell line", "primary cells", "biofluid",
                             "whole organism", "plasma", "serum", "organoid"},
    "acquisition_method":   {"dda", "dia", "data-dependent acquisition", "data-dependent",
                             "data dependent acquisition"},
    "experimental_design":  {"treated vs control", "case vs control", "time course",
                             "dose response", "cross-sectional", "longitudinal"},
}


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
reduction_concentration : Concentration of the reduction reagent (e.g. "5 mM DTT"). Must be a numeric quantity with a unit paired with the reagent name.
collision_energy     : Collision energy used in MS/MS (e.g. "normalized collision energy 25", "27 eV").
fractionation        : Offline peptide/protein fractionation method applied before LC-MS (e.g. "high-pH reverse-phase fractionation", "strong anion exchange"). NOT biological/cellular fractionation. Equivalent to 'fractionation_method'.
enrichment_method    : Enrichment protocol applied before MS (e.g. "TiO2 phosphopeptide enrichment", "immunoprecipitation").
acquisition_method   : MS acquisition scheme (e.g. "DDA", "DIA", "data-dependent", "data-independent"). Must be a scheme name, NOT an instrument name. Equivalent to 'acquisition_method'.
alkylation_reagent   : Chemical used for cysteine alkylation (e.g. "iodoacetamide", "IAA", "NEM").
alkylation_concentration : Concentration of the alkylation reagent (e.g. "10 mM IAA"). Must be a numeric quantity with a unit.
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
                  -- If absent AND not a safe default -> HALLUCINATED: yes
                     (rule A -> VERDICT: low).
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

=================================================================
VERDICT RULES
=================================================================

LOW (most severe) -- use when ANY of:
  (A) HALLUCINATED: absent from paper text and not a safe default.
  (B) TYPE MISMATCH: the value's fundamental nature belongs to a
      completely different metadata field.
      Examples of TRUE type mismatches:
        A tissue name ("liver") labeled as instrument;
        An instrument name ("Q Exactive") labeled as acquisition_method;
        A numeric count labeled as a field expecting a name;
        An alkylation reagent name labeled as reduction_reagent.
  (D) CONTRADICTION / INCORRECT: present in source, type is correct,
      but the value is factually wrong or contradicts the paper.
      Examples:
        "Mus musculus" when paper uses human samples;
        "DDA" when paper explicitly uses DIA;
        A cleavage agent not mentioned in the paper.

MEDIUM -- type correct, value present, no LOW rule applies, AND any of:
  (a) SUBTYPE / SUPERTYPE DRIFT.
  (b) SINGLE CHANNEL of a multi-channel scheme.
  (c) Partial extraction from a multi-value field.

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

Include when the candidate:
  - Matches exactly or by domain equivalence.
  - Is a subtype, supertype, variant, or configuration of a reference.
  - Is a non-canonical phrasing of a reference.
  - Contains a reference as one of its items.
  - Is the expanded form or abbreviation of a reference.

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
  HALLUCINATED      = yes ONLY when absent from paper text AND not a safe default.
  VERDICT           = overall high | medium | low.

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

COVERAGE: a candidate covers gt_value when it is semantically equivalent,
a more-specific subtype/variant, a broader term that clearly subsumes it,
a verbose/abbreviated phrasing of the same concept, or a list that contains
it as one of its items.

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

Example 2 -- HIGH: safe default, not explicitly stated
field_name=disease, candidate_value="normal",
reference_values=[]

TYPE CHECK:        Disease state field -- correct.
SOURCE CHECK:      "normal" is a safe default when subjects are healthy controls;
                   no isobaric disease label mentioned -- not hallucinated.
TRUTH CHECK:       Paper describes healthy volunteers -- "normal" is factually valid.
COMPLETENESS CHECK:Complete.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

Example 3 -- HIGH: correct not in reference_values
field_name=acquisition_method, candidate_value="data-dependent acquisition",
reference_values=["DDA"]

TYPE CHECK:        MS acquisition scheme -- correct field.
SOURCE CHECK:      Present in methods.
TRUTH CHECK:       "data-dependent acquisition" is the full form of "DDA" -- correct.
COMPLETENESS CHECK:Complete.
MATCHED_REFERENCE: DDA
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

Example 4 -- LOW (rule B): instrument name labeled as acquisition_method
field_name=acquisition_method, candidate_value="Q Exactive Plus",
reference_values=["DDA"]

TYPE CHECK:        "Q Exactive Plus" is a mass spectrometer model -- its
                   fundamental nature belongs to the 'instrument' field, not
                   acquisition_method.
SOURCE CHECK:      Present in paper.
TRUTH CHECK:       Not applicable -- wrong fundamental type.
COMPLETENESS CHECK:Not applicable -- wrong fundamental type.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      no
CORRECT_TYPE_NAME: instrument
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           low

Example 5 -- LOW (rule A): hallucinated
field_name=enrichment_method, candidate_value="IMAC phosphopeptide enrichment",
reference_values=[]

TYPE CHECK:        Enrichment protocol -- correct field.
SOURCE CHECK:      No mention of IMAC or phosphopeptide enrichment in the paper,
                   and this is not a safe default -- HALLUCINATED.
TRUTH CHECK:       Cannot be valid -- not in paper.
COMPLETENESS CHECK:Not applicable.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      yes
VERDICT:           low

Example 6 -- MEDIUM: single channel of triple SILAC
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

Example 7 -- LOW (rule D): wrong species
field_name=species, candidate_value="Mus musculus",
reference_values=["Homo sapiens"]

TYPE CHECK:        Species name -- correct field.
SOURCE CHECK:      "Mus musculus" does not appear in a paper studying human samples.
TRUTH CHECK:       Paper uses human subjects; "Mus musculus" contradicts this.
COMPLETENESS CHECK:Not applicable.
MATCHED_REFERENCE: NONE
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      yes
VERDICT:           low

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
    "(3) TRUTH: is it factually correct per the paper and field definition? "
    "    Synonyms, abbreviations, digit/word swaps = CORRECT. "
    "    Differing from reference_values is NEVER a reason to fail. "
    "(4) COMPLETENESS: single-channel stand-in or partial multi-value -> medium. "
    "    Subtype or supertype drift -> medium. "
    "HIGH = correct type + present/inferable + factually valid + complete. "
    "MEDIUM = correct type + present + valid, but partial/drift. "
    "LOW = hallucinated, type mismatch, or factually wrong."
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
    "If ABSENT and not a safe default: HALLUCINATED: yes -> VERDICT: low (rule A). "
    "If PRESENT but in wrong context: HALLUCINATED: no, VALUE_CORRECT: no (rule D).",

    "TRUTH CHECK: assess whether the candidate is factually correct AND fits the field "
    "definition for this experiment. Synonyms, abbreviations, digit/word swaps, "
    "common/scientific name swaps are CORRECT. Differing from reference_values alone "
    "is NEVER a reason to fail -- a correct value not in reference_values is HIGH.",

    "COMPLETENESS CHECK: flag MEDIUM only for (a) subtype drift, (b) supertype drift, "
    "(c) single-channel stand-in for multi-channel scheme, or (d) partial extraction "
    "from a known multi-value field. Do NOT flag medium for verbose phrasing, "
    "abbreviation expansion, or quality qualifiers.",

    "MATCHED_REFERENCE: identify the closest reference from reference_values for "
    "tracking only -- does NOT affect verdict. Output verbatim or NONE.",

    "Output TYPE CHECK / SOURCE CHECK / TRUTH CHECK / COMPLETENESS CHECK, "
    "MATCHED_REFERENCE, TYPE_CORRECT (yes/no), CORRECT_TYPE_NAME (field_name or NONE), "
    "VALUE_CORRECT (yes/no), VALUE_COMPLETE (yes/no), "
    "HALLUCINATED (yes/no), and VERDICT: high | medium | low.",
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
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(paper_id: str, task: str, field_name: str, primary_value: str) -> str:
        blob = f"{EVALUATION_MODEL}|{paper_id}|{task}|{field_name.lower()}|{primary_value}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, paper_id: str, task: str, field_name: str,
            primary_value: str) -> str | None:
        if not self._enabled:
            return None
        key  = self._make_key(paper_id, task, field_name, primary_value)
        path = self._dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._hits += 1
                return data.get("response")
            except (json.JSONDecodeError, OSError):
                return None
        self._misses += 1
        return None

    def put(self, paper_id: str, task: str, field_name: str,
            primary_value: str, response: str) -> None:
        if not self._enabled:
            return
        key  = self._make_key(paper_id, task, field_name, primary_value)
        path = self._dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps({
                    "key": key, "paper_id": paper_id, "task": task,
                    "field_name": field_name, "primary_value": primary_value,
                    "response": response, "timestamp": time.time(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"  [cache-write-error] {exc}")

    def stats(self) -> dict:
        n_files = sum(1 for _ in self._dir.glob("*.json")) if self._enabled else 0
        total   = self._hits + self._misses
        return {
            "enabled":                   self._enabled,
            "cache_dir":                 str(self._dir),
            "cached_responses_on_disk":  n_files,
            "session_hits":              self._hits,
            "session_misses":            self._misses,
            "session_hit_rate":          f"{self._hits/total*100:.1f}%" if total > 0 else "n/a",
        }



def _build_messages(paper_text: str, entity_context: dict) -> list[dict]:
    return [
        {"role": "system",    "content": _STATIC_SYSTEM_PROMPT},
        {"role": "user",      "content": (
            "PAPER TEXT (title + abstract + methods) — use as source of truth "
            "for all SOURCE CHECK and TRUTH CHECK assessments:\n\n" + paper_text
        )},
        {"role": "assistant", "content": (
            "Understood. I have read the paper text and will assess all metadata "
            "field values against it using the definitions and rules in my instructions."
        )},
        {"role": "user",      "content": json.dumps(entity_context, indent=2,
                                                     ensure_ascii=False)},
    ]



def _build_openai_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return openai.OpenAI(api_key=api_key)


class GPT52Judge(DeepEvalBaseLLM):
    def __init__(self):
        self._model_name             = EVALUATION_MODEL
        self._client                 = None
        self._paper_text:  str       = ""
        self._current_context: dict | None = None
        self._current_paper_id: str  = ""
        self._response_cache         = DiskResponseCache(CACHE_DIR, enabled=CACHE_ENABLED)
        self._call_count             = 0
        self._api_call_count         = 0
        self._disk_hit_count         = 0
        self._openai_prefix_hit_count = 0

    def set_paper_context(self, source_text: str, paper_id: str = "") -> None:
        self._paper_text       = source_text
        self._current_paper_id = paper_id

    def set_entity_context(self, context: dict) -> None:
        self._current_context = context

    def load_model(self) -> openai.OpenAI:
        if self._client is None:
            self._client = _build_openai_client()
        return self._client

    def generate(self, prompt: str, schema=None):
        self._call_count += 1
        context    = self._current_context or {}
        task       = context.get("task", "verify")
        field_name = context.get("field_name", "")

        if task == "verify":
            primary_value = context.get("candidate_value", prompt)
        elif task == "check_coverage":
            primary_value = context.get("gt_value", prompt)
        else:
            primary_value = field_name 

        cached = self._response_cache.get(
            self._current_paper_id, task, field_name, primary_value
        )
        if cached is not None:
            self._disk_hit_count += 1
            print(
                f"  [call #{self._call_count}] DISK CACHE HIT  "
                f"paper={self._current_paper_id!r}  task={task}  "
                f"field={field_name}  value={primary_value[:40]!r}"
            )
            content = cached
        else:
            self._api_call_count += 1
            messages = _build_messages(self._paper_text, context)
            print(
                f"  [call #{self._call_count}] API CALL  "
                f"paper={self._current_paper_id!r}  task={task}  "
                f"field={field_name}  value={primary_value[:40]!r}  "
                f"[sys={len(_STATIC_SYSTEM_PROMPT):,}  "
                f"paper={len(self._paper_text):,}  "
                f"entity={len(json.dumps(context)):,} chars]"
            )
            response = self.load_model().chat.completions.create(
                model=self._model_name,
                messages=messages,
                reasoning_effort=REASONING_EFFORT,
                max_completion_tokens=16000,
                store=True,
            )
            content = response.choices[0].message.content or ""

            if hasattr(response, "usage") and response.usage:
                usage         = response.usage
                cached_tokens = 0
                if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                    cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)
                if cached_tokens > 0:
                    self._openai_prefix_hit_count += 1
                print(
                    f"  [tokens] prompt={usage.prompt_tokens:,}  "
                    f"openai_prefix_cached={cached_tokens:,} "
                    f"({'HIT' if cached_tokens > 0 else 'miss'})  "
                    f"completion={usage.completion_tokens:,}"
                )
            self._response_cache.put(
                self._current_paper_id, task, field_name, primary_value, content
            )

        print(f"  [response] {content[:200]}...")

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
        return f"{self._model_name}-reasoning-{REASONING_EFFORT}"

    def get_cache_stats(self) -> dict:
        stats = self._response_cache.stats()
        stats["total_generate_calls"]     = self._call_count
        stats["actual_api_calls"]         = self._api_call_count
        stats["disk_cache_hits"]          = self._disk_hit_count
        stats["openai_prefix_cache_hits"] = self._openai_prefix_hit_count
        if self._call_count > 0:
            stats["overall_disk_hit_rate"] = (
                f"{self._disk_hit_count / self._call_count * 100:.1f}%"
            )
        return stats


_gpt52_model:       GPT52Judge | None = None
_annotation_metric: GEval | None = None
_missed_metric:     GEval | None = None
_coverage_metric:   GEval | None = None


def _get_judge_model() -> GPT52Judge:
    global _gpt52_model
    if _gpt52_model is None:
        _gpt52_model = GPT52Judge()
    return _gpt52_model


def _get_annotation_metric() -> GEval:
    global _annotation_metric
    if _annotation_metric is None:
        _annotation_metric = GEval(
            name="AnnotationJudge",
            criteria=_ANNOTATION_GEVAL_CRITERIA,
            evaluation_steps=_ANNOTATION_GEVAL_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=_get_judge_model(),
            threshold=0.5,
        )
    return _annotation_metric


def _get_missed_metric() -> GEval:
    global _missed_metric
    if _missed_metric is None:
        _missed_metric = GEval(
            name="MissedFieldDetector",
            criteria=_MISSED_GEVAL_CRITERIA,
            evaluation_steps=_MISSED_GEVAL_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=_get_judge_model(),
            threshold=0.5,
        )
    return _missed_metric


def _get_coverage_metric() -> GEval:
    global _coverage_metric
    if _coverage_metric is None:
        _coverage_metric = GEval(
            name="GTCoverageChecker",
            criteria=_COVERAGE_GEVAL_CRITERIA,
            evaluation_steps=_COVERAGE_GEVAL_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=_get_judge_model(),
            threshold=0.5,
        )
    return _coverage_metric


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

    def _accept(name: str) -> str | None:
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

    prose_patterns = [
        r"correct\s+(?:field|type)\s+is\s+([a-z_]+)",
        r"belongs\s+to\s+(?:the\s+)?([a-z_]+)\s+field",
        r"should\s+be\s+(?:labeled\s+as\s+|classified\s+as\s+)?([a-z_]+)",
    ]
    for pat in prose_patterns:
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


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s.strip().lower())


def _find_matching_reference(predicted: str,
                              gt_candidates: list[str]) -> str | None:
    p = _norm(predicted)
    if not p:
        return None
    for g in gt_candidates:
        if _norm(g) == p:
            return g
    return None


def _resolve_matched_reference(raw_ref: str | None,
                                gt_candidates: list[str]) -> str | None:
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


def _exact_match_result(matched_ref: str) -> dict:
    return {
        "verdict":           "high",
        "type_mismatch":     False,
        "correct_type":      None,
        "value_correct":     True,
        "value_complete":    True,
        "hallucination":     False,
        "matched_reference": matched_ref,
        "issue_summary":     "Exact match with a ground-truth value -- LLM skipped.",
        "match_type":        "CORRECT",
        "any_match":         True,
        "geval_score":       None,
    }


def _safe_default_result(field_name: str, extracted_value: str) -> dict:
    return {
        "verdict":           "high",
        "type_mismatch":     False,
        "correct_type":      None,
        "value_correct":     True,
        "value_complete":    True,
        "hallucination":     False,
        "matched_reference": None,
        "issue_summary":     f"Safe default value '{extracted_value}' -- LLM skipped.",
        "match_type":        "CORRECT",
        "any_match":         True,
        "geval_score":       None,
    }


def _run_geval_semantic(field_name: str, extracted_value: str,
                        gt_candidates: list[str]) -> dict:
    model = _get_judge_model()
    model.set_entity_context({
        "task":             "verify",
        "field_name":       field_name,
        "field_definition": _field_def_for(field_name),
        "candidate_value":  extracted_value,
        "reference_values": gt_candidates,
    })
    expected_str = (
        "Assess the candidate_value against the paper text AND the field definition: "
        "correct metadata field in FUNDAMENTAL NATURE (TYPE_CORRECT yes/no), "
        "present/inferable in source (HALLUCINATED yes/no), "
        "factually valid (VALUE_CORRECT yes/no), complete (VALUE_COMPLETE yes/no). "
        "Synonyms, abbreviations, safe defaults = CORRECT and not hallucinated. "
        "Differing from reference_values alone is NEVER a reason to fail. "
        "Verdict: high (all pass) | medium (drift/partial) | low (absent/mismatch/wrong). "
        + (
            f"Reference values for MATCHED_REFERENCE tracking only: "
            f"{'; '.join(gt_candidates)}"
            if gt_candidates else "No reference values provided."
        )
    )
    test_case = LLMTestCase(
        input=f"Assess field='{field_name}' candidate_value='{extracted_value}'",
        actual_output=f"field_name: {field_name}\ncandidate_value: {extracted_value}",
        expected_output=expected_str,
    )
    metric = _get_annotation_metric()
    try:
        metric.measure(test_case)
        reason = metric.reason or ""
        score  = metric.score
    except Exception as exc:
        print(f"  [GEval error] {field_name}: {type(exc).__name__}: {exc}")
        return {
            "verdict": None, "type_mismatch": None, "correct_type": None,
            "value_correct": None, "value_complete": None, "hallucination": None,
            "matched_reference": None, "issue_summary": f"GEval error: {exc}",
            "match_type": "ERROR", "any_match": False, "geval_score": None,
        }

    verdict_match   = re.search(r"VERDICT:\s*(high|medium|low)", reason, re.IGNORECASE)
    verdict         = verdict_match.group(1).lower() if verdict_match else None
    hall_parsed     = _parse_yes_no(reason, "HALLUCINATED")
    is_hallucinated = bool(hall_parsed) if hall_parsed is not None else False

    type_correct_parsed = _parse_type_correct(reason)
    if type_correct_parsed is not None:
        is_mismatch = not type_correct_parsed
    else:
        is_mismatch = (verdict == "low") and not is_hallucinated and bool(
            re.search(r"type mismatch|wrong (?:field|type)|incorrect type|belongs to",
                      reason, re.IGNORECASE)
        )

    correct_type = _parse_correct_type_name(reason, field_name) if is_mismatch else None

    matched_ref_raw   = _parse_matched_reference(reason)
    matched_reference = _resolve_matched_reference(matched_ref_raw, gt_candidates)

    vc_parsed    = _parse_yes_no(reason, "VALUE_CORRECT")
    vcomp_parsed = _parse_yes_no(reason, "VALUE_COMPLETE")
    value_correct  = vc_parsed    if vc_parsed    is not None else (
        (verdict in ("high", "medium")) and not is_hallucinated and not is_mismatch
    )
    value_complete = vcomp_parsed if vcomp_parsed is not None else (
        (verdict == "high") and not is_hallucinated and not is_mismatch
    )

    if is_hallucinated or is_mismatch:
        value_correct = False
    if is_hallucinated:
        value_complete = False

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

    return {
        "verdict":           verdict,
        "type_mismatch":     is_mismatch,
        "correct_type":      correct_type,
        "value_correct":     value_correct,
        "value_complete":    value_complete,
        "hallucination":     is_hallucinated,
        "matched_reference": matched_reference,
        "issue_summary":     reason,
        "match_type":        match_type,
        "any_match":         match_type in ("CORRECT", "PARTIAL"),
        "geval_score":       score,
    }


def check_missed_annotation(field_name: str) -> str:
    model = _get_judge_model()
    model.set_entity_context({
        "task":             "check_missing",
        "field_name":       field_name,
        "field_definition": _field_def_for(field_name),
    })
    test_case = LLMTestCase(
        input=f"Is '{field_name}' present in the source text?",
        actual_output=f"Field '{field_name}' was NOT extracted by the pipeline.",
        expected_output="Determine if the source text contains this field.",
    )
    metric = _get_missed_metric()
    try:
        metric.measure(test_case)
        reason = metric.reason or ""
        m      = re.search(r"VERDICT:\s*(high|medium|low)", reason, re.IGNORECASE)
        return m.group(1).lower() if m else "low"
    except Exception as exc:
        print(f"  [missed GEval error] {field_name}: {exc}")
        return "low"



def _substr_covers(gt_v: str, predictions: list[str]) -> bool:
    r = _norm(gt_v)
    if not r or len(r) < 4:
        return False
    return any(r in _norm(pred) for pred in predictions)


def _geval_covers(field_name: str, gt_v: str,
                  predictions: list[str]) -> bool:
    model = _get_judge_model()
    model.set_entity_context({
        "task":             "check_coverage",
        "field_name":       field_name,
        "field_definition": _field_def_for(field_name),
        "gt_value":         gt_v,
        "candidate_values": predictions[:20],
    })
    test_case = LLMTestCase(
        input=f"Is '{gt_v}' covered by any candidate for {field_name}?",
        actual_output=f"Candidate values: {'; '.join(predictions[:20])}",
        expected_output=f"Determine if any candidate covers the reference: {gt_v}",
    )
    metric = _get_coverage_metric()
    try:
        metric.measure(test_case)
        reason  = metric.reason or ""
        m       = re.search(r"COVERED:\s*(yes|no)", reason, re.IGNORECASE)
        covered = bool(m and m.group(1).lower() == "yes")
        print(f"  [coverage-geval] {field_name}: '{gt_v[:50]}' -> COVERED={covered}")
        return covered
    except Exception as exc:
        print(f"  [coverage-geval error] {field_name}: '{gt_v[:40]}': {exc}")
        return False


def _second_pass_coverage(field_name: str, gt_v: str,
                          predictions: list[str]) -> bool:
    if _substr_covers(gt_v, predictions):
        print(f"  [coverage-substr] {field_name}: '{gt_v[:50]}' covered by substring")
        return True
    return _geval_covers(field_name, gt_v, predictions)


def evaluate_paper_with_geval(
    paper_id: str,
    tier_rows: list[dict],
    all_golden: dict[str, str],
    source_text: str,
) -> list[dict]:
    _get_judge_model().set_paper_context(source_text, paper_id=paper_id)

    gt_lists: dict[str, list[str]] = defaultdict(list)
    for k, v in all_golden.items():
        if v:
            gt_lists[k].append(str(v).strip())

    covered_gt_refs: dict[str, set] = defaultdict(set)
    n_total = len(tier_rows)

    for idx, tr in enumerate(tier_rows):
        field_name      = tr["annotation_type"]
        extracted_value = tr["extracted_value"]
        golden_value    = tr["golden_value"] or ""
        gt_candidates   = gt_lists.get(field_name, (
            [golden_value] if golden_value else []
        ))

        if not extracted_value or _is_not_extracted(str(extracted_value)):
            tr["verdict"]        = None
            tr["value_correct"]  = None
            tr["value_complete"] = None
            tr["hallucination"]  = None
            tr["type_mismatch"]  = None
            tr["issue_summary"]  = "Skipped (sentinel / empty — not extracted)"
            continue

        if tr.get("tier1_exact"):
            result = _exact_match_result(golden_value or extracted_value)
            if golden_value:
                covered_gt_refs[field_name].add(golden_value)
        elif extracted_value.strip().lower() in SAFE_DEFAULTS.get(field_name, set()):
            result = _safe_default_result(field_name, extracted_value)
        elif (matched := _find_matching_reference(extracted_value, gt_candidates)) is not None:
            result = _exact_match_result(matched)
            covered_gt_refs[field_name].add(matched)
        else:
            print(
                f"      [{idx+1}/{n_total}] {field_name}: '{extracted_value[:40]}' "
                f"vs golden='{golden_value[:40]}'", end="  "
            )
            result = _run_geval_semantic(field_name, extracted_value, gt_candidates)
            mref = result.get("matched_reference")
            if mref:
                covered_gt_refs[field_name].add(mref)
            if result.get("verdict") is not None:
                print(
                    f"-> verdict={result['verdict']}  "
                    f"correct={result['value_correct']}  "
                    f"halluc={result['hallucination']}  "
                    f"mismatch={result['type_mismatch']}  "
                    f"score={result.get('geval_score')}"
                )
            else:
                print("-> ERROR")

        tr["verdict"]        = result["verdict"]
        tr["value_correct"]  = result["value_correct"]
        tr["value_complete"] = result["value_complete"]
        tr["hallucination"]  = result["hallucination"]
        tr["type_mismatch"]  = result["type_mismatch"]
        tr["correct_type"]   = result.get("correct_type")
        tr["issue_summary"]  = result["issue_summary"]

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


def load_llm_output_json(extraction_dir: str, pxd_id: str, agent: str) -> dict:
    suffix    = AGENT_FILE_SUFFIX.get(agent, agent.lower())
    agent_dir = os.path.join(extraction_dir, agent)
    if not os.path.exists(agent_dir):
        print(f"    WARNING: agent dir not found: {agent_dir}")
        return {}

    raw_data  = None
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
        if isinstance(val, list):
            meaningful = [str(v).strip() for v in val if not _is_not_extracted(str(v))]
            if not meaningful:
                print(f"    [{agent}] {field}: all list values are sentinels — skipping")
                continue
            extracted = meaningful[0]
        elif isinstance(val, str):
            extracted = val.strip()
        else:
            extracted = str(val).strip()
        if _is_not_extracted(extracted):
            print(f"    [{agent}] {field}: value {extracted!r} is a sentinel — skipping")
            continue
        parsed[FIELD_NAME_MAP.get(field, field)] = extracted

    return parsed


def load_golden_json(pxd_id: str, agent: str) -> dict:
    fname = f"{pxd_id}_{agent}_golden.json"
    path  = os.path.join(GOLDEN_DIR, fname)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        fields = raw.get("fields", raw) if isinstance(raw, dict) else raw
        return {
            k: str(v).strip()
            for k, v in fields.items()
            if v is not None and str(v).strip().lower() not in ("nan", "none", "null", "")
        }
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
                  extraction_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    detailed_df = load_detailed_csv(model_result_dir)
    if detailed_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Loaded {len(detailed_df)} rows from sdrf_benchmark_detailed.csv")

    pxd_ids = sorted(detailed_df["pxd_id"].unique())
    print(f"  Papers: {len(pxd_ids)}")
    if LIMIT and LIMIT < len(pxd_ids):
        pxd_ids = pxd_ids[:LIMIT]
        print(f"  Limit applied: running only first {LIMIT} papers: {pxd_ids}")

    all_rows          = []
    all_coverage_rows = []

    for pxd_id in pxd_ids:
        print(f"\n  --- {pxd_id} ---")
        source_text = load_manuscript(pxd_id)

        all_predicted: dict[str, dict] = {}
        all_golden_by_agent: dict[str, dict] = {}
        for agent in AGENTS:
            pred = load_llm_output_json(extraction_dir, pxd_id, agent)
            gold = load_golden_json(pxd_id, agent)
            if pred:
                all_predicted[agent] = pred
            if gold:
                all_golden_by_agent[agent] = gold

        pxd_df = detailed_df[detailed_df["pxd_id"] == pxd_id].copy()

        flat_golden: dict[str, str] = {}
        for agent in AGENTS:
            flat_golden.update(load_golden_json(pxd_id, agent))

        coverage_rows = []
        for agent in AGENTS:
            pred = all_predicted.get(agent, {})
            for raw_field in PIPELINE_FIELDS.get(agent, []):
                mapped     = FIELD_NAME_MAP.get(raw_field, raw_field)
                extracted  = pred.get(mapped) or pred.get(raw_field)
                golden_val = flat_golden.get(mapped) or flat_golden.get(raw_field)
                was_extracted = bool(extracted and not _is_not_extracted(str(extracted)))
                coverage_rows.append({
                    "paper_id":        pxd_id,
                    "agent":           agent,
                    "pipeline_field":  raw_field,
                    "mapped_field":    mapped,
                    "extracted_value": extracted or "",
                    "was_extracted":   was_extracted,
                    "has_golden":      bool(golden_val),
                    "golden_value":    golden_val or "",
                })
        all_coverage_rows.extend(coverage_rows)

        n_extracted  = sum(1 for r in coverage_rows if r["was_extracted"])
        n_has_golden = sum(1 for r in coverage_rows if r["has_golden"])
        n_both       = sum(1 for r in coverage_rows if r["was_extracted"] and r["has_golden"])
        total_fields = len(coverage_rows)
        print(f"    Coverage: {n_extracted}/{total_fields} extracted  "
              f"| {n_has_golden}/{total_fields} have golden  "
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
                "verdict":            None,
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
                if field in sdrf_fields_this_paper or _is_not_extracted(str(val)):
                    continue
                golden_val = flat_golden.get(field, "")
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
                    "verdict":            None,
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
            print(f"    [judge] Running GEval on {len(tier_rows)} annotations...")
            tier_rows, covered_gt_refs = evaluate_paper_with_geval(
                pxd_id, tier_rows, flat_golden, source_text
            )

            extracted_fields = {
                tr["annotation_type"].lower()
                for tr in tier_rows
                if tr["extracted_value"] and not _is_not_extracted(str(tr["extracted_value"]))
            }

            missed_rows = []

            for field, golden_val in flat_golden.items():
                if field.lower() in extracted_fields:
                    continue
                if field.lower() in SKIP_MISSED_FIELDS:
                    continue
                preds_for_field = [
                    tr["extracted_value"]
                    for tr in tier_rows
                    if tr["annotation_type"].lower() == field.lower()
                    and tr["extracted_value"]
                    and not _is_not_extracted(str(tr["extracted_value"]))
                ]
                if preds_for_field:
                    if _second_pass_coverage(field, golden_val, preds_for_field):
                        continue
                if field in covered_gt_refs and golden_val in covered_gt_refs[field]:
                    continue
                print(f"        [missed] {field}: golden exists but not extracted - MISSING")
                sdrf_agent = FIELD_TO_AGENT.get(field, "BiologicalAgent")
                missed_rows.append(_make_missed_row(
                    pxd_id, sdrf_agent, field, golden_val,
                    f"Not extracted — golden value exists"
                ))


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
        "verdict", "has_golden",
        "type_mismatch", "correct_type", "value_correct", "value_complete",
        "hallucination", "source_evidence", "issue_summary", "corrected_value",
    ]
    for c in final_cols:
        if c not in df.columns:
            df[c] = None
    return df[final_cols], coverage_df


def _make_missed_row(paper_id: str, agent: str, field: str,
                     golden_val: str, summary: str) -> dict:
    return {
        "paper_id":           paper_id,
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
        "verdict":            "missing",
        "has_golden":         bool(golden_val),
        "type_mismatch":      False,
        "correct_type":       None,
        "value_correct":      False,
        "value_complete":     False,
        "hallucination":      False,
        "source_evidence":    "",
        "issue_summary":      summary,
        "corrected_value":    golden_val or None,
        "row_type":           "missed",
    }



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
    BG, GRID_C, TEXT_C, SUB_C, SPINE_C = (
        "white", "#E5E7EB", "#111827", "#6B7280", "#D1D5DB")
    CLR = dict(correct="#27ae60", hall="#e74c3c", mismatch="#9b59b6",
               wrong="#e67e22", incomplete="#3498db", missed="#95a5a6")

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
    ax.set_title(f"{model_name.upper()} — 5-Tier Match Distribution "
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
    b = np.zeros(len(names))
    for arr, col, lbl in [
        (n_correct,    CLR["correct"],    "Correct"),
        (n_hall,       CLR["hall"],       "Hallucinated"),
        (n_mismatch,   CLR["mismatch"],   "Type Mismatch"),
        (n_wrong,      CLR["wrong"],      "Wrong Value"),
        (n_incomplete, CLR["incomplete"], "Incomplete"),
        (n_missed,     CLR["missed"],     "Missed"),
    ]:
        ax.bar(x, arr, w, bottom=b, label=lbl, color=col,
               edgecolor="white", lw=0.5, alpha=0.92)
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
    plt.close()
    print(f"  Saved: llm_judge_annotation_quality_counts.png")

    acc_vals = per_paper_df["judge_accuracy"].tolist()
    overall  = float(np.mean(acc_vals))
    fig, ax  = plt.subplots(figsize=(max(14, len(names) * 0.55), 6), facecolor=BG)
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
    ax.set_title(f"{model_name.upper()} — LLM Judge Accuracy per Paper",
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
                  int(n_wrong.sum()), int(n_incomplete.sum()), int(n_missed.sum())]
    agg_colors = [CLR["correct"], CLR["hall"], CLR["mismatch"],
                  CLR["wrong"], CLR["incomplete"], CLR["missed"]]
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
    plt.close()
    print(f"  Saved: llm_judge_aggregate.png")


def main():
    parser = argparse.ArgumentParser(
        description="SDRF benchmark + Code-1-style LLM judge (high/medium/low verdicts)")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_RESULT_DIRS.keys()),
                        help="Model: claude / gpt / gemini / llama")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N papers")
    args = parser.parse_args()

    global USE_LLM_JUDGE, LIMIT
    if args.no_judge:
        USE_LLM_JUDGE = False
    if args.limit:
        LIMIT = args.limit

    model_name       = args.model
    model_result_dir = MODEL_RESULT_DIRS[model_name]
    extraction_dir   = MODEL_EXTRACTION_DIRS[model_name]
    out_dir          = os.path.join(BASE_DIR, f"evaluation3_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"  MODEL          : {model_name.upper()}")
    print(f"  Results dir    : {model_result_dir}")
    print(f"  Extraction dir : {extraction_dir}")
    print(f"  Manuscripts    : {TEST_SET}/<PXD>/manuscript.txt")
    print(f"  Golden dir     : {GOLDEN_DIR}")
    print(f"  Output folder  : {out_dir}")
    print(f"  Judge model    : {EVALUATION_MODEL}  reasoning_effort={REASONING_EFFORT}")
    print(f"  Disk cache     : {'ENABLED' if CACHE_ENABLED else 'DISABLED'}  dir={CACHE_DIR}")
    print(f"  LLM judge      : {'ENABLED  verdicts: high/medium/low' if USE_LLM_JUDGE else 'DISABLED'}")
    print(f"  Limit          : {LIMIT if LIMIT else 'ALL papers'}")
    print()
    print("Cache architecture:")
    print(f"  Layer 1 (disk):   key = SHA-256(model|paper_id|task|field_name|value)")
    print("                    Static system prompt excluded → cross-paper hits possible.")
    print("  Layer 2 (OpenAI): Static system prompt prefix cached automatically.")
    print("                    Paper text in conversation turn, not system prompt.")

    df, coverage_df = process_model(model_name, model_result_dir, extraction_dir)
    if df.empty:
        print("ERROR: No data processed.")
        return

    if not coverage_df.empty:
        cov_path = os.path.join(out_dir, "llm_judge_coverage.csv")
        coverage_df.to_csv(cov_path, index=False, escapechar="\\", doublequote=True)
        print(f"\n  Coverage report : {cov_path}")
        for agent in AGENTS:
            ag     = coverage_df[coverage_df["agent"] == agent]
            n_ext  = ag["was_extracted"].sum()
            n_gold = ag["has_golden"].sum()
            n_both = (ag["was_extracted"] & ag["has_golden"]).sum()
            total  = len(ag)
            print(f"    {agent:<28}: extracted={n_ext}/{total}  "
                  f"has_golden={n_gold}/{total}  overlap={n_both}/{total}")

    review_path = os.path.join(out_dir, "llm_judge_annotation_review.csv")
    df.to_csv(review_path, index=False, escapechar="\\", doublequote=True)
    print(f"\n  Annotation review : {review_path}")
    print(f"  Total rows        : {len(df)}")

    per_paper_df = compute_per_paper_stats(df)
    if not per_paper_df.empty:
        per_paper_path = os.path.join(out_dir, "llm_judge_per_paper.csv")
        per_paper_df.to_csv(per_paper_path, index=False, escapechar="\\", doublequote=True)
        print(f"  Per-paper stats   : {per_paper_path}")

    cache_stats = _get_judge_model().get_cache_stats()
    print(f"\n{'='*60}")
    print("CACHE STATISTICS")
    print(f"{'='*60}")
    for k, v in cache_stats.items():
        print(f"  {k:<35} {v}")

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
        print(f"\nLLM Judge mean accuracy (high verdicts): "
              f"{float(per_paper_df['judge_accuracy'].mean()):.1%}")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
