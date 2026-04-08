import os
import re
import unicodedata
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import openai

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

PAPERS_JSON      = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/papers_dataset.json"
RESULTS_DIR      = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/evaluation_results_gpt/LLM_as_Judge"
GROUND_TRUTH_DIR = "/Users/fateme/Downloads/Ians_Annotations"
PREDICTIONS_DIR  = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/filtered_output"
EVALUATION_MODEL = "gpt-5.2"

os.makedirs(RESULTS_DIR, exist_ok=True)

_ENTITY_DEFINITIONS = """
SAMPLE / BIOLOGICAL ENTITY TYPES:
Age                         : Age of the donor or developmental stage (e.g. "45 years", "E14.5 embryo").
AlkylationReagent           : Chemical that alkylates cysteine -SH groups (e.g. "IAA", "NEM").
AnatomicSiteTumor           : Anatomical location of tumor sample (e.g. "left lung lobe").
AncestryCategory            : Donor ancestry / ethnicity (e.g. "European", "East Asian").
Bait                        : The specific named protein or molecule used as the affinity handle in AP-MS / pull-down. Must be a concrete identifier such as a gene name, protein name, or tagged construct (e.g. "Spt16-TAP", "GFP-HDAC1", "Flag-Ago2"). Generic tag names alone ("Flag", "GFP", "HA") are NOT Bait.
BMI                         : Body-Mass Index of donor (kg/m^2).
BiologicalReplicate         : Identifier label for a specific biological replicate instance (e.g. "bioRep1"). NOT a count.
CellLine                    : Name of immortalized cell line (e.g. "HEK293T", "U2OS").
CellPart                    : Subcellular compartment / fraction (e.g. "nucleus", "mitochondria").
CellType                    : Primary cell type or lineage (e.g. "neurons", "fibroblasts").
CleavageAgent               : Protease or chemical used for protein digestion (e.g. "trypsin", "Lys-C").
Compound                    : Chemical or small molecule added to sample. The extracted value MAY include a concentration prefix or suffix (e.g. "20 nM Calyculin A", "PDP(m)-Nal 50 uM", "10 ug/mL EGF") — this is intentional pipeline behaviour where compound and concentration are annotated together, and does NOT make the annotation incorrect or mistyped. NOT the dose/concentration alone without a chemical name. Example standalone names: "IFNbeta", "doxycycline", "rapamycin".
ConcentrationOfCompound     : The dose or amount of a Compound used. Must be a numeric quantity with a unit (e.g. "10 uM", "1000 U/mL", "100 ng/mL"). Must pair with a named Compound.
Depletion                   : Method to remove high-abundance proteins (e.g. "albumin depletion kit").
DevelopmentalStage          : Developmental stage of source (e.g. "adult", "P7 pup").
Disease                     : Disease state or diagnosis (e.g. "breast cancer", "Type 2 diabetes").
DiseaseTreatment            : Pre-treatment applied to diseased samples (e.g. "chemotherapy", "radiation").
GeneticModification         : Genetic alteration in organism/cells including tagging constructs (e.g. "GFP-tagged", "Flag-tagged", "CRISPR knockout", "overexpression").
Genotype                    : Genotypic background (e.g. "C57BL/6J", "BRCA1-mutant").
GrowthRate                  : Doubling time or growth rate (e.g. "24 h doubling time").
Label                       : Isobaric or metabolic label applied (e.g. "TMT-126", "SILAC heavy", "label-free", "triple SILAC"). The method name ("SILAC", "triple SILAC"), shorthand isotope pairs ("Arg6/Lys4", "Arg10/Lys8"), AND full IUPAC isotope names ("L-lysine - U-13C4 15N0 (Lys4)", "L-arginine - U-13C6 15N4 (Arg10)") are ALL valid Label representations. Full isotope chemical names are semantically equivalent to their shorthand forms.
MaterialType                : Broad material class (e.g. "tissue", "cell line", "biofluid").
Modification                : PTM studied or enriched for (e.g. "phosphorylation", "ubiquitination", "acetylation").
NumberOfBiologicalReplicates: Total COUNT of biological replicates in the study (e.g. "3"). NOT an identifier.
NumberOfSamples             : Total COUNT of samples processed (e.g. "12").
NumberOfTechnicalReplicates : Total COUNT of technical replicates per sample (e.g. "2").
Organism                    : Source species (e.g. "Homo sapiens", "Mus musculus").
OrganismPart                : Tissue or organ of origin (e.g. "liver", "brain cortex", "plasma").
OriginSiteDisease           : Anatomical site of disease origin (e.g. "colon", "prostate").
PooledSample                : Whether multiple samples were pooled (e.g. "pool1 of reps1-3").
ReductionReagent            : Chemical used to reduce disulfide bonds (e.g. "DTT", "TCEP").
SamplingTime                : Time point of sample collection (e.g. "T0", "24 h post-treatment").
Sex                         : Donor sex (e.g. "male", "female").
Specimen                    : Description of biological specimen type (e.g. "biopsy", "plasma", "urine").
SpikedCompound              : Exogenous standard or spike-in added to sample (e.g. "iRT peptides").
Staining                    : Staining applied to sample prior to MS.
Strain                      : Animal strain (e.g. "BALB/c", "FVB/N").
SyntheticPeptide            : Indicates a synthetic peptide sample.
Temperature                 : Growth or perturbation temperature (e.g. "37 C").
Time                        : Broad time parameter of experiment (e.g. "day 5", "week 2").
Treatment                   : Experimental treatment applied (e.g. "EGF stimulation 10 ng/mL 30 min").
TumorCellularity            : Percentage of tumor cells in sample (e.g. "80%").
TumorGrade                  : Histological tumor grade (e.g. "Grade II").
TumorSize                   : Physical size of tumor (e.g. "3 cm diameter").
TumorSite                   : Anatomical site of tumor (e.g. "breast", "pancreas").
TumorStage                  : Clinical staging (e.g. "Stage III").

TECHNICAL / MS ENTITY TYPES:
AcquisitionMethod           : MS acquisition scheme (e.g. "DDA", "DIA", "PRM"). Must be a scheme name, NOT an instrument name.
CollisionEnergy             : Collision energy in MS/MS (e.g. "27 eV", "normalized CE 28%", "25").
EnrichmentMethod            : Peptide/protein enrichment protocol before MS (e.g. "TiO2 phosphopeptide enrichment").
FlowRateChromatogram        : LC flow rate (e.g. "300 nL/min").
FractionationMethod         : OFF-LINE fractionation of bulk sample before LC-MS runs (e.g. "high-pH reverse-phase fractionation", "SCX"). Distinct from Separation.
FractionIdentifier          : ID label for each fraction (e.g. "F1", "F2", "fraction 3").
FragmentationMethod         : Ion-fragmentation technique (e.g. "HCD", "CID", "ETD").
FragmentMassTolerance       : Mass tolerance for fragment ion matching in search (e.g. "0.02 Da", "20 mDa").
GradientTime                : Total LC gradient length (e.g. "120 min", "90-min gradient").
Instrument                  : Mass spectrometer make and model (e.g. "Thermo Q-Exactive Plus", "Orbitrap Fusion Lumos"). LC instruments (e.g. "EASY nLC", "Eksigent") are NOT this type.
IonizationType              : Ionization source type (e.g. "nano-ESI", "MALDI").
MS2MassAnalyzer             : Analyzer used for MS2 scans (e.g. "Orbitrap", "ion trap").
NumberOfMissedCleavages     : Max missed cleavages allowed in database search (e.g. "2").
NumberOfFractions           : Total COUNT of fractions generated per sample.
PrecursorMassTolerance      : Mass tolerance for precursor matching in search (e.g. "10 ppm", "5 ppm").
Separation                  : ON-LINE LC separation method -- must describe the stationary phase, column chemistry, or chromatographic approach (e.g. "reverse-phase C18 nano-LC", "RPLC", "ReproSil-Pur C18-AQ beads"). LC instrument model names alone (e.g. "EASY nLC-II", "Eksigent nanoLC", "HPLC") are NOT valid Separation values.
"""

ANNOTATION_CRITERIA = """You are an expert in proteomics and mass-spectrometry experimental metadata.

You are evaluating ONE predicted annotation extracted from a proteomics paper.
The 5-tier deterministic matching has ALREADY been run. GEval is called ONLY when
tiers 1-4 (exact, normalized, ontology, hierarchical) all failed. Your job is to
assess semantic / domain-knowledge equivalence -- tier 5.

INPUT LAYOUT:
  - 'actual output'   = the annotation type and extracted value to evaluate.
  - 'expected output' = all ground-truth values for this annotation type (match ANY one),
                        OR "NO GROUND TRUTH - evaluate against source text only."
  - 'context'[0]      = full source paper text (abstract + methods).
  - 'context'[1]      = definition of this specific annotation type.
  - 'context'[2]      = definitions of ALL annotation types (for type-mismatch detection).

BUNDLED COMPOUND+CONCENTRATION RULE (CRITICAL -- read before Step 1):
  The extraction pipeline intentionally annotates Compound and concentration
  together in a single value. A Compound annotation whose value includes a numeric
  dose or concentration qualifier (e.g. "20 nM Calyculin A", "PDP(m)-Nal 50 uM",
  "10 ug/mL EGF", "Calyculin A 20 nM") is VALID and CORRECT. You MUST NOT flag
  such values as a type mismatch or suggest that the correct type is
  ConcentrationOfCompound. The chemical name must be present in the value; if it
  is, the bundled form is treated as a correct Compound annotation provided the
  value appears in the source text. Similarly, a ground-truth value of just the
  chemical name (e.g. "Calyculin A") matches a predicted value that bundles the
  concentration (e.g. "20 nM Calyculin A") -- treat them as semantically equivalent.

STEP 1 - TYPE MISMATCH CHECK:
  Carefully read the annotation type definition in context[1].
  Ask: does the extracted value fit this definition, or does it clearly belong
  to a different annotation type listed in context[2]?

  Enforce these distinctions without exception:
  - Separation vs Instrument: LC instrument model names ("EASY nLC-II", "Eksigent",
    "HPLC") are NOT valid Separation values. Separation must describe the stationary
    phase or chromatographic method. If mismatched, state "Correct type: X".
  - AcquisitionMethod must be a scheme name (DDA, DIA, PRM), not an instrument.
  - Bait must name a specific protein; generic tags alone are GeneticModification.
  - Compound is a chemical name; a numeric dose WITHOUT any chemical name is
    ConcentrationOfCompound. A value containing BOTH a chemical name AND a numeric
    dose is a valid Compound -- do NOT flag it as ConcentrationOfCompound.

  IMPORTANT: If you identify a type mismatch, you MUST write "Correct type: <TypeName>"
  explicitly in your reason. Do NOT use vague language like "belongs to another category"
  without naming the correct type. If you are uncertain whether it is a mismatch,
  do NOT flag it as a mismatch.

STEP 2 - HALLUCINATION CHECK:
  Search context[0] for the extracted value or any unambiguous synonym/abbreviation.
  If absent entirely, hallucination = true, score = 0.
  Standard expansions, morphological variants, unit reformatting are NOT hallucinations.

  NOTE: A hallucinated value cannot simultaneously be a type mismatch. If the value
  is not found in the source text at all, report hallucination=true and do NOT flag
  type_mismatch. Type mismatch requires the value to actually appear in the source.

STEP 3 - VALUE CORRECTNESS (semantic / domain equivalence):
  The value did NOT match exactly, after normalization, as an abbreviation, or as a
  substring -- so assess domain-knowledge equivalence:
  - "SILAC" is semantically equivalent to "triple SILAC".
  - Full IUPAC isotope names are equivalent to shorthand pairs:
      "L-lysine - U-13C4 15N0 (Lys4)"   == "Arg6/Lys4"
      "L-arginine - U-13C6 15N0 (Arg6)" == "Arg6/Lys4"
      "L-lysine - U-13C6 15N2 (Lys8)"   == "Arg10/Lys8"
      "L-arginine - U-13C6 15N4 (Arg10)"== "Arg10/Lys8"
  - Digit vs word ("3" vs "three") = correct.
  - A Compound value bundling a concentration (e.g. "20 nM Calyculin A") is
    semantically equivalent to a ground-truth Compound value of just "Calyculin A"
    and vice versa -- treat them as correct matches.
  - Abbreviation vs expansion already tested in tier 3 -- if reached here, assess
    whether the values describe the same real-world entity by domain knowledge.

STEP 4 - COMPLETENESS (lenient):
  Incomplete ONLY when the value is so truncated it cannot discriminate the entity,
  or a critical qualifier that changes identity is missing.

SCORING:
  10 = Semantically equivalent to a ground-truth value, not hallucinated, correct type.
   8 = Correct but minor completeness issue.
   5 = Same domain concept but detail-level mismatch with ground truth.
   3 = Wrong value or type mismatch, but value present in source.
   0 = Hallucinated -- value absent from source text.

Your reason MUST be a complete paragraph covering type, hallucination, correctness,
completeness, and which ground-truth value was matched (if any). Do not truncate."""

ANNOTATION_STEPS = [
    "Read the BUNDLED COMPOUND+CONCENTRATION RULE at the top of the criteria. If the annotation type is Compound and the value contains both a chemical name and a concentration, it is valid -- do NOT flag as type mismatch or ConcentrationOfCompound.",
    "Read context[1] (type definition) and context[2] (all type definitions).",
    "Check type mismatch: if mismatched, you MUST write 'Correct type: X' in your reason. If hallucinated (score=0), do NOT flag as type mismatch.",
    "Search context[0] for the extracted value or any unambiguous synonym. If absent, hallucination=true, score=0.",
    "Assess semantic / domain-knowledge equivalence against all ground-truth values in 'expected output'. Full IUPAC isotope names for SILAC labels are equivalent to their shorthand pairs. A Compound value bundling a concentration is equivalent to the bare compound name.",
    "Check completeness leniently.",
    "Assign score 0-10. Write a complete paragraph covering all four checks.",
]

MISSED_CRITERIA = """You are an expert in proteomics and mass-spectrometry experimental metadata.

Determine whether the source text contains extractable information for a specific
annotation type that was NOT extracted by the pipeline.

INPUT LAYOUT:
  - 'actual output'   = states which annotation type was NOT extracted.
  - 'expected output' = "Check if source text contains this annotation type."
  - 'context'[0]      = full source paper text (abstract + methods).
  - 'context'[1]      = definition of this annotation type.

SCORING:
  10 = Source text CLEARLY and EXPLICITLY mentions a value for this annotation type.
   7 = Value can be REASONABLY INFERRED from context.
   3 = Only a vague or ambiguous mention.
   0 = Field is genuinely NOT mentioned or not applicable to this paper."""

MISSED_STEPS = [
    "Read context[0] (source text) carefully.",
    "Use context[1] (type definition) to determine what a valid value would look like.",
    "Search context[0] for any mention of this annotation type.",
    "Score 10 if clearly present, 7 if inferable, 3 if vague, 0 if absent.",
]

_ABBREV_MAP = {
    "hcd":  "higher energy collisional dissociation",
    "cid":  "collision induced dissociation",
    "etd":  "electron transfer dissociation",
    "dda":  "data dependent acquisition",
    "dia":  "data independent acquisition",
    "prm":  "parallel reaction monitoring",
    "iaa":  "iodoacetamide",
    "iam":  "iodoacetamide",
    "dtt":  "dithiothreitol",
    "tcep": "tris(2-carboxyethyl)phosphine",
    "tmt":  "tandem mass tag",
    "itraq":"isobaric tags for relative and absolute quantitation",
    "silac":"stable isotope labeling by amino acids in cell culture",
    "fasp": "filter aided sample preparation",
    "sds":  "sodium dodecyl sulfate",
    "lc":   "liquid chromatography",
    "ms":   "mass spectrometry",
    "rplc": "reverse phase liquid chromatography",
    "scx":  "strong cation exchange",
    "sax":  "strong anion exchange",
    "esi":  "electrospray ionization",
    "maldi":"matrix assisted laser desorption ionization",
    "tof":  "time of flight",
    "nano-esi": "nano electrospray ionization",
    "uplc": "ultra performance liquid chromatography",
    "uhplc":"ultra high performance liquid chromatography",
}

_WORD_NUM_MAP = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}

_CONC_PREFIX_RE = re.compile(
    r"^\s*\d+(?:[.,]\d+)?\s*"
    r"(?:fM|pM|nM|uM|mM|M|ng/mL|ng/ml|ug/mL|ug/ml|mg/mL|mg/ml"
    r"|U/mL|U/ml|IU/mL|IU/ml|%|v/v|w/v)\s+",
    re.IGNORECASE,
)
_CONC_SUFFIX_RE = re.compile(
    r"\s+\d+(?:[.,]\d+)?\s*"
    r"(?:fM|pM|nM|uM|mM|M|ng/mL|ng/ml|ug/mL|ug/ml|mg/mL|mg/ml"
    r"|U/mL|U/ml|IU/mL|IU/ml|%|v/v|w/v)\s*$",
    re.IGNORECASE,
)


def _strip_concentration(s: str) -> str:
    s = _CONC_PREFIX_RE.sub("", s)
    s = _CONC_SUFFIX_RE.sub("", s)
    return s.strip()


def _ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _tier1_norm(s: str) -> str:
    return s.strip()

def _tier2_norm(s: str) -> str:
    s = _ascii_fold(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for word, digit in _WORD_NUM_MAP.items():
        s = re.sub(rf"\b{word}\b", digit, s)
    s = re.sub(r"(\d)\s+([cfk])\b", r"\1\2", s)
    return s

def _strip_trailing_units(s: str) -> str:
    _UNIT_WORDS = {
        "fractions", "fraction", "samples", "sample", "replicates", "replicate",
        "runs", "run", "min", "minutes", "hours", "hour", "sec", "seconds",
        "ms", "ul", "ml", "l", "ng", "ug", "mg", "g", "kg",
        "nm", "um", "mm", "cm", "m", "ppm", "da", "mda", "ev", "kev",
        "nl", "nL",
    }
    tokens = s.split()
    if not tokens:
        return s
    if not re.match(r"^\d", tokens[0]):
        return s
    while tokens and tokens[-1].lower() in _UNIT_WORDS:
        tokens.pop()
    return " ".join(tokens) if tokens else s

def _expand_abbrevs(s: str) -> str:
    tokens = s.lower().split()
    expanded = []
    for t in tokens:
        expanded.append(_ABBREV_MAP.get(t, t))
    return " ".join(expanded)

def _tier3_norm(s: str) -> str:
    base = _tier2_norm(s)
    return _tier2_norm(_expand_abbrevs(base))


def _best_tier(predicted: str, gt_candidates: list, annotation_type: str = "") -> dict:

    result = {
        "tier1_exact":        False,
        "tier2_normalized":   False,
        "tier3_ontology":     False,
        "tier4_hierarchical": False,
        "any_match":          False,
        "no_match":           True,
        "match_score":        0.0,
        "match_type":         "NO_MATCH",
        "matched_gt":         "",
    }

    if not gt_candidates:
        return result

    is_compound = annotation_type.lower() == "compound"

    p1  = _tier1_norm(predicted)
    p2  = _tier2_norm(predicted)
    p3  = _tier3_norm(predicted)
    p2s = _strip_trailing_units(p2)

    if is_compound:
        p1_sc = _tier1_norm(_strip_concentration(predicted))
        p2_sc = _tier2_norm(_strip_concentration(predicted))
        p3_sc = _tier3_norm(_strip_concentration(predicted))
    else:
        p1_sc = p1
        p2_sc = p2
        p3_sc = p3

    for gt in gt_candidates:
        g1    = _tier1_norm(gt)
        g1_sc = _tier1_norm(_strip_concentration(gt)) if is_compound else g1
        if p1 == g1 or (is_compound and (p1_sc == g1 or p1 == g1_sc or p1_sc == g1_sc)):
            result.update({
                "tier1_exact": True, "any_match": True, "no_match": False,
                "match_score": 1.0, "match_type": "EXACT", "matched_gt": gt,
            })
            return result

    for gt in gt_candidates:
        g2    = _tier2_norm(gt)
        g2s   = _strip_trailing_units(g2)
        g2_sc = _tier2_norm(_strip_concentration(gt)) if is_compound else g2
        if (p2 == g2 or p2s == g2s
                or (is_compound and (p2_sc == g2 or p2 == g2_sc or p2_sc == g2_sc))):
            result.update({
                "tier2_normalized": True, "any_match": True, "no_match": False,
                "match_score": 0.9, "match_type": "NORMALIZED", "matched_gt": gt,
            })
            return result

    for gt in gt_candidates:
        g3    = _tier3_norm(gt)
        g3s   = _strip_trailing_units(g3)
        g3_sc = _tier3_norm(_strip_concentration(gt)) if is_compound else g3
        p3s   = _strip_trailing_units(p3)
        p3s_sc = _strip_trailing_units(p3_sc) if is_compound else p3s

        if (p3 == g3 or p3s == g3s
                or (is_compound and (
                    p3_sc == g3 or p3 == g3_sc or p3_sc == g3_sc
                    or p3s_sc == g3s or p3s == _strip_trailing_units(g3_sc)
                ))):
            result.update({
                "tier3_ontology": True, "any_match": True, "no_match": False,
                "match_score": 0.8, "match_type": "ONTOLOGY", "matched_gt": gt,
            })
            return result

        candidate_pairs = [(p3, g3)]
        if is_compound:
            candidate_pairs += [(p3_sc, g3), (p3, g3_sc), (p3_sc, g3_sc)]
        for pc, gc in candidate_pairs:
            if pc and gc and (pc in gc or gc in pc):
                if len(min(pc, gc, key=len)) >= 3:
                    pred_is_superset = gc in pc
                    result.update({
                        "tier4_hierarchical": True, "any_match": True, "no_match": False,
                        "match_score": 0.7, "match_type": "HIERARCHICAL", "matched_gt": gt,
                        "pred_is_superset": pred_is_superset,
                    })
                    return result

    for gt in gt_candidates:
        p2_tok    = set(p2.split())
        g2_tok    = set(_tier2_norm(gt).split())
        p2_sc_tok = set(p2_sc.split()) if is_compound else p2_tok
        g2_sc_tok = set(_tier2_norm(_strip_concentration(gt)).split()) if is_compound else g2_tok

        token_pairs = [(p2_tok, g2_tok)]
        if is_compound:
            token_pairs += [
                (p2_sc_tok, g2_tok),
                (p2_tok,    g2_sc_tok),
                (p2_sc_tok, g2_sc_tok),
            ]

        for pt, gt_tok in token_pairs:
            if pt and gt_tok:
                overlap = pt & gt_tok
                if len(overlap) >= max(1, min(len(pt), len(gt_tok)) // 2):
                    if len(overlap) >= 2 or (len(overlap) == 1 and len(list(overlap)[0]) >= 4):
                        pred_is_superset = len(pt) >= len(gt_tok)
                        result.update({
                            "tier4_hierarchical": True, "any_match": True, "no_match": False,
                            "match_score": 0.65, "match_type": "HIERARCHICAL", "matched_gt": gt,
                            "pred_is_superset": pred_is_superset,
                        })
                        return result

    return result


class GPT52MediumReasoning(DeepEvalBaseLLM):

    def __init__(self):
        self._model_name = "gpt-5.2"
        self._client     = None

    def load_model(self):
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def generate(self, prompt: str, schema=None) -> str:
        client      = self.load_model()
        call_kwargs = {
            "model":                 self._model_name,
            "messages":              [{"role": "user", "content": prompt}],
            "reasoning_effort":      "medium",
            "max_completion_tokens": 32000,
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

_gpt52_model       = None
_annotation_metric = None
_missed_metric     = None

def _get_judge_model():
    global _gpt52_model
    if _gpt52_model is None:
        _gpt52_model = GPT52MediumReasoning()
    return _gpt52_model

def _get_annotation_metric():
    global _annotation_metric
    if _annotation_metric is None:
        _annotation_metric = GEval(
            name="AnnotationJudge",
            criteria=ANNOTATION_CRITERIA,
            evaluation_steps=ANNOTATION_STEPS,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=_get_judge_model(),
            threshold=0.5,
        )
    return _annotation_metric

def _get_missed_metric():
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
            model=_get_judge_model(),
            threshold=0.5,
        )
    return _missed_metric

def _type_def_for(annotation_type: str) -> str:
    for line in _ENTITY_DEFINITIONS.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith(annotation_type + " ") or stripped.startswith(annotation_type + ":"):
            return stripped
    return f"Annotation type: {annotation_type}"

def _score_to_flags(score: float) -> dict:
    if score >= 0.8:
        return {"value_correct": True,  "value_complete": True,  "hallucination": False}
    elif score >= 0.5:
        return {"value_correct": True,  "value_complete": False, "hallucination": False}
    elif score >= 0.2:
        return {"value_correct": False, "value_complete": False, "hallucination": False}
    else:
        return {"value_correct": False, "value_complete": False, "hallucination": True}


def _parse_type_mismatch_from_reason(
    reason: str,
    geval_score: float,
    annotation_type: str = "",
) -> tuple:

    if geval_score is None or geval_score == 0.0:
        return False, None

    known_types = set(
        line.strip().split(":")[0].strip()
        for line in _ENTITY_DEFINITIONS.strip().splitlines()
        if ":" in line
        and not line.strip().startswith("SAMPLE")
        and not line.strip().startswith("TECHNICAL")
        and line.strip().split(":")[0].strip()
    )

    explicit_patterns = [
        r"[Cc]orrect\s+type:\s*(\w+)",
        r"[Cc]orrect(?:\s+annotation)?\s+type\s+is\s+['\"]?(\w+)['\"]?",
    ]

    for pat in explicit_patterns:
        m = re.search(pat, reason, re.IGNORECASE)
        if m:
            candidate = m.group(1)

            if (annotation_type.lower() == "compound"
                    and candidate.lower() == "concentrationofcompound"):
                return False, None

            if candidate in known_types:
                return True, candidate
            return True, None

    return False, None


def _run_geval_semantic(
    source_text:     str,
    annotation_type: str,
    extracted_value: str,
    expected_output: str,
) -> dict:
    type_def      = _type_def_for(annotation_type)
    actual_output = f"Annotation type: {annotation_type}\nExtracted value: {extracted_value}"
    context       = [source_text, type_def, _ENTITY_DEFINITIONS]

    test_case = LLMTestCase(
        input=f"Evaluate extraction of '{annotation_type}' = '{extracted_value}'",
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
    )

    metric = _get_annotation_metric()
    try:
        metric.measure(test_case)
        score  = metric.score
        reason = metric.reason or ""
    except Exception as e:
        print(f"          [GEval error] {annotation_type}: {type(e).__name__}: {e}")
        return {
            "type_mismatch":   None,
            "correct_type":    None,
            "value_correct":   None,
            "value_complete":  None,
            "hallucination":   None,
            "issue_summary":   f"GEval error: {e}",
            "geval_score":     None,
            "match_type":      "NO_MATCH",
            "match_score":     0.0,
            "tier5_semantic":  False,
        }

    flags = _score_to_flags(score)

    is_mismatch, correct_type = _parse_type_mismatch_from_reason(
        reason, score, annotation_type
    )

    if flags["hallucination"]:
        is_mismatch  = False
        correct_type = None

    is_semantic_match = (
        flags["value_correct"]
        and not is_mismatch
        and not flags["hallucination"]
    )

    return {
        "type_mismatch":   is_mismatch,
        "correct_type":    correct_type,
        "value_correct":   flags["value_correct"],
        "value_complete":  flags["value_complete"],
        "hallucination":   flags["hallucination"],
        "issue_summary":   f"GEval={score:.2f}. {reason}",
        "geval_score":     score,
        "match_type":      "SEMANTIC" if is_semantic_match else "NO_MATCH",
        "match_score":     round(score * 0.6, 4) if is_semantic_match else 0.0,
        "tier5_semantic":  is_semantic_match,
    }

def check_missed_annotation(source_text: str, annotation_type: str) -> float:
    type_def  = _type_def_for(annotation_type)
    test_case = LLMTestCase(
        input=f"Is '{annotation_type}' present in the source text?",
        actual_output=f"Annotation type '{annotation_type}' was NOT extracted by the pipeline.",
        expected_output="Check if source text contains this annotation type.",
        context=[source_text, type_def],
    )
    metric = _get_missed_metric()
    try:
        metric.measure(test_case)
        return metric.score
    except Exception as e:
        print(f"          [missed GEval error] {annotation_type}: {e}")
        return 0.0

def evaluate_with_geval(
    paper_id:          str,
    predicted_anns:    dict,
    ground_truth_anns: dict,
    source_text:       str,
) -> dict:
    if not source_text or not predicted_anns:
        return {}

    gt_lists: dict[str, list[str]] = {}
    for k, v in ground_truth_anns.items():
        if not v:
            continue
        gt_lists[k] = [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]

    per_annotation_review = []
    n_total    = 0
    n_halluc   = 0
    n_mismatch = 0
    n_wrong    = 0
    n_incompl  = 0

    for ann_type, values in predicted_anns.items():
        if not isinstance(values, list):
            values = [values]

        gt_candidates = gt_lists.get(ann_type, [])
        gt_display    = "; ".join(gt_candidates) if gt_candidates else ""
        has_golden    = bool(gt_candidates)

        for val in values:
            val = str(val).strip()
            if not val or val.lower() == "unknown":
                continue

            n_total += 1

            # Pass annotation_type so Compound concentration-stripping applies.
            tier = _best_tier(val, gt_candidates, ann_type) if has_golden else {
                "tier1_exact": False, "tier2_normalized": False,
                "tier3_ontology": False, "tier4_hierarchical": False,
                "any_match": False, "no_match": True,
                "match_score": 0.0, "match_type": "NO_MATCH", "matched_gt": "",
            }

            if tier["tier1_exact"]:
                print(f"        [T1-exact]  {ann_type}: '{val[:55]}' OK")
                per_annotation_review.append({
                    "annotation_type":  ann_type,
                    "extracted_value":  val,
                    "golden_value":     gt_display,
                    "has_golden":       has_golden,
                    "match_score":      tier["match_score"],
                    "match_type":       tier["match_type"],
                    "tier1_exact":      True,
                    "tier2_normalized": False,
                    "tier3_ontology":   False,
                    "tier4_hierarchical": False,
                    "tier5_semantic":   False,
                    "any_match":        True,
                    "no_match":         False,
                    "type_mismatch":    False,
                    "correct_type":     None,
                    "value_correct":    True,
                    "value_complete":   True,
                    "hallucination":    False,
                    "issue_summary":    f"Tier 1 -- exact match with GT '{tier['matched_gt']}'.",
                    "corrected_value":  None,
                })
                continue

            if tier["tier2_normalized"]:
                print(f"        [T2-norm]   {ann_type}: '{val[:55]}' OK")
                per_annotation_review.append({
                    "annotation_type":  ann_type,
                    "extracted_value":  val,
                    "golden_value":     gt_display,
                    "has_golden":       has_golden,
                    "match_score":      tier["match_score"],
                    "match_type":       tier["match_type"],
                    "tier1_exact":      False,
                    "tier2_normalized": True,
                    "tier3_ontology":   False,
                    "tier4_hierarchical": False,
                    "tier5_semantic":   False,
                    "any_match":        True,
                    "no_match":         False,
                    "type_mismatch":    False,
                    "correct_type":     None,
                    "value_correct":    True,
                    "value_complete":   True,
                    "hallucination":    False,
                    "issue_summary":    f"Tier 2 -- normalized match with GT '{tier['matched_gt']}'.",
                    "corrected_value":  None,
                })
                continue

            if tier["tier3_ontology"]:
                print(f"        [T3-onto]   {ann_type}: '{val[:55]}' OK")
                per_annotation_review.append({
                    "annotation_type":  ann_type,
                    "extracted_value":  val,
                    "golden_value":     gt_display,
                    "has_golden":       has_golden,
                    "match_score":      tier["match_score"],
                    "match_type":       tier["match_type"],
                    "tier1_exact":      False,
                    "tier2_normalized": False,
                    "tier3_ontology":   True,
                    "tier4_hierarchical": False,
                    "tier5_semantic":   False,
                    "any_match":        True,
                    "no_match":         False,
                    "type_mismatch":    False,
                    "correct_type":     None,
                    "value_correct":    True,
                    "value_complete":   True,
                    "hallucination":    False,
                    "issue_summary":    f"Tier 3 -- ontology/abbreviation match with GT '{tier['matched_gt']}'.",
                    "corrected_value":  None,
                })
                continue

            if tier["tier4_hierarchical"]:
                pred_is_superset = tier.get("pred_is_superset", False)
                complete_flag    = True if pred_is_superset else False
                summary_suffix   = "predicted is more detailed than GT (superset -- complete)." if pred_is_superset else "predicted is less specific than GT (subset -- incomplete)."
                if not complete_flag:
                    n_incompl += 1
                print(f"        [T4-hier]   {ann_type}: '{val[:55]}' {'(superset)' if pred_is_superset else '(subset)'}")
                per_annotation_review.append({
                    "annotation_type":  ann_type,
                    "extracted_value":  val,
                    "golden_value":     gt_display,
                    "has_golden":       has_golden,
                    "match_score":      tier["match_score"],
                    "match_type":       tier["match_type"],
                    "tier1_exact":      False,
                    "tier2_normalized": False,
                    "tier3_ontology":   False,
                    "tier4_hierarchical": True,
                    "tier5_semantic":   False,
                    "any_match":        True,
                    "no_match":         False,
                    "type_mismatch":    False,
                    "correct_type":     None,
                    "value_correct":    True,
                    "value_complete":   complete_flag,
                    "hallucination":    False,
                    "issue_summary":    f"Tier 4 -- hierarchical match with GT '{tier['matched_gt']}'; {summary_suffix}",
                    "corrected_value":  None,
                })
                continue

            if gt_candidates:
                expected_str = f"Ground truth values for '{ann_type}' (match ANY one): {gt_display}"
            else:
                expected_str = "NO GROUND TRUTH - evaluate against source text only."

            print(
                f"        [T5-GEval]  {ann_type}: '{val[:50]}' "
                f"vs gt={[g[:25] for g in gt_candidates]}",
                end="  ",
            )

            geval_result = _run_geval_semantic(source_text, ann_type, val, expected_str)

            is_semantic = geval_result.get("tier5_semantic", False)
            geval_s     = geval_result.get("geval_score")

            if geval_s is not None:
                print(
                    f"-> score={geval_s:.2f}  correct={geval_result['value_correct']}  "
                    f"mismatch={geval_result['type_mismatch']}  halluc={geval_result['hallucination']}"
                )
            else:
                print("-> ERROR")

            if geval_result.get("hallucination"):
                n_halluc += 1
            elif geval_result.get("type_mismatch"):
                n_mismatch += 1
            elif geval_result.get("value_correct") is False:
                n_wrong += 1
            elif geval_result.get("value_correct") and not geval_result.get("value_complete"):
                n_incompl += 1

            per_annotation_review.append({
                "annotation_type":    ann_type,
                "extracted_value":    val,
                "golden_value":       gt_display,
                "has_golden":         has_golden,
                "match_score":        geval_result["match_score"],
                "match_type":         geval_result["match_type"],
                "tier1_exact":        False,
                "tier2_normalized":   False,
                "tier3_ontology":     False,
                "tier4_hierarchical": False,
                "tier5_semantic":     is_semantic,
                "any_match":          is_semantic,
                "no_match":           not is_semantic,
                "type_mismatch":      geval_result["type_mismatch"],
                "correct_type":       geval_result["correct_type"],
                "value_correct":      geval_result["value_correct"],
                "value_complete":     geval_result["value_complete"],
                "hallucination":      geval_result["hallucination"],
                "issue_summary":      geval_result["issue_summary"],
                "corrected_value":    None,
            })

    extracted_types    = set(predicted_anns.keys())
    missed_extractions = []

    for ann_type, gt_vals in gt_lists.items():
        if ann_type in extracted_types:
            continue
        correct_value_str = "; ".join(gt_vals)
        print(f"        [missed] {ann_type}: in GT but not extracted (gt={correct_value_str[:60]})")
        missed_extractions.append({
            "annotation_type": ann_type,
            "correct_value":   correct_value_str,
            "in_ground_truth": True,
        })

    n_missed = len(missed_extractions)
    n_clean  = sum(
        1 for r in per_annotation_review
        if not r.get("type_mismatch")
        and r.get("value_correct")
        and r.get("value_complete")
        and not r.get("hallucination")
    )
    n_value_correct = sum(
        1 for r in per_annotation_review
        if r.get("value_correct") and not r.get("hallucination")
    )
    score_overall = n_value_correct / n_total if n_total > 0 else 0.0
    halluc_rate   = n_halluc / n_total if n_total > 0 else 0.0
    summary       = (
        f"{n_clean}/{n_total} fully correct, {n_value_correct}/{n_total} value correct. "
        f"{n_halluc} hallucinated, {n_mismatch} type mismatch, "
        f"{n_wrong} wrong, {n_incompl} incomplete, {n_missed} missed."
    )

    print(
        f"        [summary] clean={n_clean}/{n_total}  halluc={n_halluc}  "
        f"mismatch={n_mismatch}  wrong={n_wrong}  incompl={n_incompl}  missed={n_missed}"
    )

    return {
        "completeness_score":     score_overall,
        "completeness_reason":    summary,
        "accuracy_score":         score_overall,
        "accuracy_reason":        summary,
        "consistency_score":      score_overall,
        "consistency_reason":     summary,
        "per_annotation_review":  per_annotation_review,
        "missed_extractions":     missed_extractions,
        "hallucination_rate":     halluc_rate,
        "type_mismatch_count":    n_mismatch,
        "wrong_value_count":      n_wrong,
        "incomplete_count":       n_incompl,
        "missed_count":           n_missed,
        "overall_summary":        summary,
        "judge_n_total":          n_total,
        "judge_n_clean":          n_clean,
        "judge_n_value_correct":  n_value_correct,
    }

def parse_ann_file(file_path):
    entities = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                entity_type, value = line.split(":", 1)
                entity_type = entity_type.strip()
                value       = value.strip()
                if entity_type and value:
                    entities.append((entity_type, value, line_num))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return entities

def should_exclude_entity(entity_type: str) -> bool:
    return entity_type.lower().strip().startswith("factorvalue")

def normalize_entity_type(entity_type: str) -> str:
    ENTITY_TYPE_ALIASES = {
        "diseasetreatment": "treatment", "disease treatment": "treatment",
        "experimentaltreatment": "treatment", "experimental treatment": "treatment",
        "condition": "treatment",
        "enzyme": "cleavageagent", "protease": "cleavageagent",
        "digestingenzyme": "cleavageagent", "digestingagent": "cleavageagent",
        "tissue": "organismpart", "organ": "organismpart",
        "bodyfluids": "organismpart", "bodyfluid": "organismpart",
        "sample source": "organismpart", "samplesource": "organismpart",
        "sampletype": "materialtype", "sample type": "materialtype",
        "biologicalmaterial": "materialtype", "biological material": "materialtype",
        "samplematrix": "materialtype", "sample matrix": "materialtype",
        "chromatography": "separation", "lcmethod": "separation",
        "lc method": "separation", "liquidchromatography": "separation",
        "liquid chromatography": "separation",
        "massspectrometer": "instrument", "mass spectrometer": "instrument",
        "msplatform": "instrument", "ms platform": "instrument",
        "labelingmethod": "label", "labeling method": "label",
        "quantificationmethod": "label", "quantification method": "label",
        "isotopelabel": "label",
        "acquisitionmode": "acquisitionmethod", "acquisition mode": "acquisitionmethod",
        "scanmode": "acquisitionmethod", "scan mode": "acquisitionmethod",
        "biologicalreplicates": "numberofbiologicalreplicates",
        "biological replicates": "numberofbiologicalreplicates",
        "technicalreplicates": "numberoftechnicalreplicates",
        "technical replicates": "numberoftechnicalreplicates",
        "replicates": "numberofbiologicalreplicates",
        "enrichment": "enrichmentmethod", "enrichment method": "enrichmentmethod",
        "fractionation": "enrichmentmethod", "fractionation method": "enrichmentmethod",
    }
    if "[" in entity_type:
        entity_type = entity_type.split("[")[1].rstrip("]")
    key = entity_type.strip().lower()
    if key in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[key]
    key_nospace = key.replace(" ", "").replace("-", "").replace("_", "")
    if key_nospace in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[key_nospace]
    return entity_type.strip()

def load_papers_dataset(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        papers_dict = {}
        for paper in papers:
            paper_id = paper.get("filename") or paper.get("stem", "")
            if not paper_id:
                continue
            key = paper_id.replace(".txt", "").strip()
            papers_dict[key] = {
                "abstract":      paper.get("abstract")      or "",
                "methods":       paper.get("methods")       or "",
                "supplementary": paper.get("supplementary") or "",
            }
        print(f"  Loaded {len(papers_dict)} papers")
        return papers_dict
    except FileNotFoundError:
        print(f"Warning: papers_dataset.json not found at: {json_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not load papers dataset: {e}")
        return {}

def run_geval_judge(gt_dir, pred_dir, papers_json):
    gt_files     = {f.stem: f for f in Path(gt_dir).glob("*.ann")}
    pred_files   = {f.stem: f for f in Path(pred_dir).glob("*.ann")}
    common_files = set(gt_files.keys()) & set(pred_files.keys())

    if not common_files:
        print("No matching .ann files found!")
        return None, None

    print(f"Found {len(common_files)} matching files")
    papers_dict     = load_papers_dataset(papers_json)
    per_paper_judge = {}
    per_file_stats  = []

    for file_name in sorted(common_files):
        print(f"\n{'='*80}\nProcessing: {file_name}.ann")

        gt_entities   = parse_ann_file(gt_files[file_name])
        pred_entities = parse_ann_file(pred_files[file_name])

        gt_filtered   = [(t, txt, ln) for t, txt, ln in gt_entities   if not should_exclude_entity(t)]
        pred_filtered = [(t, txt, ln) for t, txt, ln in pred_entities if not should_exclude_entity(t)]

        source_sections = papers_dict.get(file_name, {})
        source_text = "\n\n".join(
            f"### {section.upper()}\n{text}"
            for section, text in source_sections.items() if text
        )

        if not source_text:
            print(f"  WARNING: no source text for {file_name}, skipping.")
            continue

        gt_normalized   = [(normalize_entity_type(t), txt, ln, file_name) for t, txt, ln in gt_filtered]
        pred_normalized = [(normalize_entity_type(t), txt, ln, file_name) for t, txt, ln in pred_filtered]

        pred_anns_dict = defaultdict(list)
        gt_anns_dict   = defaultdict(list)
        for et, txt, _, _ in pred_normalized:
            pred_anns_dict[et].append(txt)
        for et, txt, _, _ in gt_normalized:
            gt_anns_dict[et].append(txt)

        n_pred = sum(len(v) for v in pred_anns_dict.values())
        n_gt   = sum(len(v) for v in gt_anns_dict.values())
        print(f"  {n_pred} pred values | {n_gt} GT values | {len(source_text):,} chars")

        judge_scores = evaluate_with_geval(
            file_name, dict(pred_anns_dict), dict(gt_anns_dict), source_text
        )

        if judge_scores:
            per_paper_judge[file_name] = judge_scores
            per_file_stats.append({
                "file_name":                  file_name,
                "judge_n_total":              judge_scores.get("judge_n_total", 0),
                "judge_n_clean":              judge_scores.get("judge_n_clean", 0),
                "judge_n_value_correct":      judge_scores.get("judge_n_value_correct", 0),
                "deepeval_accuracy":          judge_scores.get("accuracy_score"),
                "deepeval_completeness":      judge_scores.get("completeness_score"),
                "judge_hallucination_rate":   judge_scores.get("hallucination_rate"),
                "judge_type_mismatch_count":  judge_scores.get("type_mismatch_count"),
                "judge_wrong_value_count":    judge_scores.get("wrong_value_count"),
                "judge_incomplete_count":     judge_scores.get("incomplete_count", 0),
                "judge_missed_count":         judge_scores.get("missed_count"),
            })

    return per_paper_judge, per_file_stats

def save_geval_results(per_paper_judge, per_file_stats, output_dir):
    rows = []
    for paper_id, judge_result in per_paper_judge.items():
        for review in judge_result.get("per_annotation_review", []):
            rows.append({
                "paper_id":           paper_id,
                "annotation_type":    review.get("annotation_type"),
                "extracted_value":    review.get("extracted_value"),
                "golden_value":       review.get("golden_value", ""),
                "has_golden":         review.get("has_golden", False),
                "match_score":        review.get("match_score"),
                "match_type":         review.get("match_type", ""),
                "tier1_exact":        review.get("tier1_exact", False),
                "tier2_normalized":   review.get("tier2_normalized", False),
                "tier3_ontology":     review.get("tier3_ontology", False),
                "tier4_hierarchical": review.get("tier4_hierarchical", False),
                "tier5_semantic":     review.get("tier5_semantic", False),
                "any_match":          review.get("any_match", False),
                "no_match":           review.get("no_match", True),
                "type_mismatch":      review.get("type_mismatch"),
                "correct_type":       review.get("correct_type"),
                "value_correct":      review.get("value_correct"),
                "value_complete":     review.get("value_complete"),
                "hallucination":      review.get("hallucination"),
                "issue_summary":      review.get("issue_summary"),
                "corrected_value":    review.get("corrected_value"),
            })
        for missed in judge_result.get("missed_extractions", []):
            rows.append({
                "paper_id":           paper_id,
                "annotation_type":    missed.get("annotation_type"),
                "extracted_value":    "MISSING",
                "golden_value":       missed.get("correct_value", ""),
                "has_golden":         missed.get("in_ground_truth", False),
                "match_score":        0.0,
                "match_type":         "NO_MATCH",
                "tier1_exact":        False,
                "tier2_normalized":   False,
                "tier3_ontology":     False,
                "tier4_hierarchical": False,
                "tier5_semantic":     False,
                "any_match":          False,
                "no_match":           True,
                "type_mismatch":      False,
                "correct_type":       None,
                "value_correct":      False,
                "value_complete":     False,
                "hallucination":      False,
                "issue_summary":      "Not extracted -- present in source text or ground truth.",
                "corrected_value":    missed.get("correct_value"),
            })
    if rows:
        path = os.path.join(output_dir, "llm_judge_annotation_review.csv")
        pd.DataFrame(rows).to_csv(path, index=False, escapechar="\\", doublequote=True)
        print(f"  Annotation review saved to: {path}")

    if per_file_stats:
        pd.DataFrame(per_file_stats).to_csv(
            os.path.join(output_dir, "llm_judge_per_file.csv"),
            index=False, escapechar="\\", doublequote=True,
        )
        print("  Per-file judge stats saved.")

def _me_counts_from_reviews(reviews):
    n_correct = n_halluc = n_mismatch = n_wrong = n_incompl = n_vc = 0
    for r in reviews:
        hall  = bool(r.get("hallucination"))
        mis   = bool(r.get("type_mismatch"))
        vc    = bool(r.get("value_correct"))
        vcomp = bool(r.get("value_complete"))
        if hall:
            n_halluc  += 1
        elif mis:
            n_mismatch += 1
        elif not vc:
            n_wrong   += 1
        elif not vcomp:
            n_incompl += 1
        else:
            n_correct += 1
        if vc and not hall:
            n_vc += 1
    return n_correct, n_halluc, n_mismatch, n_wrong, n_incompl, n_vc

def plot_geval_results(per_file_stats, output_dir):
    judge_stats = [
        s for s in per_file_stats
        if s.get("judge_n_total") is not None and s.get("judge_n_total", 0) > 0
    ]
    if not judge_stats:
        print("No GEval scores to plot.")
        return

    CLR_CLEAN    = "#27ae60"
    CLR_HALL     = "#e74c3c"
    CLR_MISMATCH = "#9b59b6"
    CLR_WRONG    = "#e67e22"
    CLR_INCOMPL  = "#3498db"
    CLR_MISSED   = "#95a5a6"

    names      = [s["file_name"] for s in judge_stats]
    n_total    = np.array([s.get("judge_n_total", 0)               for s in judge_stats], dtype=float)
    n_vc       = np.array([s.get("judge_n_value_correct", 0)        for s in judge_stats], dtype=float)
    n_hall     = np.array([round(s.get("judge_hallucination_rate", 0) * s.get("judge_n_total", 0))
                           for s in judge_stats], dtype=float)
    n_mismatch = np.array([s.get("judge_type_mismatch_count", 0)   for s in judge_stats], dtype=float)
    n_wrong    = np.array([s.get("judge_wrong_value_count", 0)      for s in judge_stats], dtype=float)
    n_incompl  = np.array([s.get("judge_incomplete_count", 0)       for s in judge_stats], dtype=float)
    n_missed   = np.array([s.get("judge_missed_count", 0)           for s in judge_stats], dtype=float)
    n_correct  = n_total - n_hall - n_mismatch - n_wrong - n_incompl

    x = np.arange(len(names))
    w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("GEval -- Annotation Quality Analysis (mutually exclusive categories)",
                 fontsize=15, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    b = np.zeros(len(names))
    for arr, col, lbl in [
        (n_correct,  CLR_CLEAN,    "Correct"),
        (n_incompl,  CLR_INCOMPL,  "Incomplete"),
        (n_wrong,    CLR_WRONG,    "Wrong Value"),
        (n_mismatch, CLR_MISMATCH, "Type Mismatch"),
        (n_hall,     CLR_HALL,     "Hallucinated"),
    ]:
        bars = ax.bar(x, arr, w, bottom=b, color=col, alpha=0.9, edgecolor="white", lw=0.5, label=lbl)
        for bar, val, bot in zip(bars, arr, b):
            if val >= 2:
                ax.text(bar.get_x()+bar.get_width()/2, bot+val/2, str(int(val)),
                        ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        b += arr
    for i, tot in enumerate(n_total):
        ax.text(i, tot+0.3, f"n={int(tot)}", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Annotations", fontsize=10)
    ax.set_title("Annotation Quality per Paper (raw counts, ME categories)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)

    ax = axes[0, 1]
    denom = np.where(n_total == 0, 1, n_total)
    b2 = np.zeros(len(names))
    for arr, col, lbl in [
        (n_correct,  CLR_CLEAN,    "Correct"),
        (n_incompl,  CLR_INCOMPL,  "Incomplete"),
        (n_wrong,    CLR_WRONG,    "Wrong Value"),
        (n_mismatch, CLR_MISMATCH, "Type Mismatch"),
        (n_hall,     CLR_HALL,     "Hallucinated"),
    ]:
        pct = arr / denom * 100
        bars = ax.bar(x, pct, w, bottom=b2, color=col, alpha=0.9, edgecolor="white", lw=0.5, label=lbl)
        for bar, val, bot in zip(bars, pct, b2):
            if val >= 5:
                ax.text(bar.get_x()+bar.get_width()/2, bot+val/2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        b2 += pct
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of Annotations", fontsize=10); ax.set_ylim(0, 115)
    ax.axhline(100, color="#aaa", lw=0.8, linestyle="--")
    ax.set_title("Annotation Quality per Paper (normalised %, ME categories)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)

    ax = axes[1, 0]
    wg = 0.14
    offsets = np.array([-2, -1, 0, 1, 2]) * wg
    error_data = [
        (n_hall,     CLR_HALL,     "Hallucinated"),
        (n_mismatch, CLR_MISMATCH, "Type Mismatch"),
        (n_wrong,    CLR_WRONG,    "Wrong Value"),
        (n_incompl,  CLR_INCOMPL,  "Incomplete"),
        (n_missed,   CLR_MISSED,   "Missed"),
    ]
    for (data, color, label), offset in zip(error_data, offsets):
        bars = ax.bar(x + offset, data, wg, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, data):
            if val >= 1:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                        str(int(val)), ha="center", va="bottom", fontsize=7, color=color, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Error Type Breakdown per Paper", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)

    ax = axes[1, 1]
    grand_total = int(n_total.sum()) + int(n_missed.sum())
    agg_labels = ["Correct", "Incomplete", "Wrong\nValue", "Type\nMismatch", "Hallucinated", "Missed"]
    agg_vals   = [int(n_correct.sum()), int(n_incompl.sum()), int(n_wrong.sum()),
                  int(n_mismatch.sum()), int(n_hall.sum()), int(n_missed.sum())]
    agg_colors = [CLR_CLEAN, CLR_INCOMPL, CLR_WRONG, CLR_MISMATCH, CLR_HALL, CLR_MISSED]
    bars4 = ax.bar(agg_labels, agg_vals, color=agg_colors, alpha=0.88, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars4, agg_vals):
        pct = val / grand_total * 100 if grand_total > 0 else 0
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Count (all papers)", fontsize=10)
    ax.set_title(f"Aggregate Annotation Quality  (total incl. missed={grand_total})", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    vc_pct = int(n_vc.sum()) / int(n_total.sum()) * 100 if int(n_total.sum()) > 0 else 0
    ax.text(0.5, 0.97, f"Value-correct rate: {vc_pct:.1f}%  |  Strictly correct: {agg_vals[0]/int(n_total.sum())*100:.1f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, fontweight="bold", color=CLR_CLEAN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor=CLR_CLEAN, alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, "llm_judge_summary.png")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"GEval summary plot saved to: {out}")

    acc_vals     = [s["judge_n_value_correct"] / s["judge_n_total"]
                    if s.get("judge_n_total") and s["judge_n_total"] > 0 else 0
                    for s in judge_stats]
    overall_mean = float(np.mean(acc_vals))

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bar_colors = ["#27ae60" if v >= 0.7 else "#e67e22" if v >= 0.5 else "#e74c3c" for v in acc_vals]
    xb = np.arange(len(names))
    bars_acc = ax2.bar(xb, acc_vals, color=bar_colors, alpha=0.88, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars_acc, acc_vals):
        ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222222")
    ax2.axhline(overall_mean, color="#2471a3", lw=2, linestyle="--", label=f"Mean = {overall_mean:.3f}")
    ax2.axhline(0.7, color="#e74c3c", lw=1.5, linestyle=":", alpha=0.7, label="Threshold = 0.70")
    ax2.set_xticks(xb); ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("value_correct rate  (correct / total predicted)", fontsize=12, fontweight="bold")
    ax2.set_title("LLM Judge -- Extraction Accuracy per Paper  (based on value_correct)",
                  fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylim([0, 1.15]); ax2.grid(axis="y", alpha=0.3, linestyle="--"); ax2.set_axisbelow(True)
    ax2.legend(fontsize=10)
    ax2.text(0.98, 0.97, f"Overall mean: {overall_mean:.1%}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=11, fontweight="bold", color="#27ae60",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60", alpha=0.85))
    plt.tight_layout()
    out2 = os.path.join(output_dir, "llm_judge_accuracy.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Accuracy plot saved to: {out2}")

def plot_tier_distribution(per_paper_judge: dict, output_dir: str):
    T1  = "#10B981"
    T2  = "#3B82F6"
    T3  = "#8B5CF6"
    T4  = "#F59E0B"
    T5  = "#EC4899"
    NOM = "#EF4444"
    MIS = "#94A3B8"

    paper_ids = sorted(per_paper_judge.keys())
    counts = {pid: {"T1":0,"T2":0,"T3":0,"T4":0,"T5":0,"NO_MATCH":0,"MISSING":0}
              for pid in paper_ids}

    for pid in paper_ids:
        reviews = per_paper_judge[pid].get("per_annotation_review", [])
        missed  = per_paper_judge[pid].get("missed_extractions", [])
        for r in reviews:
            mt = r.get("match_type", "NO_MATCH")
            if mt == "EXACT":
                counts[pid]["T1"] += 1
            elif mt == "NORMALIZED":
                counts[pid]["T2"] += 1
            elif mt == "ONTOLOGY":
                counts[pid]["T3"] += 1
            elif mt == "HIERARCHICAL":
                counts[pid]["T4"] += 1
            elif mt == "SEMANTIC":
                counts[pid]["T5"] += 1
            else:
                counts[pid]["NO_MATCH"] += 1
        counts[pid]["MISSING"] = len(missed)

    names  = paper_ids
    x      = np.arange(len(names))
    w      = 0.6
    keys   = ["T1","T2","T3","T4","T5","NO_MATCH","MISSING"]
    colors = [T1, T2, T3, T4, T5, NOM, MIS]
    labels = ["Tier1 Exact","Tier2 Normalized","Tier3 Ontology",
              "Tier4 Hierarchical","Tier5 Semantic","No Match","Missing"]

    data = {k: np.array([counts[p][k] for p in names], dtype=float) for k in keys}
    totals = sum(data[k] for k in keys)

    fig, axes = plt.subplots(1, 2, figsize=(max(18, len(names)*1.4), 7))

    ax = axes[0]
    bottom = np.zeros(len(names))
    for k, col, lbl in zip(keys, colors, labels):
        bars = ax.bar(x, data[k], w, bottom=bottom, color=col, alpha=0.88,
                      edgecolor="white", linewidth=0.5, label=lbl)
        for bar, val, bot in zip(bars, data[k], bottom):
            if val >= 2:
                ax.text(bar.get_x() + bar.get_width()/2, bot + val/2,
                        str(int(val)), ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
        bottom += data[k]
    for i, tot in enumerate(totals):
        ax.text(i, tot + 0.3, f"n={int(tot)}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Annotations", fontsize=11)
    ax.set_title("5-Tier Match Distribution per Paper (raw counts)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)

    ax2 = axes[1]
    denom = np.where(totals == 0, 1, totals)
    bottom2 = np.zeros(len(names))
    for k, col, lbl in zip(keys, colors, labels):
        pct = data[k] / denom * 100
        bars2 = ax2.bar(x, pct, w, bottom=bottom2, color=col, alpha=0.88,
                        edgecolor="white", linewidth=0.5, label=lbl)
        for bar, val, bot in zip(bars2, pct, bottom2):
            if val >= 5:
                ax2.text(bar.get_x() + bar.get_width()/2, bot + val/2,
                         f"{val:.0f}%", ha="center", va="center",
                         fontsize=7, fontweight="bold", color="white")
        bottom2 += pct
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("% of Annotations", fontsize=11)
    ax2.set_ylim(0, 115); ax2.axhline(100, color="#aaa", lw=0.8, linestyle="--")
    ax2.set_title("5-Tier Match Distribution per Paper (normalised %)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3, linestyle="--"); ax2.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(output_dir, "tier_distribution_per_paper.png")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: tier_distribution_per_paper.png")

    agg = {k: int(sum(data[k])) for k in keys}
    grand = sum(agg.values())

    fig, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Aggregate 5-Tier Match Distribution  (total = {grand})",
                 fontsize=14, fontweight="bold")

    ax3 = axes2[0]
    wedge_vals   = [agg[k] for k in keys]
    wedge_colors = colors
    wedge_labels = [f"{lbl}\n{agg[k]} ({agg[k]/grand*100:.1f}%)" if grand > 0 else lbl
                    for lbl, k in zip(labels, keys)]
    wedges, texts = ax3.pie(
        wedge_vals, colors=wedge_colors, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        pctdistance=0.75,
    )
    ax3.legend(wedges, wedge_labels, loc="center left",
               bbox_to_anchor=(1, 0.5), fontsize=9)
    ax3.set_title("Donut breakdown", fontsize=11)
    centre = plt.Circle((0, 0), 0.55, color="white")
    ax3.add_artist(centre)
    any_match_pct = sum(agg[k] for k in ["T1","T2","T3","T4","T5"]) / grand * 100 if grand else 0
    ax3.text(0, 0, f"{any_match_pct:.1f}%\nmatched",
             ha="center", va="center", fontsize=11, fontweight="bold", color="#333")

    ax4 = axes2[1]
    bar_vals = [agg[k] for k in keys]
    bars4 = ax4.bar(labels, bar_vals, color=colors, alpha=0.88,
                    edgecolor="white", linewidth=1.0)
    for bar, val in zip(bars4, bar_vals):
        pct = val / grand * 100 if grand else 0
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax4.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax4.set_ylabel("Total count (all papers)", fontsize=11)
    ax4.set_title("Bar breakdown", fontsize=11)
    ax4.grid(axis="y", alpha=0.3, linestyle="--"); ax4.set_axisbelow(True)

    plt.tight_layout()
    out2 = os.path.join(output_dir, "tier_distribution_aggregate.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: tier_distribution_aggregate.png")

if __name__ == "__main__":
    print("PART 3 -- GEVAL AS JUDGE (5-tier matching)")
    print(f"Ground Truth Dir : {GROUND_TRUTH_DIR}")
    print(f"Predictions Dir  : {PREDICTIONS_DIR}")
    print(f"Papers JSON      : {PAPERS_JSON}")
    print(f"Results Dir      : {RESULTS_DIR}")
    print(f"Judge model      : {EVALUATION_MODEL} (via DeepEval GEval, reasoning=medium)\n")

    per_paper_judge, per_file_stats = run_geval_judge(GROUND_TRUTH_DIR, PREDICTIONS_DIR, PAPERS_JSON)

    if per_paper_judge:
        save_geval_results(per_paper_judge, per_file_stats, RESULTS_DIR)
        plot_geval_results(per_file_stats, RESULTS_DIR)
        plot_tier_distribution(per_paper_judge, RESULTS_DIR)

        print(f"\nProcessed {len(per_paper_judge)} papers.")
        for stat in per_file_stats:
            n_tot   = stat.get("judge_n_total", 0)
            n_clean = stat.get("judge_n_clean", 0)
            acc     = n_clean / n_tot if n_tot > 0 else 0
            print(f"  {stat['file_name']:<30} total={n_tot}  clean={n_clean}  "
                  f"accuracy={acc:.2f}  halluc_rate={stat.get('judge_hallucination_rate', 0):.2f}")

        print("\nPART 3 COMPLETE.")
    else:
        print("No results produced.")
