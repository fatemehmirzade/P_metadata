import os
import sys
import json
import csv
import re
from pathlib import Path
from openai import OpenAI

MODEL = "gpt-5.2"
REASONING_EFFORT = "medium"

ANNOTATION_TYPES = [
    "AcquisitionMethod","Age","AlkylationReagent","AnatomicSiteTumor",
    "AncestryCategory","Bait","BiologicalReplicate","BMI","CellLine",
    "CellPart","CellType","CleavageAgent","CollisionEnergy","Compound",
    "Depletion","DevelopmentalStage","Disease","DiseaseTreatment",
    "EnrichmentMethod","FactorValue","FlowRateChromatogram",
    "FragmentationMethod","FragmentMassTolerance","FractionationMethod",
    "FractionIdentifier","GeneticModification","Genotype","GradientTime",
    "GrowthRate","Instrument","IonizationType","Label","MaterialType",
    "Modification","MS2MassAnalyzer","NumberOfBiologicalReplicates",
    "NumberOfFractions","NumberOfMissedCleavages","NumberOfSamples",
    "NumberOfTechnicalReplicates","Organism","OrganismPart",
    "OriginSiteDisease","PooledSample","PrecursorMassTolerance",
    "ReductionReagent","SamplingTime","Separation","Sex","Specimen",
    "SpikedCompound","Staining","Strain","SyntheticPeptide","Temperature",
    "Time","Treatment","TumorCellularity","TumorGrade","TumorSite",
    "TumorStage","TumorSize",
]

ANNOTATION_TYPE_DEFINITIONS = {
    "Age": "Age of the donor or developmental stage of the organism (e.g. '45 years', 'E14.5 embryo').",
    "AlkylationReagent": (
        "A chemical (like Iodoacetamide (IAA) or N-ethylmaleimide (NEM)) that irreversibly adds an alkyl group "
        "to the free sulfhydryl (-SH) of cysteine residues, blocking disulfide bonds and preventing protein re-folding."
    ),
    "AnatomicSiteTumor": "Anatomical location from which a tumor sample was taken (e.g. 'left lung lobe').",
    "AncestryCategory": "Donor ancestry or ethnicity category (e.g. 'European', 'East Asian').",
    "Bait": "The protein or molecule used as bait in an affinity-purification experiment (AP-MS / BioID / co-IP).",
    "BMI": "Body-Mass Index of the donor (kg/m2).",
    "BiologicalReplicate": "Identifier for biological replicates (e.g. 'bioRep1', 'bioRep2').",
    "CellLine": "Name of the immortalized cell line (e.g. 'HEK293T', 'U2OS').",
    "CellPart": "Subcellular compartment or fraction (e.g. 'nucleus', 'mitochondria').",
    "CellType": "Primary cell type or lineage (e.g. 'neurons', 'fibroblasts').",
    "CleavageAgent": "Protease or chemical used to digest proteins (e.g. 'trypsin', 'chymotrypsin').",
    "CollisionEnergy": "Collision energy applied in MS/MS (e.g. '27 eV').",
    "Compound": (
        "A chemical or small molecule added to the sample as a perturbation, treatment, or stimulus "
        "(e.g. 'human recombinant IFNb 1000 U/mL', 'rapamycin 10 nM', 'EGF 50 ng/mL'). "
        "The full compound description including dose/concentration is a valid single Compound value. "
        "A FactorValue that records only the dose or concentration of the same compound "
        "(e.g. FactorValue: '1000 U/mL') is a legitimate companion annotation that summarises the "
        "experimental axis defined by that compound -- it does NOT conflict with Compound. "
        "Both annotations can and should coexist."
    ),
    "Depletion": (
        "Method or condition used to remove or reduce a specific component from a sample. "
        "In nutrient-biology contexts this includes nutrient-deprivation conditions "
        "(e.g. 'S-depleted' meaning sulfur-depleted growth medium). "
        "The key signal is that something is being removed or is absent. "
        "Must NOT be confused with enrichment conditions or nutrient-replete controls: "
        "'S-rich' or 'S-replete' describes a nutrient-rich condition, which is NOT a depletion "
        "and does not belong under Depletion. Nutrient-rich control conditions belong to "
        "Treatment or FactorValue."
    ),
    "DevelopmentalStage": (
        "The developmental or growth stage of the organism, organ, or tissue at the time of sampling. "
        "Valid values describe what biological stage the material is at, not when it was collected. "
        "Examples: 'mature pistils', 'three DAP' (days after pollination as a seed developmental stage), "
        "'seven DAP', 'E14.5 embryo', 'P7 pup', 'adult', 'seedling stage'. "
        "DAP (days after pollination) values such as 'three DAP' or 'seven DAP' are valid "
        "DevelopmentalStage values because DAP describes the developmental age of the seed/embryo. "
        "The same DAP value may also appear under SamplingTime (when it was collected) -- "
        "this dual annotation is correct and expected."
    ),
    "Disease": "Disease state or diagnosis (e.g. 'breast cancer', 'Type 2 diabetes').",
    "DiseaseTreatment": "Pre-treatment applied to diseased samples (e.g. 'chemotherapy', 'radiation').",
    "EnrichmentMethod": "Peptide/protein enrichment protocol used (e.g. 'TiO2 phosphopeptide enrichment').",
    "FactorValue": (
        "The value of a meaningful experimental variable that distinguishes sample groups. "
        "FactorValue is the ONLY annotation type that is allowed to repeat -- multiple FactorValue "
        "entries are valid when the study has multiple experimental axes. "
        "A FactorValue may mirror values from other annotation types (e.g. a disease name that is "
        "also annotated under Disease, or a dose that is also part of a Compound annotation). "
        "Must be a concrete, self-contained biological/clinical/experimental descriptor. "
        "Must NOT be: a raw cell count, a bare sentence fragment, or a floating preposition "
        "with no referent (e.g. 'prior to', '3 months after' alone)."
    ),
    "FlowRateChromatogram": "LC flow rate (e.g. '300 nL/min').",
    "FragmentationMethod": "Ion-fragmentation technique (e.g. 'HCD', 'CID', 'ETD').",
    "FragmentMassTolerance": (
        "Mass tolerance for fragment ion matching in database search "
        "(e.g. '0.02 Da', '20 ppm', '20, 10, and 7 ppm'). "
        "A value listing multiple tolerances for different acquisition modes is valid as a single entry."
    ),
    "FractionationMethod": (
        "Any off-line method used to fractionate the bulk sample into the primary samples used in MS. "
        "When multiple fractionation methods are combined in one experiment, the combined description "
        "is the correct single annotation (e.g. 'A combination of SEC and top-down ODG'). "
        "A sub-step or component method that is already fully described by a more complete combined "
        "annotation should be removed as redundant "
        "(e.g. if 'A combination of SEC and top-down ODG' is present, "
        "then 'a discontinuous top-down ODG' alone is redundant and should be removed)."
    ),
    "FractionIdentifier": (
        "A numeric or text identifier for a specific chromatographic or offline fraction. "
        "Valid values identify one or more specific fractions by number or name "
        "(e.g. 'F1', 'fractions 4-7', 'Fractions 9 and 10'). "
        "Capitalisation differences (e.g. 'fractions 4-7' vs 'Fractions 4-7') do not create "
        "distinct fractions -- treat them as the same identifier. "
        "Must NOT be a description of pooled samples (those belong to PooledSample)."
    ),
    "GeneticModification": "Any genetic alteration in the source organism/cells (e.g. 'GFP-tagged', 'knockout of gene X').",
    "Genotype": "Genotypic background (e.g. 'C57BL/6J', 'BRCA1-mutant').",
    "GradientTime": (
        "The total duration of the LC gradient used to elute peptides. "
        "Valid values must state or imply a specific duration, either as a bare number with unit "
        "('120 min', '60 min'), a phrase that includes the duration "
        "('over 180 min', 'Standard 90-min gradients', 'from 5% B to 15% B over 60 min', "
        "'A linear 150-min gradient'). "
        "A value is INVALID if it describes gradient slope or composition without stating total time "
        "(e.g. 'over a 60-min gradient' is valid because it states 60 min; "
        "'gradient from 5 to 40% acetonitrile' without a duration is invalid). "
        "IMPORTANT: a phrase like 'over a 60-min gradient' IS valid -- it contains a duration (60 min). "
        "Do NOT reject gradient-time values just because they are expressed as a phrase rather than "
        "a bare number. The test is: can you extract a specific duration in minutes (or other time unit)? "
        "If yes, the value is valid."
    ),
    "GrowthRate": "Doubling time or growth rate of cell cultures (e.g. '24 h doubling').",
    "Instrument": "Mass spectrometer make/model (e.g. 'Thermo Q-Exactive Plus'). Must NOT be LC hardware or software.",
    "IonizationType": "Ionization source (e.g. 'nanoESI', 'MALDI').",
    "Label": (
        "Isobaric, isotopic, or metabolic labelling strategy applied to peptides or proteins "
        "(e.g. 'TMT-126', 'SILAC heavy', 'iTRAQ 4-plex'). "
        "Label-free quantification (LFQ) is also a valid Label value because it describes the "
        "quantification/labelling strategy (absence of label), not a treatment or method step."
    ),
    "MaterialType": "Broad class of material (e.g. 'tissue', 'cell line', 'biofluid').",
    "Modification": "Post-translational modification enriched or studied (e.g. 'phosphorylation', 'ubiquitination').",
    "MS2MassAnalyzer": "Analyzer used for MS2 (e.g. 'orbitrap', 'ion trap').",
    "NumberOfBiologicalReplicates": (
        "Total number of biological replicates in the study. "
        "May be expressed as a digit ('3'), a written-out number ('three'), "
        "or a short phrase ('n=3 biological replicates'). All are valid."
    ),
    "NumberOfFractions": (
        "Total number of fractions generated from each sample. "
        "May be a digit ('24'), a written-out number, or a short descriptive phrase. All are valid."
    ),
    "NumberOfMissedCleavages": (
        "Maximum number of missed cleavages allowed in the database search. "
        "May be expressed as a digit ('2'), a written-out number ('two'), "
        "or a short descriptive phrase ('maximum two missed cleavages', 'up to 2'). "
        "All are valid -- do NOT reject for being non-integer in form. "
        "Reject only if the string conveys no missed-cleavage count at all."
    ),
    "NumberOfSamples": (
        "Total number of samples processed. "
        "May be a digit, a written-out number, or a short descriptive phrase. All are valid."
    ),
    "NumberOfTechnicalReplicates": (
        "Total number of technical replicates per sample. "
        "May be a digit, a written-out number, or a short descriptive phrase. All are valid."
    ),
    "Organism": "Source species (NCBI Taxonomy ID and name, e.g. '9606 (Homo sapiens)').",
    "OrganismPart": (
        "Tissue, organ, biofluid, or life-stage-specific cell population of origin "
        "(Uberon term preferred, e.g. 'UBERON:0002107 (liver)'). "
        "Parasite life-stage cell forms such as 'tachyzoites' or 'bradyzoites' are valid OrganismPart "
        "values because they identify the specific cellular form of the organism used as source material. "
        "Biofluids such as 'urine' or 'plasma' are also valid OrganismPart values."
    ),
    "OriginSiteDisease": "Anatomical site of disease origin (e.g. 'colon', 'prostate').",
    "PooledSample": "Indicates if multiple samples were pooled (e.g. 'pool1 of reps1-3').",
    "PrecursorMassTolerance": "Mass tolerance for precursor matching (e.g. '10 ppm').",
    "ReductionReagent": "Chemical used to reduce disulfide bonds (e.g. 'DTT', 'TCEP').",
    "SamplingTime": (
        "The time point at which a sample was collected, relative to a defined event or time-course. "
        "Examples: 'T0', '24 h post-treatment', 'three and seven days after pollination', "
        "'day 3', '48 h'. "
        "In plant developmental studies, 'three DAP' and 'seven DAP' (days after pollination) "
        "describe BOTH a developmental stage AND a collection time point -- they are valid SamplingTime "
        "values when used to anchor when the sample was taken. "
        "HOWEVER, a SamplingTime value that also contains non-temporal descriptors such as "
        "size or morphology (e.g. 'three DAP (around 6 mm in length)') is INVALID for SamplingTime "
        "because it mixes a physical descriptor into a time annotation -- it belongs to DevelopmentalStage "
        "or OrganismPart instead. "
        "A combined time-range expression like 'three and seven days after pollination' IS valid because "
        "it is a pure temporal description of two collection points."
    ),
    "Separation": (
        "Any on-line chromatographic method or column used to separate peptides immediately before MS. "
        "Valid values include column names, stationary-phase descriptions "
        "(e.g. 'ReproSil-Pur 120 C18-AQ 3 um diameter beads'), column format descriptions, "
        "and general method labels (e.g. 'reversed-phase nanoLC'). "
        "Must NOT be an off-line fractionation method (that belongs to FractionationMethod)."
    ),
    "Sex": "Donor sex (e.g. 'male', 'female').",
    "Specimen": "Description of biological specimen type (e.g. 'biopsy', 'plasma', 'cell pellet').",
    "SpikedCompound": (
        "An exogenous standard or spike-in compound added to the sample for quantification or "
        "quality-control purposes (e.g. 'iRT peptides', 'BSA spike-in'). "
        "If a compound appears in both SpikedCompound and Treatment/Compound, the paper context "
        "determines which type is correct: if it is added as a standard/control rather than as "
        "a stimulus or perturbation, it belongs to SpikedCompound."
    ),
    "Staining": "Any staining applied to the sample prior to mass spec.",
    "Strain": (
        "The genetic strain or engineered line of a NON-HUMAN source organism used in the experiment. "
        "Valid examples: 'BALB/c', 'C57BL/6J', 'FVB/N' for inbred mouse strains; "
        "'RH', 'Me49' for Toxoplasma gondii parasite strains; "
        "named transgenic or knock-in lines of model organisms (mouse, rat, zebrafish, fly, worm). "
        "CRITICAL EXCLUSIONS -- Strain must NOT be used for: "
        "  - Human or primate cell line identifiers: alphanumeric IDs like 'ES04', 'H1', 'H9', "
        "    'WA09' that refer to a human cell line belong to CellLine, not Strain. "
        "  - Bait: Strain is the organism line, not the affinity-purification bait protein. "
        "  - GeneticModification: the modification event belongs to GeneticModification; "
        "    the resulting line name belongs to Strain."
    ),
    "SyntheticPeptide": "Indicates a synthetic peptide sample (e.g. 'synthetic phosphopeptide library').",
    "Temperature": "Growth temperature or perturbation temperature (e.g. '37 degC', '4 degC on ice').",
    "Time": (
        "A broad time parameter for the experiment that does not fit the more specific SamplingTime "
        "or DevelopmentalStage categories -- e.g. 'day 5', 'week 2', '72 h'. "
        "IMPORTANT: if a time value (such as 'three DAP' or 'seven DAP') is already annotated under "
        "DevelopmentalStage or SamplingTime with clear biological meaning in that context, "
        "then a bare repetition of the same value under Time is redundant and should be removed. "
        "Time should be used only when the time expression does not fit the more specific types."
    ),
    "Treatment": (
        "An experimental treatment actively applied to samples, typically a drug, compound, stimulus, "
        "or defined intervention with dose and/or duration (e.g. 'drug X 5 uM 24 h', 'UV irradiation 30 min'). "
        "Must NOT be: a disease name, a patient-group label, a phenotype descriptor, "
        "or a spike-in compound used as an internal standard."
    ),
    "TumorCellularity": "Percentage of tumor cells in the sample (e.g. '80%').",
    "TumorGrade": "Histological grade (e.g. 'Grade II').",
    "TumorSize": "Physical size of the tumor (e.g. '3 cm diameter').",
    "TumorSite": "Anatomical site of tumor (e.g. 'breast', 'pancreas').",
    "TumorStage": "Clinical staging (e.g. 'Stage III').",
    "AcquisitionMethod": "MS acquisition scheme (e.g. 'DDA', 'DIA', 'PRM').",
}

ANN_DIR = "./annotations"
JSON_PATH = "./papers_dataset.json"
OUTPUT_DIR = "./filtered_output"
CSV_OUT = "./removed_annotations.csv"

ALLOW_CROSS_TYPE_SHARING_WITH = {"FactorValue"}

def read_ann_file(path: str) -> list[tuple[str, str]]:
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = re.match(r'^([A-Za-z]+):\s*(.*)', line)
            if m:
                atype, value = m.group(1), m.group(2)
                if atype in ANNOTATION_TYPES:
                    entries.append((atype, value))
    return entries

def read_paper_text(json_data: list[dict], stem: str) -> str:
    for item in json_data:
        item_stem = item.get("stem", "") or Path(item.get("filename", "")).stem
        if item_stem == stem:
            parts = []
            for key in ("abstract", "methods", "supplementary"):
                val = item.get(key) or ""
                if val:
                    parts.append(val)
            return "\n\n".join(parts)
    return ""

def group_by_type(entries: list[tuple[str, str]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for atype, value in entries:
        groups.setdefault(atype, []).append(value)
    return groups

def remove_exact_duplicates(values: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    seen: dict[str, str] = {}
    kept = []
    removed = []
    for v in values:
        norm = v.strip().lower()
        if norm not in seen:
            seen[norm] = v
            kept.append(v)
        else:
            removed.append((v, seen[norm]))
    return kept, removed

def _extract_raw_text(response) -> str:
    raw_text = ""
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for content_block in item.content:
                if getattr(content_block, "type", None) == "output_text":
                    raw_text = content_block.text
                    break
        if raw_text:
            break
    return raw_text

def _safe_parse_api_response(
    raw_text: str,
    required_keys: tuple[str, ...],
    original_values: set[str],
    keep_key: str,
    remove_key: str,
    atype: str,
    context_label: str,
) -> dict:
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        result = json.loads(clean)
        for k in required_keys:
            if k not in result:
                raise ValueError(f"Missing '{k}' key in response")

        returned_values = set(result.get(keep_key, [])) | {
            r["value"] for r in result.get(remove_key, [])
        }

        missing = original_values - returned_values
        if missing:
            print(
                f"    [WARN] [{context_label}] Model dropped {len(missing)} value(s) "
                f"for {atype}, restoring to {keep_key}: {missing}",
                file=sys.stderr,
            )
            result[keep_key] = list(result.get(keep_key, [])) + list(missing)

        hallucinated = returned_values - original_values
        if hallucinated:
            print(
                f"    [WARN] [{context_label}] Model returned unknown value(s) "
                f"for {atype}, ignoring: {hallucinated}",
                file=sys.stderr,
            )
            result[keep_key] = [v for v in result[keep_key] if v in original_values]
            result[remove_key] = [
                r for r in result[remove_key] if r.get("value") in original_values
            ]

        return result

    except Exception as e:
        print(f"    [WARN] [{context_label}] Could not parse response for {atype}: {e}", file=sys.stderr)
        print(f"    Raw response: {raw_text[:500]}", file=sys.stderr)
        return {keep_key: list(original_values), remove_key: []}


ICL_DEDUP_EXAMPLES = """
IN-CONTEXT LEARNING EXAMPLES FOR DEDUPLICATION

-- EXAMPLE 1 -- Annotation type: GradientTime
Candidate values:
  1. 20 min
  2. A linear 150-min gradient
  3. 150-min
  4. 20 min
  5. 20 min

Reasoning:
  "20 min" (1): Keep -- a specific gradient duration; the first occurrence is kept.

  "A linear 150-min gradient" (2): Keep -- a different duration (150 min vs 20 min) with
    extra detail ("linear"). Genuinely distinct from value 1.

  "150-min" (3): Remove -- this is a bare abbreviation of value 2. Every piece of
    information in "150-min" is already contained in "A linear 150-min gradient" which
    also specifies the gradient shape. "150-min" adds nothing new. Rule: when a value
    is a strict subset of another value that is already in the keep list, remove the
    less informative one.

  "20 min" (4): Remove -- exact duplicate of value 1 (same string, same type).

  "20 min" (5): Remove -- exact duplicate of value 1 again.

Output:
{
  "keep": ["20 min", "A linear 150-min gradient"],
  "remove": [
    {"value": "150-min", "reason": "bare duration already fully contained in the more informative 'A linear 150-min gradient'", "kept_instead": "A linear 150-min gradient"},
    {"value": "20 min", "reason": "exact duplicate of '20 min'", "kept_instead": "20 min"},
    {"value": "20 min", "reason": "exact duplicate of '20 min'", "kept_instead": "20 min"}
  ]
}

-- EXAMPLE 2 -- Annotation type: FractionationMethod
Candidate values:
  1. A combination of SEC and top-down ODG
  2. a discontinuous top-down ODG

Reasoning:
  "A combination of SEC and top-down ODG" (1): Keep -- this describes the complete
    fractionation protocol: size-exclusion chromatography (SEC) followed by a
    discontinuous organelle density gradient (ODG). It is the most informative
    description of the actual method used.

  "a discontinuous top-down ODG" (2): Remove -- this describes only one component
    of the combined protocol. Everything it says is already contained within value 1.
    Keeping both would misrepresent the fractionation as two independent methods when
    in reality ODG was one step of a single combined protocol. Rule: when a value
    describes a sub-component of a combined method already fully described by another
    value, remove the sub-component.

Output:
{
  "keep": ["A combination of SEC and top-down ODG"],
  "remove": [
    {"value": "a discontinuous top-down ODG", "reason": "describes only the ODG sub-step of the combined SEC+ODG protocol already captured in 'A combination of SEC and top-down ODG'", "kept_instead": "A combination of SEC and top-down ODG"}
  ]
}

-- EXAMPLE 3 -- Annotation type: CellType
Candidate values:
  1. Mesenchymal stem cells (MSC)
  2. ESC-derived MSC
  3. Bone marrow-derived mesenchymal stem cells
  4. human embryonic stem cells (hESC)

Reasoning:
  "Mesenchymal stem cells (MSC)" (1): Keep -- canonical full name with abbreviation.

  "ESC-derived MSC" (2): Remove -- composite of values 1 and 4, both already listed.
    It adds no new biological entity; it merely restates the combination.

  "Bone marrow-derived mesenchymal stem cells" (3): Keep -- specifies bone marrow as the
    tissue origin of the MSC population, which is scientifically distinct from the
    generic MSC label in value 1.

  "human embryonic stem cells (hESC)" (4): Keep -- a different cell type (pluripotent
    vs. multipotent). Must not be merged with or removed in favour of value 1.

Output:
{
  "keep": ["Mesenchymal stem cells (MSC)", "Bone marrow-derived mesenchymal stem cells", "human embryonic stem cells (hESC)"],
  "remove": [
    {"value": "ESC-derived MSC", "reason": "composite of values already listed individually; no new biological entity introduced", "kept_instead": "Mesenchymal stem cells (MSC)"}
  ]
}

-- EXAMPLE 4 -- Annotation type: Instrument (KEEP BOTH -- genuinely distinct)
Candidate values:
  1. Orbitrap Fusion Lumos (Thermo Fisher Scientific)
  2. Q Exactive HF (Thermo Fisher Scientific)

Reasoning:
  Both are different instrument models with different hardware and acquisition
  characteristics. They are not duplicates under any rule.

Output:
{
  "keep": ["Orbitrap Fusion Lumos (Thermo Fisher Scientific)", "Q Exactive HF (Thermo Fisher Scientific)"],
  "remove": []
}

END OF DEDUP EXAMPLES.
"""

def build_dedup_prompt(atype: str, values: list[str], paper_text: str) -> str:
    MAX_PAPER_CHARS = 55_000
    truncated_text = paper_text[:MAX_PAPER_CHARS]
    numbered_values = "\n".join(f"  {i+1}. {v}" for i, v in enumerate(values))

    return f"""You are an expert proteomics and mass-spectrometry metadata curator. Your task is semantic deduplication of annotation values.

DEDUPLICATION RULES:
1. EXACT / NEAR-EXACT DUPLICATES: Remove values identical or differing only in whitespace, punctuation, or trivial capitalisation.
2. ABBREVIATION vs FULL NAME -- same entity: Keep the most complete form; remove the bare abbreviation.
3. SUBSET / COMPONENT already CONTAINED in another value: If value A contains all the information of value B and more, remove B and keep A.
4. COMPOSITE VALUES that COMBINE already-listed INDIVIDUAL values: Remove the composite.
5. GENUINELY DISTINCT values -- KEEP BOTH: Different durations, different instruments, different tissue origins are not duplicates.

DECISION GUIDELINE: Only remove when CONFIDENT the values convey the same information. When in doubt, keep both.

{ICL_DEDUP_EXAMPLES}

YOUR TASK
---------
Annotation type: "{atype}"

Candidate values:
{numbered_values}

Reason explicitly about each value before writing the JSON.

OUTPUT FORMAT (strict JSON only -- no markdown fences, no preamble):
{{
  "keep": ["value1", ...],
  "remove": [
    {{"value": "removed_value", "reason": "...", "kept_instead": "..."}}
  ]
}}

Every original value must appear in exactly one of "keep" or "remove". Do NOT invent or rephrase values.

PAPER TEXT (context only):
{truncated_text}
"""

def call_api_for_dedup(client: OpenAI, atype: str, values: list[str], paper_text: str) -> dict:
    if len(values) == 1:
        return {"keep": values, "remove": []}

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=[{"role": "user", "content": build_dedup_prompt(atype, values, paper_text)}],
        max_output_tokens=4096,
    )
    raw_text = _extract_raw_text(response)
    return _safe_parse_api_response(
        raw_text,
        required_keys=("keep", "remove"),
        original_values=set(values),
        keep_key="keep",
        remove_key="remove",
        atype=atype,
        context_label="dedup",
    )


ICL_VALIDATION_EXAMPLES = """
IN-CONTEXT LEARNING EXAMPLES FOR ANNOTATION-TYPE VALIDATION

Study the per-value reasoning carefully -- it explains the generalisable rules.

EXAMPLE 1  Annotation type: FactorValue
Definition: The value of a meaningful experimental variable that distinguishes
  sample groups. Must be a concrete, self-contained biological/clinical/experimental
  descriptor. Must NOT be a raw cell count, a bare sentence fragment, or a floating
  preposition with no referent.

Values:
  1. ~10 cells
  2. ~40-cell
  3. ~140-cell
  4. prior to
  5. 3 months after
  6. prostate cancer
  7. benign prostate hyperplasia
  8. urothelial cancer of the bladder
  9. renal cell carcinoma
  10. 1000 U/mL

Reasoning:
  "~10 cells" (1): INVALID. A raw cell count describes how many cells were used as
    input material, not what distinguishes sample groups experimentally. FactorValue
    must label an experimental condition or group identity, not a quantity.

  "~40-cell" (2): INVALID. Same reasoning -- a cell-count descriptor regardless of
    the hyphenated style.

  "~140-cell" (3): INVALID. Same -- cell count, not an experimental condition.

  "prior to" (4): INVALID. A bare preposition with no subject or object. Has no
    standalone experimental meaning. The full phrase (e.g. "prior to treatment")
    would be valid; this fragment is not.

  "3 months after" (5): INVALID. "After what?" is unknown. Not self-contained;
    the complete phrase (e.g. "3 months after surgery") would be valid.

  "prostate cancer" (6): VALID. A well-defined disease condition that distinguishes
    one patient cohort from another. Self-contained and biologically unambiguous.

  "benign prostate hyperplasia" (7): VALID. A distinct clinical condition labelling
    a patient group.

  "urothelial cancer of the bladder" (8): VALID. Specific disease diagnosis separating
    one cohort.

  "renal cell carcinoma" (9): VALID. Concrete, self-contained disease label.

  "1000 U/mL" (10): VALID. A dose/concentration value that represents the experimental
    axis defined by a compound (e.g. IFNb 1000 U/mL). A FactorValue recording only
    the dose is a legitimate companion to a Compound annotation that records the full
    compound+dose description. The dose is a self-contained experimental descriptor
    that distinguishes treated from untreated or different dose groups.

Output:
{
  "valid": ["prostate cancer", "benign prostate hyperplasia", "urothelial cancer of the bladder", "renal cell carcinoma", "1000 U/mL"],
  "invalid": [
    {"value": "~10 cells", "reason": "Raw cell count, not an experimental condition label."},
    {"value": "~40-cell", "reason": "Cell-count descriptor, not a meaningful experimental group label."},
    {"value": "~140-cell", "reason": "Cell count, not a valid experimental condition descriptor."},
    {"value": "prior to", "reason": "Incomplete sentence fragment with no standalone experimental meaning."},
    {"value": "3 months after", "reason": "Fragment missing its referent; not self-contained."}
  ]
}

EXAMPLE 2  Annotation type: DevelopmentalStage
Definition: The developmental or growth stage of the organism/organ/tissue
  at the time of sampling. DAP (days after pollination) values are valid
  because they describe the developmental age of seed/embryo tissue.
  The same DAP value may also appear under SamplingTime -- both are correct.

Values:
  1. mature pistils
  2. three DAP
  3. seven DAP
  4. seedling stage
  5. 37 degC

Reasoning:
  "mature pistils" (1): VALID. Describes the developmental maturity state of the
    pistil organ at the time of collection. This is exactly what DevelopmentalStage
    captures -- the biological stage of the material, not when it was collected.

  "three DAP" (2): VALID. Three days after pollination defines the developmental age
    of the seed at sampling. In plant biology, DAP is a standard developmental stage
    descriptor for seeds/embryos. The fact that "three DAP" also appears under
    SamplingTime (when the sample was collected) does not invalidate it here -- dual
    annotation is correct and expected because the same fact describes both stage and time.

  "seven DAP" (3): VALID. Same reasoning as "three DAP" -- a distinct developmental
    stage (seven days after pollination) that must be kept as a separate entry because
    it represents a different biological state from three DAP.

  "seedling stage" (4): VALID. A named plant developmental stage.

  "37 degC" (5): INVALID. A temperature, not a developmental stage. Belongs to
    Temperature. Its presence under DevelopmentalStage is a clear category error.

Output:
{
  "valid": ["mature pistils", "three DAP", "seven DAP", "seedling stage"],
  "invalid": [
    {"value": "37 degC", "reason": "A temperature value, not a developmental stage. Belongs to Temperature."}
  ]
}

EXAMPLE 3  Annotation type: SamplingTime
Definition: The time point at which a sample was collected. DAP values are
  valid SamplingTime when they anchor when the sample was taken.
  INVALID: any value that mixes a non-temporal descriptor (size, morphology)
  into the time expression.

Values:
  1. three and seven days after pollination
  2. three DAP
  3. seven DAP
  4. three DAP (around 6 mm in length)
  5. seven DAP (around 10 mm in length)
  6. T0

Reasoning:
  "three and seven days after pollination" (1): VALID. A pure temporal description
    of two collection time points. Self-contained and unambiguous as a SamplingTime
    value. It covers both collection points in a single entry, which is acceptable.

  "three DAP" (2): VALID. A specific time point (three days after pollination) at
    which the sample was harvested. Pure temporal information -- no physical descriptor
    mixed in. Valid SamplingTime even though it is also a DevelopmentalStage value.

  "seven DAP" (3): VALID. Same reasoning as "three DAP".

  "three DAP (around 6 mm in length)" (4): INVALID. This mixes a physical size
    descriptor ("around 6 mm in length") into a time annotation. Size is a morphological
    characteristic, not a time point. The parenthetical addition makes this a hybrid
    description that does not fit SamplingTime, which must be a pure temporal anchor.
    The temporal part ("three DAP") is already captured by value 2. This value belongs
    to DevelopmentalStage or OrganismPart where size context is appropriate.

  "seven DAP (around 10 mm in length)" (5): INVALID. Same reasoning as value 4 --
    a size descriptor pollutes the time annotation. The temporal part is already
    captured by value 3.

  "T0" (6): VALID. A canonical time-course anchor point.

Output:
{
  "valid": ["three and seven days after pollination", "three DAP", "seven DAP", "T0"],
  "invalid": [
    {"value": "three DAP (around 6 mm in length)", "reason": "Mixes a physical size descriptor into a time annotation. Pure temporal value 'three DAP' is already present. Size context belongs to DevelopmentalStage or OrganismPart."},
    {"value": "seven DAP (around 10 mm in length)", "reason": "Mixes a physical size descriptor into a time annotation. Pure temporal value 'seven DAP' is already present. Size context belongs to DevelopmentalStage or OrganismPart."}
  ]
}

EXAMPLE 4  Annotation type: Time
Definition: A broad time parameter for the experiment that does not fit the
  more specific SamplingTime or DevelopmentalStage categories.
  If a time value is already well-annotated under DevelopmentalStage or
  SamplingTime, its bare repetition under Time is redundant.

Values:
  1. three DAP
  2. seven DAP
  3. 72 h post-infection
  4. day 5

Reasoning:
  "three DAP" (1): INVALID here. In this experiment "three DAP" is already correctly
    annotated under DevelopmentalStage (describing the seed developmental age) and
    under SamplingTime (describing the collection time point). Adding a third annotation
    of the same value under Time creates redundancy without adding information. Time
    should be used only when the time expression does not fit the more specific types.
    Remove from Time.

  "seven DAP" (2): INVALID here. Same reasoning as "three DAP" -- already covered
    by DevelopmentalStage and SamplingTime.

  "72 h post-infection" (3): VALID. A time expression that is specific to a treatment
    time-course, not a developmental stage or a standalone sampling time point. This
    belongs under Time as a broad time parameter.

  "day 5" (4): VALID. A broad time-course descriptor that is not specifically a
    developmental stage or a sample-collection time point in context.

Output:
{
  "valid": ["72 h post-infection", "day 5"],
  "invalid": [
    {"value": "three DAP", "reason": "Already annotated under DevelopmentalStage and SamplingTime with clear biological meaning. Bare repetition under Time is redundant."},
    {"value": "seven DAP", "reason": "Already annotated under DevelopmentalStage and SamplingTime with clear biological meaning. Bare repetition under Time is redundant."}
  ]
}

EXAMPLE 5  Annotation type: GradientTime
Definition: The total duration of the LC gradient. Any value from which a
  specific duration can be extracted is valid -- whether expressed as a bare
  number, a phrase, or with gradient composition context. Do NOT reject a
  value just because it is phrased as a sentence fragment rather than a
  bare number.

Values:
  1. over a 60-min gradient
  2. over 180 min
  3. Standard 90-min gradients
  4. from 5% B to 15% B over 60 min
  5. gradient from 5 to 40% acetonitrile

Reasoning:
  "over a 60-min gradient" (1): VALID. Contains an explicit duration: 60 min. The
    prepositional phrasing ("over a ... gradient") does not disqualify it. A curator
    reading this immediately extracts "60 min" as the gradient time. Do NOT reject
    because of the prose form -- the duration is unambiguous.

  "over 180 min" (2): VALID. Explicit duration: 180 min.

  "Standard 90-min gradients" (3): VALID. Explicit duration: 90 min. The word
    "Standard" is a descriptor of the gradient type, not a barrier to extracting
    the duration.

  "from 5% B to 15% B over 60 min" (4): VALID. Explicit duration: 60 min. The
    gradient composition context (5% to 15% B) is additional information, not a
    reason to reject the time. Both the composition and the duration are useful.

  "gradient from 5 to 40% acetonitrile" (5): INVALID. Describes gradient
    composition only -- no duration in any time unit is present. Without a duration
    this cannot be a GradientTime value. Belongs to a methods description field,
    not GradientTime.

Output:
{
  "valid": ["over a 60-min gradient", "over 180 min", "Standard 90-min gradients", "from 5% B to 15% B over 60 min"],
  "invalid": [
    {"value": "gradient from 5 to 40% acetonitrile", "reason": "Describes gradient composition only; no duration in time units is present. GradientTime requires an extractable duration."}
  ]
}

EXAMPLE 6  Annotation type: Depletion
Definition: Method or condition used to remove or reduce a specific component.
  Includes nutrient-deprivation conditions. The key signal is ABSENCE or REMOVAL.
  Nutrient-RICH or nutrient-REPLETE controls are NOT depletions.

Values:
  1. S-depleted
  2. S-rich
  3. albumin depletion (MARS column)
  4. phospho-enrichment

Reasoning:
  "S-depleted" (1): VALID. "S-depleted" means sulfur-depleted growth medium --
    sulfur has been removed from the nutrient supply. The word "depleted" is the
    direct signal that something is being withheld. This is exactly what Depletion
    captures. Even though it is a growth condition rather than a protein-cleanup
    method, nutrient depletion is a legitimate Depletion annotation.

  "S-rich" (2): INVALID. "S-rich" means sulfur-replete medium -- sulfur is present
    in abundance. Nothing is being removed or depleted. This is the control condition
    opposite to S-depleted. It belongs to Treatment or FactorValue (as the contrasting
    experimental condition), NOT to Depletion. Accepting it under Depletion would
    invert the meaning of the annotation type.

  "albumin depletion (MARS column)" (3): VALID. A protein-depletion step using a
    MARS affinity column to remove high-abundance albumin from plasma samples. The
    canonical Depletion use case.

  "phospho-enrichment" (4): INVALID. Enrichment is the opposite of depletion --
    this step concentrates phosphopeptides rather than removing something. Belongs
    to EnrichmentMethod.

Output:
{
  "valid": ["S-depleted", "albumin depletion (MARS column)"],
  "invalid": [
    {"value": "S-rich", "reason": "Describes a nutrient-replete condition (sulfur-rich medium), not a depletion. The control opposite of S-depleted. Belongs to Treatment or FactorValue."},
    {"value": "phospho-enrichment", "reason": "An enrichment step, not a depletion. Belongs to EnrichmentMethod."}
  ]
}

EXAMPLE 7  Annotation type: FractionIdentifier
Definition: A numeric or text identifier for a specific fraction. Capitalisation
  differences alone do not create distinct identifiers.

Values:
  1. fractions 4-7
  2. Fractions 9 and 10
  3. pooled (1-5, 6-8, 9-10)
  4. F1

Reasoning:
  "fractions 4-7" (1): VALID. Identifies fractions 4, 5, 6, and 7 by number.
    Lowercase "fractions" is standard and does not disqualify the value.

  "Fractions 9 and 10" (2): VALID. Identifies fractions 9 and 10 by number.
    Capitalised "Fractions" is equally valid. This is a distinct identifier from
    value 1 (different fraction numbers) and must be kept as a separate entry.

  "pooled (1-5, 6-8, 9-10)" (3): INVALID. Describes how multiple fractions were
    pooled together -- a merged-sample description, not a single fraction identifier.
    Belongs to PooledSample.

  "F1" (4): VALID. A compact fraction identifier in standard notation.

Output:
{
  "valid": ["fractions 4-7", "Fractions 9 and 10", "F1"],
  "invalid": [
    {"value": "pooled (1-5, 6-8, 9-10)", "reason": "Describes a pooling scheme across fraction ranges, not a fraction identifier. Belongs to PooledSample."}
  ]
}

EXAMPLE 8  Annotation type: Compound
Definition: A chemical or small molecule added as a perturbation/stimulus.
  Full compound descriptions including dose are valid. A companion FactorValue
  recording only the dose does NOT conflict with Compound -- both coexist.

Values:
  1. human recombinant IFNb 1000 U/mL
  2. rapamycin 10 nM
  3. trypsin

Reasoning:
  "human recombinant IFNb 1000 U/mL" (1): VALID. A specific biological compound
    (interferon beta) with concentration. Full compound+dose descriptions are the
    ideal Compound annotation. Note: a companion FactorValue: "1000 U/mL" recording
    only the dose is a legitimate separate annotation -- it does NOT make this Compound
    entry invalid or redundant. Both capture different levels of information and should
    coexist.

  "rapamycin 10 nM" (2): VALID. A specific small-molecule inhibitor with dose.

  "trypsin" (3): INVALID. Trypsin is the protease used for protein digestion, not a
    compound added as a biological perturbation. Belongs to CleavageAgent.

Output:
{
  "valid": ["human recombinant IFNb 1000 U/mL", "rapamycin 10 nM"],
  "invalid": [
    {"value": "trypsin", "reason": "A protease used for sample preparation, not a biological stimulus. Belongs to CleavageAgent."}
  ]
}

EXAMPLE 9  Annotation type: Separation
Definition: Any on-line chromatographic column or method used to separate
  peptides immediately before MS. Column names, stationary-phase descriptions,
  and general method labels are all valid.

Values:
  1. ReproSil-Pur 120 C18-AQ 3 um diameter beads
  2. EASY-Spray C18 column (50 cm x 75 um ID, 2 um particle size)
  3. reversed-phase nanoLC
  4. Strong Cation Exchange (SCX) off-line fractionation

Reasoning:
  "ReproSil-Pur 120 C18-AQ 3 um diameter beads" (1): VALID. The stationary-phase
    material packed into an in-house emitter column for on-line separation. For
    in-house packed columns the bead description IS the column description.

  "EASY-Spray C18 column (50 cm x 75 um ID, 2 um particle size)" (2): VALID. A
    commercial on-line analytical column with full specifications.

  "reversed-phase nanoLC" (3): VALID. A general method label for the on-line
    separation mode.

  "Strong Cation Exchange (SCX) off-line fractionation" (4): INVALID. Off-line
    pre-fractionation performed before the LC-MS run. Belongs to FractionationMethod.

Output:
{
  "valid": ["ReproSil-Pur 120 C18-AQ 3 um diameter beads", "EASY-Spray C18 column (50 cm x 75 um ID, 2 um particle size)", "reversed-phase nanoLC"],
  "invalid": [
    {"value": "Strong Cation Exchange (SCX) off-line fractionation", "reason": "Off-line pre-fractionation step. Belongs to FractionationMethod."}
  ]
}

EXAMPLE 10  Annotation type: NumberOfMissedCleavages
Definition: Max missed cleavages allowed in database search. May be a digit,
  written-out number, or descriptive phrase. Do NOT reject for non-integer form.

Values:
  1. 2
  2. two
  3. maximum two missed cleavages
  4. up to 2 missed cleavages allowed
  5. 120 min gradient

Reasoning:
  "2" (1): VALID. Canonical bare-digit form.

  "two" (2): VALID. Written-out number meaning 2. Rejecting this while accepting "2"
    would be a formatting preference, not a semantic judgment. The information content
    is identical.

  "maximum two missed cleavages" (3): VALID. A descriptive phrase that unambiguously
    states the setting. Verbatim extractions from methods sections often look like this.

  "up to 2 missed cleavages allowed" (4): VALID. Same reasoning -- clear numeric
    content in natural language.

  "120 min gradient" (5): INVALID. A gradient duration, not a missed-cleavage count.
    Conveys no missed-cleavage information. Belongs to GradientTime.

Output:
{
  "valid": ["2", "two", "maximum two missed cleavages", "up to 2 missed cleavages allowed"],
  "invalid": [
    {"value": "120 min gradient", "reason": "A gradient duration, not a missed-cleavage setting. Belongs to GradientTime."}
  ]
}

EXAMPLE 11  Annotation type: Strain
Definition: The genetic strain of a NON-HUMAN source organism. Human or
  primate cell line identifiers belong to CellLine, not Strain.

Values:
  1. BALB/c
  2. C57BL/6J
  3. RH
  4. ES04
  5. H9

Reasoning:
  "BALB/c" (1): VALID. A well-known inbred mouse strain.

  "C57BL/6J" (2): VALID. Another standard inbred mouse strain.

  "RH" (3): VALID. The RH strain of Toxoplasma gondii -- a named parasite genetic line.

  "ES04" (4): INVALID. Registry identifier for a human embryonic stem cell line
    (hESC, WiCell). Source organism is Homo sapiens. Human cell line identifiers
    belong to CellLine, not Strain. The companion CellLine annotation in the same
    file confirms this.

  "H9" (5): INVALID. Human embryonic stem cell line (WA09/H9, WiCell). Same reasoning.

Output:
{
  "valid": ["BALB/c", "C57BL/6J", "RH"],
  "invalid": [
    {"value": "ES04", "reason": "Human embryonic stem cell line identifier. Belongs to CellLine, not Strain."},
    {"value": "H9", "reason": "Human embryonic stem cell line identifier. Belongs to CellLine, not Strain."}
  ]
}

EXAMPLE 12  Annotation type: Label
Definition: Labelling/quantification strategy. LFQ is valid.

Values:
  1. TMT-126
  2. SILAC heavy
  3. label-free quantification (LFQ)
  4. trypsin

Reasoning:
  "TMT-126" (1): VALID. An isobaric label channel.
  "SILAC heavy" (2): VALID. A stable-isotope metabolic label.
  "label-free quantification (LFQ)" (3): VALID. LFQ is the labelling strategy
    (absence of exogenous label). Explicitly included in the definition.
  "trypsin" (4): INVALID. A protease, not a labelling strategy. Belongs to CleavageAgent.

Output:
{
  "valid": ["TMT-126", "SILAC heavy", "label-free quantification (LFQ)"],
  "invalid": [
    {"value": "trypsin", "reason": "A protease/cleavage agent, not a labelling strategy. Belongs to CleavageAgent."}
  ]
}

END OF VALIDATION EXAMPLES.
"""

def build_validation_prompt(atype: str, values: list[str], paper_text: str) -> str:
    MAX_PAPER_CHARS = 55_000
    truncated_text = paper_text[:MAX_PAPER_CHARS]
    numbered_values = "\n".join(f"  {i+1}. {v}" for i, v in enumerate(values))
    definition = ANNOTATION_TYPE_DEFINITIONS.get(atype, "No definition available.")

    return f"""You are an expert proteomics and mass-spectrometry metadata curator.
Your task is to validate whether each annotation value genuinely belongs to its declared annotation type.

ANNOTATION TYPE: "{atype}"
DEFINITION: {definition}

VALIDATION RULES:
1. SEMANTIC MATCH: The value must fall within the definition of the annotation type.
2. CATEGORY ERRORS: Reject values that clearly belong to a different annotation type.
3. INCOMPLETE / MEANINGLESS STRINGS: Reject bare sentence fragments or prepositions with
   no standalone experimental meaning.
4. NATURAL LANGUAGE IS ALLOWED: Do NOT reject values merely because they are phrased as a
   sentence or phrase rather than a bare number or keyword. If the required information can
   be extracted (e.g. a duration from a gradient-time phrase), the value is valid.
5. BORDERLINE / AMBIGUOUS CASES -- KEEP: Only reject when CONFIDENT it is wrong.
6. USE PAPER CONTEXT to resolve ambiguous cases.

{ICL_VALIDATION_EXAMPLES}

YOUR TASK
---------
Annotation type: "{atype}"

Candidate values:
{numbered_values}

Reason explicitly about each value before writing the JSON.

OUTPUT FORMAT (strict JSON only -- no markdown fences, no preamble):
{{
  "valid": ["value1", ...],
  "invalid": [
    {{"value": "invalid_value", "reason": "why it does not match the type, and which type it likely belongs to if applicable"}}
  ]
}}

Every original value must appear in exactly one of "valid" or "invalid".
Do NOT invent, rephrase, or omit any value.

PAPER TEXT (context only):
{truncated_text}
"""

def call_api_for_validation(client: OpenAI, atype: str, values: list[str], paper_text: str) -> dict:
    if not values:
        return {"valid": [], "invalid": []}

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=[{"role": "user", "content": build_validation_prompt(atype, values, paper_text)}],
        max_output_tokens=4096,
    )
    raw_text = _extract_raw_text(response)
    return _safe_parse_api_response(
        raw_text,
        required_keys=("valid", "invalid"),
        original_values=set(values),
        keep_key="valid",
        remove_key="invalid",
        atype=atype,
        context_label="validation",
    )


ICL_DISAMBIGUATION_EXAMPLES = """
IN-CONTEXT LEARNING EXAMPLES FOR CROSS-ANNOTATION DISAMBIGUATION

Background rule:
  The same string value should appear under AT MOST ONE annotation type,
  with one explicit exception: FactorValue is ALWAYS allowed to mirror
  values from other types (it summarises the experimental axes of the study).
  Never remove a value from FactorValue.

  For every other conflict (same value under two or more non-FactorValue types),
  exactly one assignment is correct and the other is an extraction error.
  The paper text is the ground truth.

  ADDITIONAL RULE -- Compound + FactorValue dose split:
  When a Compound annotation contains the full compound+dose description
  (e.g. "human recombinant IFNb 1000 U/mL") AND a FactorValue annotation
  contains only the dose portion of that description (e.g. "1000 U/mL"),
  these are NOT a conflict. The dose string in FactorValue is a partial
  substring, not the same string. No action is needed.

EXAMPLE 1  Conflict: SpikedCompound vs Treatment
Conflicting value: "biotin"
  SpikedCompound: biotin
  Treatment:      biotin

Paper context (excerpt):
  "Cells expressing FLAG-BirA* were grown in media supplemented with 50 uM
   biotin for 24 h to allow biotinylation of proximal proteins."

Reasoning:
  Biotin is added "to allow biotinylation" -- it is the technical substrate for
  the BioID assay, not a biological perturbation. SpikedCompound = assay reagent;
  Treatment = deliberate biological perturbation. Remove from Treatment.

Output:
{
  "decisions": [
    {
      "value": "biotin",
      "keep_in": "SpikedCompound",
      "remove_from": "Treatment",
      "reason": "Biotin is the BioID proximity-labelling substrate (assay reagent), not a biological perturbation."
    }
  ]
}

EXAMPLE 2  Conflict: Strain vs Bait vs GeneticModification
Conflicting values (Toxoplasma BioID study):
  Strain: FLAG-BirA*
  Bait:   FLAG-BirA*

Paper context:
  "MOB1-GFP, FLAG-BirA*, FLAG-BirA*-MOB1, and FLAG-MOB1 strains were obtained
   by transfection of RH tachyzoites and DNA construct random integration."

Reasoning for "FLAG-BirA*":
  The paper calls it a "strain". In BioID the bait is the protein of interest
  fused to BirA*; the unfused enzyme is the negative-control, not a bait.
  Remove from Bait; keep in Strain.

Reasoning for "FLAG-BirA*-MOB1":
  Both a stable transgenic line (Strain) and the BioID bait (Bait). Keep in both.
  If listed under GeneticModification, that is a category error -- the strain name
  is not a description of the modification event. Remove from GeneticModification.

Output:
{
  "decisions": [
    {
      "value": "FLAG-BirA*",
      "keep_in": "Strain",
      "remove_from": "Bait",
      "reason": "Unfused BirA* is the negative-control strain, not a bait protein."
    },
    {
      "value": "FLAG-BirA*-MOB1",
      "keep_in": ["Strain", "Bait"],
      "remove_from": "GeneticModification",
      "reason": "Correctly a Strain and a Bait; listing the strain name under GeneticModification is a category error."
    }
  ]
}

EXAMPLE 3  FactorValue mirroring Disease -- NO ACTION
Conflicting value: "breast cancer"
  Disease:     breast cancer
  FactorValue: breast cancer

Reasoning:
  FactorValue is permitted to mirror values from other types. No action needed.

Output:
{
  "decisions": []
}

EXAMPLE 4  Compound + FactorValue dose split -- NO ACTION
Conflicting annotations:
  Compound:    human recombinant IFNb 1000 U/mL
  FactorValue: 1000 U/mL

Reasoning:
  "human recombinant IFNb 1000 U/mL" and "1000 U/mL" are NOT the same string.
  The Compound annotation records the full compound identity with dose.
  The FactorValue records only the dose as the experimental axis variable.
  This is a legitimate split: one annotation describes what was added,
  the other describes the experimental variable that distinguishes groups.
  FactorValue is always exempt from conflict resolution. No action needed.

Output:
{
  "decisions": []
}

EXAMPLE 5  Conflict: Label vs Treatment
Conflicting value: "label-free quantification (LFQ)"
  Label:     label-free quantification (LFQ)
  Treatment: label-free quantification (LFQ)

Reasoning:
  LFQ is a quantification/labelling strategy, not a biological perturbation.
  Its presence under Treatment is a category error.

Output:
{
  "decisions": [
    {
      "value": "label-free quantification (LFQ)",
      "keep_in": "Label",
      "remove_from": "Treatment",
      "reason": "LFQ is a quantification strategy, not an experimental treatment."
    }
  ]
}

EXAMPLE 6  Conflict: OrganismPart vs CellType
Conflicting value: "tachyzoites"
  OrganismPart: tachyzoites
  CellType:     tachyzoites

Paper context: "Toxoplasma gondii RH tachyzoites were harvested for proteomics."

Reasoning:
  Tachyzoites are a parasite life-stage form (source material) -- correctly OrganismPart.
  They are not a mammalian cell type or lineage. Remove from CellType.

Output:
{
  "decisions": [
    {
      "value": "tachyzoites",
      "keep_in": "OrganismPart",
      "remove_from": "CellType",
      "reason": "Parasite life-stage form (source material); not a mammalian cell type."
    }
  ]
}

EXAMPLE 7  Conflict: CellLine vs Strain
Conflicting value: "ES04"
  CellLine: human embryonic stem cell line (hESC) ES04 (WiCell institute)
  Strain:   ES04

Reasoning:
  ES04 is a human cell line identifier. Human cell line IDs belong to CellLine.
  Strain is reserved for non-human organism lines. Remove from Strain.

Output:
{
  "decisions": [
    {
      "value": "ES04",
      "keep_in": "CellLine",
      "remove_from": "Strain",
      "reason": "Human embryonic stem cell line identifier. Belongs to CellLine. Strain is for non-human organism lines only."
    }
  ]
}

END OF DISAMBIGUATION EXAMPLES.
"""

def build_disambiguation_prompt(
    conflicts: list[dict],
    all_annotations: dict[str, list[str]],
    paper_text: str,
) -> str:
    MAX_PAPER_CHARS = 55_000
    truncated_text = paper_text[:MAX_PAPER_CHARS]

    all_ann_lines = []
    for atype, vals in sorted(all_annotations.items()):
        for v in vals:
            all_ann_lines.append(f"  {atype}: {v}")
    all_ann_block = "\n".join(all_ann_lines) if all_ann_lines else "  (none)"

    conflict_lines = []
    for i, c in enumerate(conflicts, 1):
        types_str = ", ".join(c["types"])
        conflict_lines.append(f'  {i}. Value: "{c["value"]}"  ->  appears in: {types_str}')
    conflict_block = "\n".join(conflict_lines)

    involved_types = sorted({t for c in conflicts for t in c["types"]})
    def_lines = [
        f"  {t}: {ANNOTATION_TYPE_DEFINITIONS.get(t, 'No definition available.')}"
        for t in involved_types
    ]
    definitions_block = "\n".join(def_lines)

    return f"""You are an expert proteomics and mass-spectrometry metadata curator.

TASK: Cross-annotation disambiguation.

The same value has been assigned to more than one annotation type (excluding FactorValue,
which is permitted to mirror other types). For each conflict, use the paper text and the
annotation type definitions to decide which assignment is correct and which is an error.

DISAMBIGUATION RULES:
1. EXACTLY ONE CORRECT TYPE unless the value legitimately belongs to multiple simultaneously.
2. FACTORVALUE IS ALWAYS ALLOWED TO SHARE. Never remove a value from FactorValue.
3. COMPOUND + FACTORVALUE DOSE SPLIT: if Compound has the full "compound + dose" string
   and FactorValue has only the dose portion, these are not the same string -- no conflict.
4. USE PAPER TEXT AS GROUND TRUTH.
5. WHEN IN DOUBT -- KEEP THE MORE SPECIFIC TYPE.
6. REASON FIRST, then output JSON.

{ICL_DISAMBIGUATION_EXAMPLES}

YOUR TASK
---------

CONFLICTING VALUES:
{conflict_block}

ANNOTATION TYPE DEFINITIONS (involved types):
{definitions_block}

ALL CURRENT ANNOTATIONS (for context):
{all_ann_block}

Reason about each conflict before writing the JSON.

OUTPUT FORMAT (strict JSON only -- no markdown fences, no preamble):
{{
  "decisions": [
    {{
      "value": "the conflicting value",
      "keep_in": "TypeName"  OR  ["TypeName1", "TypeName2"],
      "remove_from": "TypeName"  OR  ["TypeName1", "TypeName2"],
      "reason": "concise explanation grounded in the paper text and type definitions"
    }},
    ...
  ]
}}

Omit a conflict from decisions entirely if no removal is needed.

PAPER TEXT:
{truncated_text}
"""

def detect_cross_type_conflicts(validated_values: dict[str, list[str]]) -> list[dict]:
    norm_to_types: dict[str, set[str]] = {}
    norm_to_raw: dict[str, str] = {}

    for atype, values in validated_values.items():
        for v in values:
            norm = v.strip().lower()
            norm_to_types.setdefault(norm, set()).add(atype)
            norm_to_raw.setdefault(norm, v)

    conflicts = []
    for norm, types in norm_to_types.items():
        if len(types) <= 1:
            continue
        non_fv = types - ALLOW_CROSS_TYPE_SHARING_WITH
        if len(non_fv) <= 1:
            continue
        conflicts.append({"value": norm_to_raw[norm], "types": sorted(types)})

    return conflicts

def call_api_for_disambiguation(
    client: OpenAI,
    conflicts: list[dict],
    all_annotations: dict[str, list[str]],
    paper_text: str,
) -> list[dict]:
    if not conflicts:
        return []

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=[{
            "role": "user",
            "content": build_disambiguation_prompt(conflicts, all_annotations, paper_text),
        }],
        max_output_tokens=4096,
    )
    raw_text = _extract_raw_text(response)

    try:
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        result = json.loads(clean)
        decisions = result.get("decisions", [])
        for d in decisions:
            if isinstance(d.get("keep_in"), str):
                d["keep_in"] = [d["keep_in"]]
            if isinstance(d.get("remove_from"), str):
                d["remove_from"] = [d["remove_from"]]
        return decisions
    except Exception as e:
        print(f"    [WARN] [disambiguation] Could not parse response: {e}", file=sys.stderr)
        print(f"    Raw response: {raw_text[:500]}", file=sys.stderr)
        return []

def apply_disambiguation_decisions(
    validated_values: dict[str, list[str]],
    decisions: list[dict],
) -> tuple[dict[str, list[str]], list[dict]]:
    updated = {atype: list(vals) for atype, vals in validated_values.items()}
    removal_records = []

    for decision in decisions:
        raw_value = decision.get("value", "")
        remove_from_types = decision.get("remove_from", [])
        keep_in_types = decision.get("keep_in", [])
        reason = decision.get("reason", "cross-annotation conflict resolved")

        if isinstance(remove_from_types, str):
            remove_from_types = [remove_from_types]
        if isinstance(keep_in_types, str):
            keep_in_types = [keep_in_types]

        for atype in remove_from_types:
            if atype not in updated:
                continue
            new_list = [
                v for v in updated[atype]
                if v.strip().lower() != raw_value.strip().lower()
            ]
            if len(new_list) < len(updated[atype]):
                kept_instead = ", ".join(f"{t}: {raw_value}" for t in keep_in_types)
                print(f"    DISAMBIGUATION REMOVE: '{raw_value}' from {atype}")
                print(f"      reason      : {reason}")
                print(f"      kept instead: {kept_instead}")
                removal_records.append({
                    "annotation_type": atype,
                    "removed_value": raw_value,
                    "kept_instead": kept_instead,
                    "reason": reason,
                    "pass": "disambiguation",
                })
                updated[atype] = new_list

    return updated, removal_records

def process_ann_file(ann_path: str, paper_text: str, client: OpenAI, output_dir: str, csv_writer):
    base = Path(ann_path).stem
    pmc_match = re.search(r'(PMC\d+)', base)
    stem = pmc_match.group(1) if pmc_match else base

    print(f"Processing: {ann_path}  (stem={stem})")

    entries = read_ann_file(ann_path)
    groups = group_by_type(entries)

    kept_values: dict[str, list[str]] = {}
    removed_records: list[dict] = []

    for atype, values in groups.items():
        print(f"  [{atype}] {len(values)} raw value(s):")
        for v in values:
            print(f"    * {v}")

        deduped, exact_removed = remove_exact_duplicates(values)
        for removed_val, kept_val in exact_removed:
            print(f"    EXACT DUPLICATE removed: '{removed_val}' (kept: '{kept_val}')")
            removed_records.append({
                "file": Path(ann_path).name,
                "stem": stem,
                "annotation_type": atype,
                "removed_value": removed_val,
                "kept_instead": kept_val,
                "reason": "exact duplicate",
                "pass": "deduplication",
            })

        if len(deduped) == 1:
            kept_values[atype] = deduped
            continue

        print(f"    -> {len(deduped)} values after exact dedup, calling API for semantic dedup...")
        result = call_api_for_dedup(client, atype, deduped, paper_text)
        kept_values[atype] = result.get("keep", deduped)

        for removal in result.get("remove", []):
            val = removal.get("value", "")
            reason = removal.get("reason", "duplicate")
            kept_instead = removal.get("kept_instead", "")
            print(f"    SEMANTIC REMOVE: '{val}'")
            print(f"      reason      : {reason}")
            print(f"      kept instead: '{kept_instead}'")
            removed_records.append({
                "file": Path(ann_path).name,
                "stem": stem,
                "annotation_type": atype,
                "removed_value": val,
                "kept_instead": kept_instead,
                "reason": reason,
                "pass": "deduplication",
            })

    print(f"  -> Running type-validity validation pass...")
    validated_values: dict[str, list[str]] = {}

    for atype, values in kept_values.items():
        if not values:
            validated_values[atype] = []
            continue

        print(f"  [VALIDATE {atype}] checking {len(values)} value(s)...")
        val_result = call_api_for_validation(client, atype, values, paper_text)
        validated_values[atype] = val_result.get("valid", values)

        for inv in val_result.get("invalid", []):
            val = inv.get("value", "")
            reason = inv.get("reason", "does not match annotation type")
            print(f"    VALIDATION REMOVE: '{val}'")
            print(f"      reason: {reason}")
            removed_records.append({
                "file": Path(ann_path).name,
                "stem": stem,
                "annotation_type": atype,
                "removed_value": val,
                "kept_instead": "",
                "reason": reason,
                "pass": "validation",
            })

    print(f"  -> Running cross-annotation disambiguation pass...")
    conflicts = detect_cross_type_conflicts(validated_values)

    if conflicts:
        print(f"  Found {len(conflicts)} cross-type conflict(s):")
        for c in conflicts:
            print(f"    * \"{c['value']}\" in: {', '.join(c['types'])}")

        decisions = call_api_for_disambiguation(client, conflicts, validated_values, paper_text)
        final_values, disambiguation_removals = apply_disambiguation_decisions(
            validated_values, decisions
        )

        for rec in disambiguation_removals:
            removed_records.append({
                "file": Path(ann_path).name,
                "stem": stem,
                **rec,
            })
    else:
        print(f"  No cross-type conflicts found.")
        final_values = validated_values

    seen_types_order: list[str] = []
    for atype, _ in entries:
        if atype not in seen_types_order:
            seen_types_order.append(atype)

    out_path = os.path.join(output_dir, f"{stem}_filtered.ann")
    with open(out_path, "w", encoding="utf-8") as fh:
        for atype in seen_types_order:
            for val in final_values.get(atype, []):
                fh.write(f"{atype}: {val}\n")

    print(f"  [OK] Filtered .ann written : {out_path}")
    print(f"  [OK] Removed {len(removed_records)} annotation(s) total from this file")

    for rec in removed_records:
        csv_writer.writerow(rec)

    return len(removed_records)

def main():
    ann_dir = Path(ANN_DIR)
    if not ann_dir.is_dir():
        print(f"[ERROR] ANN_DIR not found: {ann_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    ann_files = sorted(ann_dir.glob("*.ann"))
    if not ann_files:
        print(f"[ERROR] No .ann files found in: {ann_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ann_files)} .ann file(s) in: {ann_dir.resolve()}")

    json_path = Path(JSON_PATH)
    if not json_path.is_file():
        print(f"[ERROR] JSON file not found: {json_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading paper dataset from: {json_path.resolve()}")
    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)
    print(f"  {len(json_data)} papers loaded.")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    csv_fields = ["file", "stem", "annotation_type", "removed_value", "kept_instead", "reason", "pass"]
    csv_path = Path(CSV_OUT)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writeheader()

        total_removed = 0
        for ann_path in ann_files:
            base = ann_path.stem
            pmc_match = re.search(r'(PMC\d+)', base)
            stem = pmc_match.group(1) if pmc_match else base

            paper_text = read_paper_text(json_data, stem)
            if not paper_text:
                print(f"  [WARN] No paper text found for stem '{stem}' -- proceeding without context.")

            total_removed += process_ann_file(str(ann_path), paper_text, client, str(output_dir), writer)

    print(f"DONE. Total annotations removed: {total_removed}")
    print(f"Filtered .ann files : {output_dir.resolve()}/")
    print(f"Removal report CSV  : {csv_path.resolve()}")

if __name__ == "__main__":
    main()