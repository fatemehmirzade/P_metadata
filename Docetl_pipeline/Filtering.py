import os
import sys
import json
import csv
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

MODEL = "google/gemma-4-31b-it"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
    "CellPart": (
        "Subcellular compartment or fraction INSIDE or part of the cell membrane "
        "(e.g. 'nucleus', 'mitochondria', 'cytoplasm', 'plasma membrane'). "
        "Must NOT be an extracellular or secreted structure such as 'extracellular vesicles' or 'exosomes' -- "
        "those belong to OrganismPart or Specimen."
    ),
    "CellType": "Primary cell type or lineage (e.g. 'neurons', 'fibroblasts').",
    "CleavageAgent": "Protease or chemical used to digest proteins (e.g. 'trypsin', 'chymotrypsin'). Grade modifiers like 'MS-grade' or 'sequencing-grade' do not create a distinct entry from the plain enzyme name.",
    "CollisionEnergy": "Collision energy applied in MS/MS (e.g. '27 eV'). '42.0 eV' and '42 eV' are the same value.",
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
    "EnrichmentMethod": (
        "A specific peptide or protein enrichment protocol used to concentrate a class of analytes "
        "prior to MS (e.g. 'TiO2 phosphopeptide enrichment', 'IMAC', 'streptavidin bead enrichment for biotinylated peptides'). "
        "Must NOT be an overall experimental workflow or protein-capture approach: "
        "immunoprecipitation (IP), IP-MS, pull-down, co-IP, BioID, and proximity-labeling are experimental "
        "designs, not enrichment methods, and do NOT belong here. "
        "However, 'anti-GFP affinity purification' or 'streptavidin pull-down of biotinylated proteins' "
        "can be valid enrichment steps if they are specifically used to enrich a protein class for MS."
    ),
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
    "FragmentationMethod": "Ion-fragmentation technique (e.g. 'HCD', 'CID', 'ETD'). The abbreviation and full name of the same technique are duplicates; keep the full name.",
    "FragmentMassTolerance": (
        "Mass tolerance for fragment ion matching in database search "
        "(e.g. '0.02 Da', '20 ppm', '20, 10, and 7 ppm'). "
        "A value listing multiple tolerances for different acquisition modes is valid as a single entry."
    ),
    "FractionationMethod": (
        "Any off-line method used to fractionate the bulk sample into primary samples used in MS. "
        "When multiple fractionation methods are combined in one experiment, the combined description "
        "is the correct single annotation. A sub-step or component method already fully described by "
        "a more complete combined annotation should be removed as redundant. "
        "A sucrose gradient used for organelle/ribosome sedimentation as a sample preparation step "
        "is a valid FractionationMethod only if it directly produces the fractions analyzed by MS."
    ),
    "FractionIdentifier": (
        "A numeric or text identifier for a specific chromatographic or offline fraction. "
        "Valid values identify one or more specific fractions by number or name "
        "(e.g. 'F1', 'fractions 4-7', 'Fractions 9 and 10'). "
        "Capitalisation differences do not create distinct fractions -- treat them as the same identifier. "
        "Must NOT be a description of pooled samples (those belong to PooledSample)."
    ),
    "GeneticModification": "Any genetic alteration in the source organism/cells (e.g. 'GFP-tagged', 'knockout of gene X').",
    "Genotype": "Genotypic background (e.g. 'C57BL/6J', 'BRCA1-mutant').",
    "GradientTime": (
        "The total duration of the LC gradient used to elute peptides. "
        "Valid values must state or imply a specific duration, either as a bare number with unit "
        "('120 min', '60 min'), a phrase that includes the duration "
        "('over 180 min', 'Standard 90-min gradients', '60-min gradient'). "
        "'60-min gradient' and '60 min' are the same value. "
        "A value is INVALID if it describes gradient slope or composition without stating total time."
    ),
    "GrowthRate": "Doubling time or growth rate of cell cultures (e.g. '24 h doubling').",
    "Instrument": "Mass spectrometer make/model (e.g. 'Thermo Q-Exactive Plus'). Must NOT be LC hardware or software.",
    "IonizationType": "Ionization source (e.g. 'nanoESI', 'MALDI'). 'positive ion mode' and 'positive ionization mode' are duplicates; keep the more specific/complete form.",
    "Label": (
        "Isobaric, isotopic, or metabolic labelling strategy applied to peptides or proteins "
        "(e.g. 'TMT-126', 'SILAC heavy', 'iTRAQ 4-plex'). "
        "Label-free quantification (LFQ) is also a valid Label value. "
        "'LFQ' and 'label-free quantification' or 'label-free quantitation' are duplicates; keep the full name."
    ),
    "MaterialType": "Broad class of material (e.g. 'tissue', 'cell line', 'biofluid').",
    "Modification": (
        "Post-translational modification enriched or studied (e.g. 'phosphorylation', 'ubiquitination', 'N-acetylation'). "
        "The modification name and a descriptive phrase about the same modification (e.g. 'N-acetylation' and "
        "'N-acetyl peptides') are duplicates; keep the canonical modification name."
    ),
    "MS2MassAnalyzer": "Analyzer used for MS2 (e.g. 'orbitrap', 'ion trap').",
    "NumberOfBiologicalReplicates": "Total number of biological replicates in the study.",
    "NumberOfFractions": "Total number of fractions generated from each sample.",
    "NumberOfMissedCleavages": "Maximum number of missed cleavages allowed in the database search.",
    "NumberOfSamples": "Total number of samples processed.",
    "NumberOfTechnicalReplicates": "Total number of technical replicates per sample.",
    "Organism": "Source species (NCBI Taxonomy ID and name, e.g. '9606 (Homo sapiens)').",
    "OrganismPart": (
        "Tissue, organ, biofluid, or life-stage-specific cell population of origin "
        "(Uberon term preferred, e.g. 'UBERON:0002107 (liver)'). "
        "Parasite life-stage cell forms such as 'tachyzoites' or 'bradyzoites' are valid OrganismPart "
        "values because they identify the specific cellular form of the organism used as source material. "
        "Biofluids such as 'urine' or 'plasma' are also valid OrganismPart values."
    ),
    "OriginSiteDisease": (
        "Anatomical site of disease origin (e.g. 'colon', 'prostate'). "
        "When the same organ already appears under OrganismPart as the tissue source, "
        "it is redundant in OriginSiteDisease and should be removed from OriginSiteDisease. "
        "Only keep OriginSiteDisease values that are NOT already listed in OrganismPart."
    ),
    "PooledSample": "Indicates if multiple samples were pooled (e.g. 'pool1 of reps1-3').",
    "PrecursorMassTolerance": "Mass tolerance for precursor matching (e.g. '10 ppm').",
    "ReductionReagent": "Chemical used to reduce disulfide bonds (e.g. 'DTT', 'TCEP'). 'DTT' is an abbreviation of 'dithiothreitol' / 'DL-Dithiothreitol'; keep the full name.",
    "SamplingTime": (
        "The time point at which a sample was collected, relative to a defined event or time-course. "
        "Examples: 'T0', '24 h post-treatment', 'three and seven days after pollination', 'day 3', '48 h'. "
        "INVALID: values that describe a treatment duration mixed with a drug name "
        "(e.g. '4 h IFNβ stimulation', '24 h IFNβ stimulation') where the drug/treatment context "
        "is captured elsewhere. These describe TREATMENT duration, not sample COLLECTION time point. "
        "INVALID: any value that mixes a non-temporal descriptor (size, morphology) into the time expression."
    ),
    "Separation": (
        "Any on-line chromatographic column or method used to separate peptides immediately before MS. "
        "Valid: column names, stationary-phase descriptions, general method labels (e.g. 'reversed-phase nanoLC'). "
        "INVALID: standalone LC system/instrument brand names that do NOT describe the column or separation "
        "chemistry (e.g. 'UltiMate 3000 RSLC nano LC system', 'Dionex Ultimate3000 nano-HPLC', "
        "'Easy nano LC 1000 HPLC', 'Agilent 1100 HPLC system'). These are instrument hardware names, not "
        "separation methods. Must NOT be an off-line fractionation method (that belongs to FractionationMethod)."
    ),
    "Sex": "Donor sex (e.g. 'male', 'female').",
    "Specimen": (
        "Description of biological specimen type as it was physically collected "
        "(e.g. 'biopsy', 'plasma', 'cell pellet', 'tissue section', 'swab'). "
        "Must NOT be a processed material (e.g. 'extracts' describes what was done TO the specimen, not the specimen itself). "
        "Must NOT be pharmaceutical products, vaccines, or reagents used as study input material."
    ),
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
        "'RH', 'Me49' for Toxoplasma gondii parasite strains. "
        "CRITICAL EXCLUSIONS -- Strain must NOT be used for: "
        "  - Human or primate cell line identifiers (belong to CellLine). "
        "  - Bait proteins. "
        "  - GeneticModification events."
    ),
    "SyntheticPeptide": "Indicates a synthetic peptide sample (e.g. 'synthetic phosphopeptide library').",
    "Temperature": "Growth temperature or perturbation temperature (e.g. '37 degC', '4 degC on ice').",
    "Time": (
        "A broad time parameter for the experiment that does not fit the more specific SamplingTime "
        "or DevelopmentalStage categories -- e.g. 'day 5', 'week 2', '72 h'. "
        "IMPORTANT: if a time value is already annotated under DevelopmentalStage or SamplingTime "
        "with clear biological meaning, a bare repetition under Time is redundant and should be removed. "
        "Time should be used only when the time expression does not fit the more specific types."
    ),
    "Treatment": (
        "An experimental treatment actively applied to samples, typically a drug, compound, stimulus, "
        "or defined intervention WITH dose and/or duration (e.g. 'drug X 5 uM 24 h', 'UV irradiation 30 min'). "
        "Must NOT be: "
        "  - A bare chemical compound or drug name without dose/duration context (use Compound instead). "
        "  - An siRNA construct identifier (e.g. 'siRPL28', 'siRPS26') -- those are molecular reagents/tools, "
        "    NOT treatments; use Compound or GeneticModification. "
        "  - A disease name, patient-group label, phenotype descriptor. "
        "  - A nutrient-replete control condition already captured under FactorValue or Depletion context. "
        "  - A spike-in compound used as internal standard (use SpikedCompound)."
    ),
    "TumorCellularity": "Percentage of tumor cells in the sample (e.g. '80%').",
    "TumorGrade": "Histological grade (e.g. 'Grade II').",
    "TumorSize": "Physical size of the tumor (e.g. '3 cm diameter').",
    "TumorSite": "Anatomical site of tumor (e.g. 'breast', 'pancreas').",
    "TumorStage": "Clinical staging (e.g. 'Stage III').",
    "AcquisitionMethod": (
        "MS acquisition scheme (e.g. 'DDA', 'DIA', 'PRM'). "
        "'DDA' and 'data-dependent acquisition' are duplicates; keep the full form. "
        "Different acquisition modes (e.g. 'data-dependent PASEF' vs. plain 'data-dependent acquisition') "
        "may be genuinely distinct if used on different instruments in the same study."
    ),
}

ANN_DIR = "./annotations"
JSON_PATH = "./papers_dataset.json"
OUTPUT_DIR = "./filtered_output_test2"
CSV_OUT = "./removed_annotations.csv"

MAX_WORKERS = 4
MAX_PAPER_CHARS = 55_000

ALLOW_CROSS_TYPE_SHARING_WITH = {"FactorValue"}


ABBREV_FULL: dict[str, list[tuple[str, str]]] = {
    "FragmentationMethod": [
        ("hcd", "higher energy collisional"),
        ("hcd", "higher-energy collisional"),
        ("hcd", "higher-energy collision"),
        ("cid", "collision-induced dissociation"),
        ("etd", "electron transfer dissociation"),
        ("ecd", "electron capture dissociation"),
        ("uvpd", "ultraviolet photodissociation"),
    ],
    "ReductionReagent": [("dtt", "dithiothreitol")],
    "AlkylationReagent": [("iaa", "iodoacetamide"), ("nem", "n-ethylmaleimide")],
    "Label": [("lfq", "label-free")],
    "AcquisitionMethod": [
        ("dda", "data-dependent acquisition"),
        ("dia", "data-independent acquisition"),
        ("prm", "parallel reaction monitoring"),
        ("srm", "selected reaction monitoring"),
        ("mrm", "multiple reaction monitoring"),
    ],
    "IonizationType": [
        ("esi", "electrospray ionization"),
        ("nanoesi", "nano electrospray"),
        ("maldi", "matrix-assisted laser"),
    ],
    "Depletion": [("s depleted", "sulfur-depleted"), ("s-depleted", "sulfur-depleted")],
}

_GRADE_PREFIX_RE = re.compile(
    r'^(ms[-\s]grade|sequencing[-\s]grade|lys-c grade|ultra[-\s]pure|hplc[-\s]grade)\s+',
    re.IGNORECASE,
)

_LC_SYSTEM_RE = re.compile(
    r'\b(ultimate\s*\d+|dionex|easy[-\s]nano\s*lc\s*\d+|agilent\s+\d+\s*hplc\s+system'
    r'|nano[-\s]hplc\s+system|rslc\s+nano\s+lc\s+system|nanoflow\s+hplc)\b',
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def _normalize_number(s: str) -> str:
    return re.sub(r'\b(\d+)\.0+\b', r'\1', s)


def _comparison_key(value: str, atype: str) -> str:
    v = _norm(value)
    v = _normalize_number(v)
    if atype == "GradientTime":
        v = re.sub(r'[-–]\s*min\s+gradient\b', ' min', v)
        v = re.sub(r'\bmin\s+gradient\b', 'min', v)
    if atype == "CleavageAgent":
        v = _GRADE_PREFIX_RE.sub('', v)
    if atype == "Modification":
        v = re.sub(r'\s+peptides?\b|\s+proteins?\b', '', v)
    return v.strip()


def _is_abbrev_of(short_val: str, long_val: str, atype: str) -> bool:
    pairs = ABBREV_FULL.get(atype, [])
    sn = _norm(short_val)
    ln = _norm(long_val)
    for abbrev, fragment in pairs:
        if sn == abbrev and fragment in ln:
            return True
    return False


def deterministic_dedup(atype: str, values: list[str]) -> tuple[list[str], list[dict]]:
    kept: list[str] = list(values)
    removed: list[dict] = []
    changed = True
    while changed:
        changed = False
        for i in range(len(kept)):
            for j in range(len(kept)):
                if i == j:
                    continue
                vi, vj = kept[i], kept[j]
                if _is_abbrev_of(vi, vj, atype):
                    removed.append({
                        "value": vi,
                        "reason": f"abbreviation of '{vj}'; keeping the full name",
                        "kept_instead": vj,
                    })
                    kept = [v for k, v in enumerate(kept) if k != i]
                    changed = True
                    break
                if i < j and _comparison_key(vi, atype) == _comparison_key(vj, atype):
                    if len(vi) >= len(vj):
                        removed.append({
                            "value": vj,
                            "reason": f"equivalent to '{vi}' after normalization",
                            "kept_instead": vi,
                        })
                        kept = [v for k, v in enumerate(kept) if k != j]
                    else:
                        removed.append({
                            "value": vi,
                            "reason": f"equivalent to '{vj}' after normalization",
                            "kept_instead": vj,
                        })
                        kept = [v for k, v in enumerate(kept) if k != i]
                    changed = True
                    break
            if changed:
                break
    return kept, removed


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


#gemma(OpenRouter)
def _extract_response_text(response) -> str:
    if not response or not getattr(response, "choices", None):
        raw = getattr(response, "model_dump", lambda: None)()
        if raw is None:
            raw = str(response)
        raise RuntimeError(
            f"OpenRouter returned no choices. Full response: {json.dumps(raw, default=str)[:500]}"
        )
    choice = response.choices[0]
    content = choice.message.content or ""
    if not content and hasattr(choice.message, "reasoning"):
        print(
            "    [WARN] message.content is empty but reasoning is present; "
            "model may have used all tokens for thinking.",
            file=sys.stderr,
        )
    return content


def _extract_json_from_text(raw_text: str) -> dict:
    clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    json_match = None
    for m in re.finditer(r'\{', clean):
        start = m.start()
        try:
            obj, _ = json.JSONDecoder().raw_decode(clean, start)
            if isinstance(obj, dict):
                json_match = obj
        except json.JSONDecodeError:
            continue
    if json_match is None:
        raise ValueError("No valid JSON object found in response")
    return json_match


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
        result = _extract_json_from_text(raw_text)
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
        print(f"    Raw response snippet: {raw_text[:500]}", file=sys.stderr)
        return {keep_key: list(original_values), remove_key: []}


def _call_gemma_cached(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=16384,
                temperature=0,
                top_p=0.95,
                extra_body={"enable_thinking": True},
            )
            return _extract_response_text(response)
        except Exception as e:
            last_err = e
            print(f"    [WARN] API call attempt {attempt}/{max_retries} failed: {e}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"All {max_retries} API attempts failed. Last error: {last_err}")


#system prompt 
ICL_DEDUP_EXAMPLES = """
IN-CONTEXT LEARNING EXAMPLES FOR DEDUPLICATION

-- EXAMPLE 1 -- Annotation type: GradientTime
Candidate values:
  1. 20 min
  2. A linear 150-min gradient
  3. 150-min
  4. 20 min

Reasoning:
  "20 min" (1): Keep.
  "A linear 150-min gradient" (2): Keep -- different duration AND adds shape detail.
  "150-min" (3): Remove -- strict subset of value 2 which is already in keep list.
  "20 min" (4): Remove -- exact duplicate of value 1.

Output:
{"keep":["20 min","A linear 150-min gradient"],"remove":[{"value":"150-min","reason":"bare duration already fully contained in 'A linear 150-min gradient'","kept_instead":"A linear 150-min gradient"},{"value":"20 min","reason":"exact duplicate of '20 min'","kept_instead":"20 min"}]}

-- EXAMPLE 2 -- Annotation type: FractionationMethod
Candidate values:
  1. A combination of SEC and top-down ODG
  2. a discontinuous top-down ODG

Reasoning:
  Value 2 describes only the ODG sub-step of the combined protocol in value 1.

Output:
{"keep":["A combination of SEC and top-down ODG"],"remove":[{"value":"a discontinuous top-down ODG","reason":"sub-component of combined SEC+ODG protocol; redundant","kept_instead":"A combination of SEC and top-down ODG"}]}

-- EXAMPLE 3 -- Annotation type: AcquisitionMethod
Candidate values:
  1. data-dependent acquisition
  2. DDA

Reasoning:
  "DDA" is the abbreviation of "data-dependent acquisition". Remove the abbreviation.

Output:
{"keep":["data-dependent acquisition"],"remove":[{"value":"DDA","reason":"abbreviation of 'data-dependent acquisition'","kept_instead":"data-dependent acquisition"}]}

-- EXAMPLE 4 -- Annotation type: FragmentationMethod
Candidate values:
  1. higher energy collisional dissociation
  2. HCD
  3. collision-induced dissociation
  4. CID

Reasoning:
  HCD is the abbreviation of value 1. CID is the abbreviation of value 3.
  Both full-name values describe genuinely different techniques and must both be kept.

Output:
{"keep":["higher energy collisional dissociation","collision-induced dissociation"],"remove":[{"value":"HCD","reason":"abbreviation of 'higher energy collisional dissociation'","kept_instead":"higher energy collisional dissociation"},{"value":"CID","reason":"abbreviation of 'collision-induced dissociation'","kept_instead":"collision-induced dissociation"}]}

-- EXAMPLE 5 -- Annotation type: Label
Candidate values:
  1. label-free quantitation
  2. LFQ

Reasoning:
  "LFQ" is the abbreviation. Remove it.

Output:
{"keep":["label-free quantitation"],"remove":[{"value":"LFQ","reason":"abbreviation of 'label-free quantitation'","kept_instead":"label-free quantitation"}]}

-- EXAMPLE 6 -- Annotation type: Instrument (keep both -- genuinely distinct)
Candidate values:
  1. Orbitrap Fusion Lumos (Thermo Fisher Scientific)
  2. Q Exactive HF (Thermo Fisher Scientific)

Output:
{"keep":["Orbitrap Fusion Lumos (Thermo Fisher Scientific)","Q Exactive HF (Thermo Fisher Scientific)"],"remove":[]}

-- EXAMPLE 7 -- Annotation type: CollisionEnergy
Candidate values:
  1. 42.0 eV
  2. 42 eV

Reasoning:
  42.0 eV and 42 eV are numerically identical. Keep the cleaner form.

Output:
{"keep":["42 eV"],"remove":[{"value":"42.0 eV","reason":"numerically identical to '42 eV'","kept_instead":"42 eV"}]}

-- EXAMPLE 8 -- Annotation type: Separation
Candidate values:
  1. reversed-phase pre-column
  2. reversed-phase analytical column

Reasoning:
  These describe two different columns in the LC setup (trap vs. analytical).
  However, if the paper uses only one reversed-phase analytical column and the
  'pre-column' description is a less complete duplicate of the setup, consult
  paper context. If the paper explicitly uses both a trap and an analytical column
  as separate LC stages, keep both. If 'reversed-phase pre-column' is simply a
  generic re-description of the same chromatographic approach, remove it.
  Use paper context to decide.

-- EXAMPLE 9 -- Annotation type: Modification
Candidate values:
  1. N-acetylation
  2. N-acetyl peptides

Reasoning:
  "N-acetyl peptides" is a circumlocution for the same modification. Remove.

Output:
{"keep":["N-acetylation"],"remove":[{"value":"N-acetyl peptides","reason":"circumlocution for 'N-acetylation'; same modification","kept_instead":"N-acetylation"}]}

-- EXAMPLE 10 -- Annotation type: CleavageAgent
Candidate values:
  1. trypsin
  2. MS-grade trypsin

Reasoning:
  "MS-grade trypsin" specifies a quality grade of the same enzyme. The grade
  does not create a scientifically distinct entry. Keep the canonical name.

Output:
{"keep":["trypsin"],"remove":[{"value":"MS-grade trypsin","reason":"grade modifier of 'trypsin'; does not create a distinct enzyme entry","kept_instead":"trypsin"}]}

-- EXAMPLE 11 -- Annotation type: Depletion
Candidate values:
  1. S depleted
  2. sulfur-depleted

Reasoning:
  "S depleted" is an abbreviated form of "sulfur-depleted". Same condition.

Output:
{"keep":["sulfur-depleted"],"remove":[{"value":"S depleted","reason":"abbreviated form of 'sulfur-depleted'","kept_instead":"sulfur-depleted"}]}

END OF DEDUP EXAMPLES.
"""



ICL_DISAMBIGUATION_EXAMPLES = """
IN-CONTEXT LEARNING EXAMPLES FOR CROSS-ANNOTATION DISAMBIGUATION

Background rule:
  FactorValue is ALWAYS allowed to mirror values from other types. Never remove from FactorValue.
  For every other conflict, exactly one assignment is correct and the other is an extraction error.

EXAMPLE 1  Conflict: SpikedCompound vs Treatment
Value: "biotin"
Decision: Remove from Treatment; keep in SpikedCompound (BioID substrate, not a perturbation).

Output:
{"decisions":[{"value":"biotin","keep_in":"SpikedCompound","remove_from":"Treatment","reason":"BioID proximity-labeling substrate (assay reagent), not a biological perturbation."}]}

EXAMPLE 2  FactorValue mirroring Disease -- NO ACTION
Value: "breast cancer" in both Disease and FactorValue.

Output:
{"decisions":[]}

EXAMPLE 3  Compound + FactorValue dose split -- NO ACTION
Compound: "human recombinant IFNb 1000 U/mL" / FactorValue: "1000 U/mL"

Output:
{"decisions":[]}

EXAMPLE 4  Conflict: OrganismPart vs OriginSiteDisease
Conflicting values: "prostate", "bladder"
  OrganismPart:      urine, prostate, bladder
  OriginSiteDisease: prostate, bladder, renal

Decision: Remove "prostate" and "bladder" from OriginSiteDisease (already in OrganismPart).
  "renal" is ONLY in OriginSiteDisease -- keep it.

Output:
{"decisions":[{"value":"prostate","keep_in":"OrganismPart","remove_from":"OriginSiteDisease","reason":"Already in OrganismPart as tissue source; redundant in OriginSiteDisease."},{"value":"bladder","keep_in":"OrganismPart","remove_from":"OriginSiteDisease","reason":"Already in OrganismPart; redundant in OriginSiteDisease."}]}

EXAMPLE 5  Conflict: Depletion vs Treatment
Value: "S-rich" in both Depletion and Treatment.
Decision: Remove from Depletion (nutrient-replete condition is opposite of depletion).

Output:
{"decisions":[{"value":"S-rich","keep_in":"Treatment","remove_from":"Depletion","reason":"Nutrient-replete condition; antithesis of depletion."}]}

END OF DISAMBIGUATION EXAMPLES.
"""


DEDUP_SYSTEM_PROMPT = f"""You are an expert proteomics and mass-spectrometry metadata curator. Your task is semantic deduplication of annotation values.

DEDUPLICATION RULES:
1. EXACT / NEAR-EXACT DUPLICATES: Remove values identical or differing only in whitespace, punctuation, or trivial capitalisation.
2. ABBREVIATION vs FULL NAME -- same entity: Keep the most complete form; remove the bare abbreviation.
3. GRADE / QUALITY MODIFIERS: For CleavageAgent, ReductionReagent, AlkylationReagent -- 'MS-grade X' and 'X' are the same reagent. Remove the grade-modified form.
4. SUBSET / COMPONENT already CONTAINED in another value: If value A contains all the information of value B and more, remove B and keep A.
5. COMPOSITE VALUES that COMBINE already-listed INDIVIDUAL values: Remove the composite.
6. NUMERIC NORMALISATION: '42.0 eV' and '42 eV' are the same. Keep the cleaner form.
7. GRADIENT TIME SUFFIX: '60-min gradient' and '60 min' are the same GradientTime. Keep the more descriptive form.
8. MODIFICATION CIRCUMLOCUTIONS: 'N-acetyl peptides' and 'N-acetylation' describe the same PTM. Keep the canonical modification name.
9. GENUINELY DISTINCT values -- KEEP BOTH: Different durations, different instruments, different tissue origins are not duplicates.

DECISION GUIDELINE: Only remove when CONFIDENT the values convey the same information. When in doubt, keep both.

{ICL_DEDUP_EXAMPLES}

OUTPUT FORMAT (strict JSON only -- no markdown fences, no preamble):
{{"keep": ["value1", ...], "remove": [{{"value": "removed_value", "reason": "...", "kept_instead": "..."}}]}}

Every original value must appear in exactly one of "keep" or "remove". Do NOT invent or rephrase values.
Reason explicitly about each value before writing the JSON."""





DISAMBIGUATION_SYSTEM_PROMPT = f"""You are an expert proteomics and mass-spectrometry metadata curator.

TASK: Cross-annotation disambiguation.

The same value has been assigned to more than one annotation type (excluding FactorValue).
Use the paper text and type definitions to decide which assignment is correct.

DISAMBIGUATION RULES:
1. EXACTLY ONE CORRECT TYPE unless value legitimately belongs to multiple simultaneously.
2. FACTORVALUE IS ALWAYS ALLOWED TO SHARE. Never remove a value from FactorValue.
3. COMPOUND + FACTORVALUE DOSE SPLIT: if Compound has the full "compound + dose" string and FactorValue has only the dose portion, these are NOT the same string -- no conflict.
4. ORIGINSITEDISEASE vs ORGANISMPART: When the same organ appears in both types and the tissue source IS the disease-origin organ, OriginSiteDisease is redundant.
5. USE PAPER TEXT AS GROUND TRUTH.
6. WHEN IN DOUBT -- KEEP THE MORE SPECIFIC TYPE.

{ICL_DISAMBIGUATION_EXAMPLES}

OUTPUT FORMAT (strict JSON only -- no markdown fences, no preamble):
{{"decisions": [{{"value": "the conflicting value", "keep_in": "TypeName" OR ["TypeName1", "TypeName2"], "remove_from": "TypeName" OR ["TypeName1", "TypeName2"], "reason": "concise explanation"}}]}}

Omit a conflict from decisions entirely if no removal is needed.
Reason about each conflict before writing the JSON."""


def build_dedup_user_prompt(atype: str, values: list[str], paper_text: str) -> str:
    numbered_values = "\n".join(f"  {i+1}. {v}" for i, v in enumerate(values))
    return f"""YOUR TASK
---------
Annotation type: "{atype}"

Candidate values:
{numbered_values}

PAPER TEXT (context only):
{paper_text}"""


def call_api_for_dedup(client: OpenAI, atype: str, values: list[str], paper_text: str) -> dict:
    if len(values) == 1:
        return {"keep": values, "remove": []}
    raw_text = _call_gemma_cached(
        client,
        DEDUP_SYSTEM_PROMPT,
        build_dedup_user_prompt(atype, values, paper_text),
    )
    return _safe_parse_api_response(
        raw_text,
        required_keys=("keep", "remove"),
        original_values=set(values),
        keep_key="keep",
        remove_key="remove",
        atype=atype,
        context_label="dedup",
    )


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


def build_disambiguation_user_prompt(
    conflicts: list[dict],
    all_annotations: dict[str, list[str]],
    paper_text: str,
) -> str:
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

    return f"""YOUR TASK
---------

CONFLICTING VALUES:
{conflict_block}

ANNOTATION TYPE DEFINITIONS (involved types):
{definitions_block}

ALL CURRENT ANNOTATIONS (for context):
{all_ann_block}

PAPER TEXT:
{paper_text}"""


def call_api_for_disambiguation(
    client: OpenAI,
    conflicts: list[dict],
    all_annotations: dict[str, list[str]],
    paper_text: str,
) -> list[dict]:
    if not conflicts:
        return []
    raw_text = _call_gemma_cached(
        client,
        DISAMBIGUATION_SYSTEM_PROMPT,
        build_disambiguation_user_prompt(conflicts, all_annotations, paper_text),
    )
    try:
        result = _extract_json_from_text(raw_text)
        decisions = result.get("decisions", [])
        for d in decisions:
            if isinstance(d.get("keep_in"), str):
                d["keep_in"] = [d["keep_in"]]
            if isinstance(d.get("remove_from"), str):
                d["remove_from"] = [d["remove_from"]]
        return decisions
    except Exception as e:
        print(f"    [WARN] [disambiguation] Could not parse response: {e}", file=sys.stderr)
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


def _process_single_type(
    client: OpenAI,
    atype: str,
    deduped_values: list[str],
    paper_text: str,
) -> tuple[str, list[str], list[dict]]:

    dedup_removals = []

    if len(deduped_values) >= 2:
        result = call_api_for_dedup(client, atype, deduped_values, paper_text)
        kept = result.get("keep", deduped_values)
        for removal in result.get("remove", []):
            dedup_removals.append(removal)
    else:
        kept = deduped_values

    return atype, kept, dedup_removals


def process_ann_file(
    ann_path: str,
    paper_text: str,
    client: OpenAI,
    output_dir: str,
    csv_writer,
    csv_fh,
):
    base = Path(ann_path).stem
    pmc_match = re.search(r'(PMC\d+)', base)
    stem = pmc_match.group(1) if pmc_match else base

    print(f"Processing: {ann_path}  (stem={stem})")
    print(f"{'='*70}")

    entries = read_ann_file(ann_path)
    if not entries:
        print(f"  [WARN] No valid annotation entries found in {ann_path}")
        return 0

    groups = group_by_type(entries)
    removed_records: list[dict] = []

    truncated_paper = paper_text[:MAX_PAPER_CHARS]

    pre_deduped: dict[str, list[str]] = {} 
    single_value_types: dict[str, list[str]] = {} 

    for atype, values in groups.items():
        print(f"\n  [{atype}] {len(values)} raw value(s):")
        for v in values:
            print(f"    * {v}")

        #exact duplicates
        deduped, exact_removed = remove_exact_duplicates(values)
        for removed_val, kept_val in exact_removed:
            print(f"    EXACT DUPLICATE removed: '{removed_val}' (kept: '{kept_val}')")
            rec = {
                "file": Path(ann_path).name, "stem": stem,
                "annotation_type": atype, "removed_value": removed_val,
                "kept_instead": kept_val,
                "reason": "exact duplicate (case-insensitive)", "pass": "deduplication",
            }
            removed_records.append(rec)
            csv_writer.writerow(rec)
            csv_fh.flush()

        #deterministic abbreviation / normalisation
        deduped, det_removed = deterministic_dedup(atype, deduped)
        for dr in det_removed:
            print(f"    DETERMINISTIC REMOVE: '{dr['value']}' -- {dr['reason']}")
            rec = {
                "file": Path(ann_path).name, "stem": stem,
                "annotation_type": atype, "removed_value": dr["value"],
                "kept_instead": dr.get("kept_instead", ""),
                "reason": dr["reason"], "pass": "deduplication",
            }
            removed_records.append(rec)
            csv_writer.writerow(rec)
            csv_fh.flush()

        if len(deduped) <= 1:
            single_value_types[atype] = deduped
        else:
            pre_deduped[atype] = deduped

    # semantic dedup per type
    print(f"\n  -> Running concurrent semantic dedup for {len(pre_deduped)} multi-value types "
          f"+ {len(single_value_types)} single-value types...")

    validated_values: dict[str, list[str]] = {}

    all_types_to_process = {}
    for atype, vals in single_value_types.items():
        all_types_to_process[atype] = vals
    for atype, vals in pre_deduped.items():
        all_types_to_process[atype] = vals

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for atype, vals in all_types_to_process.items():
            if not vals:
                validated_values[atype] = []
                continue
            future = executor.submit(
                _process_single_type, client, atype, vals, truncated_paper
            )
            futures[future] = atype

        for future in as_completed(futures):
            atype = futures[future]
            try:
                _, valid_vals, dedup_rems = future.result()
                validated_values[atype] = valid_vals

                #dedup removals
                for removal in dedup_rems:
                    val = removal.get("value", "")
                    reason = removal.get("reason", "duplicate")
                    kept_instead = removal.get("kept_instead", "")
                    print(f"    SEMANTIC REMOVE [{atype}]: '{val}'")
                    print(f"      reason      : {reason}")
                    print(f"      kept instead: '{kept_instead}'")
                    rec = {
                        "file": Path(ann_path).name, "stem": stem,
                        "annotation_type": atype, "removed_value": val,
                        "kept_instead": kept_instead,
                        "reason": reason, "pass": "deduplication",
                    }
                    removed_records.append(rec)
                    csv_writer.writerow(rec)
                    csv_fh.flush()

            except Exception as e:
                print(f"    [ERROR] Failed processing {atype}: {e}", file=sys.stderr)
                validated_values[atype] = all_types_to_process.get(atype, [])


    #cross annotation disambiguation 
    print(f"\n  -> Running cross-annotation disambiguation pass...")
    conflicts = detect_cross_type_conflicts(validated_values)

    if conflicts:
        print(f"  Found {len(conflicts)} cross-type conflict(s):")
        for c in conflicts:
            print(f"    * \"{c['value']}\" in: {', '.join(c['types'])}")

        decisions = call_api_for_disambiguation(
            client, conflicts, validated_values, truncated_paper
        )
        final_values, disambiguation_removals = apply_disambiguation_decisions(
            validated_values, decisions
        )
        for rec in disambiguation_removals:
            full_rec = {"file": Path(ann_path).name, "stem": stem, **rec}
            removed_records.append(full_rec)
            csv_writer.writerow(full_rec)
            csv_fh.flush()
    else:
        print(f"  No cross-type conflicts found.")
        final_values = validated_values

    #produce output files
    seen_types_order: list[str] = []
    for atype, _ in entries:
        if atype not in seen_types_order:
            seen_types_order.append(atype)

    out_path = os.path.join(output_dir, f"{stem}_filtered.ann")
    with open(out_path, "w", encoding="utf-8") as fh:
        for atype in seen_types_order:
            for val in final_values.get(atype, []):
                fh.write(f"{atype}: {val}\n")

    n_removed = len(removed_records)
    print(f"\n  [OK] Filtered .ann written : {out_path}")
    print(f"  [OK] Removed {n_removed} annotation(s) total from this file")
    return n_removed


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

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("[WARN] OPENROUTER_API_KEY is not set -- LLM calls will fail.", file=sys.stderr)

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    csv_fields = [
        "file", "stem", "annotation_type", "removed_value",
        "kept_instead", "reason", "pass",
    ]
    csv_path = Path(CSV_OUT)

    total_removed = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(
            csv_fh, fieldnames=csv_fields,
            quoting=csv.QUOTE_ALL, extrasaction="ignore",
        )
        writer.writeheader()
        csv_fh.flush()

        for ann_path in ann_files:
            base = ann_path.stem
            pmc_match = re.search(r'(PMC\d+)', base)
            stem = pmc_match.group(1) if pmc_match else base

            paper_text = read_paper_text(json_data, stem)
            if not paper_text:
                print(
                    f"  [WARN] No paper text found for stem '{stem}' -- proceeding without context.",
                    file=sys.stderr,
                )

            try:
                n = process_ann_file(
                    str(ann_path), paper_text, client,
                    str(output_dir), writer, csv_fh,
                )
                total_removed += n
            except Exception as e:
                print(f"  [ERROR] Failed to process {ann_path.name}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

    print(f"\nDONE. Total annotations removed: {total_removed}")
    print(f"Filtered .ann files : {output_dir.resolve()}/")
    print(f"Removal report CSV  : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
