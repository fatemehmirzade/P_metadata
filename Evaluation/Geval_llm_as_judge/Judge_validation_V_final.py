import os
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import openai
import hashlib
import time

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


PAPERS_JSON      = "//Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/papers_dataset.json"
RESULTS_DIR      = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/results_gemma/Judge_Gemma"
PREDICTIONS_DIR  = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/filtered_output_test2"
FILTERED_DIR     = os.path.join(RESULTS_DIR, "filtered_annotations")


EVALUATION_MODEL      = "google/gemma-4-31b-it"
OPENROUTER_BASE_URL   = "https://openrouter.ai/api/v1"
MODEL_TEMPERATURE     = 0
MODEL_ENABLE_THINKING = True

CACHE_DIR     = os.path.join(RESULTS_DIR, ".prompt_cache")
CACHE_ENABLED = True

#parallel LLM 
MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_DELAY = 2  

os.makedirs(RESULTS_DIR, exist_ok=True)


_thread_local = threading.local()


_ENTITY_DEFINITIONS = """\
SAMPLE / BIOLOGICAL ENTITY TYPES:
Age                         : Age of the donor or developmental stage (e.g. "45 years", "E14.5 embryo").
AlkylationReagent           : Chemical that alkylates cysteine -SH groups (e.g. "IAA", "NEM").
AnatomicSiteTumor           : Anatomical location of tumor sample (e.g. "left lung lobe").
AncestryCategory            : Donor ancestry / ethnicity (e.g. "European", "East Asian").
Bait                        : The specific named protein or molecule used as the affinity handle in AP-MS / pull-down. Must be a concrete identifier such as a gene name, protein name, or tagged construct (e.g. "Spt16-TAP", "GFP-HDAC1", "Flag-Ago2"). Generic tag names alone ("Flag", "GFP", "HA") are NOT Bait.
BMI                         : Body-Mass Index of donor (kg/m^2).
BiologicalReplicate         : Identifier label for a specific biological replicate instance (e.g. "bioRep1"). NOT a count.
CellLine                    : Name of immortalized cell line (e.g. "HEK293T", "U2OS").
CellPart                    : Subcellular compartment / fraction (e.g. "nucleus", "mitochondria"). Must be a structural compartment or organelle, NOT a method or process.
CellType                    : Primary cell type or lineage (e.g. "neurons", "fibroblasts").
CleavageAgent               : Protease or chemical used for protein digestion (e.g. "trypsin", "Lys-C").
Compound                    : Chemical, small molecule, or protein/biological molecule added to sample as a biological treatment or perturbation. Includes embedding/processing media such as OCT, paraffin, or formalin. The extracted value MAY include a concentration prefix or suffix (e.g. "20 nM Calyculin A"). NOT the dose alone. NOT sample preparation reagents, LC solvents, MS calibrants, or labeling reagents.
ConcentrationOfCompound     : The dose or amount of a Compound used. Must be a numeric quantity with a unit. Must pair with a named Compound.
Depletion                   : Method to remove high-abundance proteins from a sample (e.g. "albumin depletion kit", "immunodepletion"). Must be a TECHNICAL METHOD for protein removal. Biological conditions describing nutrient depletion (e.g. "sulfur-depleted", "serum-starved") are NOT Depletion -- they may be FactorValue, Treatment, or Compound depending on context.
DevelopmentalStage          : Developmental stage of source (e.g. "adult", "P7 pup").
Disease                     : Disease state or diagnosis (e.g. "breast cancer", "Type 2 diabetes").
DiseaseTreatment            : Pre-treatment applied to diseased samples (e.g. "chemotherapy", "radiation").
FactorValue                 : The names of study variables or experimental factors that define what differs between sample groups in the proteomics experiment. These are the systematically varied conditions across proteomics samples (e.g. "prostate cancer", "benign prostate hyperplasia", "4 h", "24 h", "1000 U/mL"). Extract only the factor names/values, not the individual sample identifiers. Overlap with other annotation types (Time, Temperature, Compound) is allowed and expected when those values also define the axes of experimental variation. If the study is purely descriptive or has a single condition with no comparison, FactorValue may be empty. Sample identifiers, manufacturer codes, species names, and reference labels are NOT FactorValues.
GeneticModification         : Genetic alteration in organism/cells (e.g. "GFP-tagged", "CRISPR knockout").
Genotype                    : Genotypic background (e.g. "C57BL/6J", "BRCA1-mutant").
GrowthRate                  : Doubling time or growth rate (e.g. "24 h doubling time").
Label                       : Isobaric or metabolic label applied (e.g. "TMT-126", "SILAC heavy", "label-free", "triple SILAC"). Full IUPAC isotope names are valid. A single channel descriptor ALONE ("light", "medium", "heavy") is NOT a complete Label. NOTE: "SILAC" alone is NOT equivalent to "triple SILAC" -- if the paper describes a triple SILAC experiment, "SILAC" is a supertype (broader term) and should receive MEDIUM verdict (supertype drift). The specific SILAC channel labels (e.g. "Arg6/Lys4", "Arg10/Lys8") are also valid Label values.
MaterialType                : Broad material class (e.g. "tissue", "cell line", "biofluid").
Modification                : PTM studied or enriched for, and chemical modifications used as search parameters.
NumberOfBiologicalReplicates: Total COUNT of biological replicates in the study (e.g. "3"). NOT an identifier.
NumberOfSamples             : Total COUNT of samples processed (e.g. "12").
NumberOfTechnicalReplicates : Total COUNT of technical replicates per sample (e.g. "2").
Organism                    : Source species (e.g. "Homo sapiens", "Mus musculus"). Common names ("human", "mouse") are equivalent to scientific names.
OrganismPart                : Tissue, organ, body part, or organism life stage/morphological form of origin (e.g. "liver", "brain cortex", "plasma", "tachyzoites", "sporozoites", "oocysts"). Includes parasite life stages when used as the biological source material for proteomics.
OriginSiteDisease           : Anatomical site of disease origin (e.g. "colon", "prostate").
PooledSample                : Whether multiple samples were pooled (e.g. "pool1 of reps1-3").
ReductionReagent            : Chemical used to reduce disulfide bonds (e.g. "DTT", "TCEP").
SamplingTime                : Time point of sample collection (e.g. "T0", "24 h post-treatment").
Sex                         : Donor sex (e.g. "male", "female").
Specimen                    : Description of biological specimen type as collected from the donor (e.g. "biopsy", "plasma", "urine", "FFPE tissue"). Must name the biological material itself, NOT the embedding medium / preservative (OCT, paraffin, formalin alone are Compound, not Specimen) and NOT a processing container ("blocks", "sections" alone are not Specimen).
SpikedCompound              : Exogenous standard or spike-in added to sample (e.g. "iRT peptides").
Staining                    : Staining applied to sample prior to MS.
Strain                      : Organism strain or cultivar -- includes animal strains (e.g. "BALB/c", "FVB/N"), plant cultivars/varieties (e.g. "Nipponbare", "Col-0"), and microbial strains. NOT limited to animals.
SyntheticPeptide            : Indicates a synthetic peptide sample.
Temperature                 : Growth or perturbation temperature applied to living cells/organisms before lysis (e.g. "37 C", "42 C heat shock"). NOT instrument temperatures, column oven temperatures, or reagent incubation temperatures from sample preparation protocols.
Time                        : Broad time parameter of experiment -- biological time describing when samples were collected or how long a biological treatment lasted (e.g. "day 5", "week 2", "4 h post-stimulation"). NOT LC gradient times, MS acquisition times, or reagent incubation durations from sample preparation protocols.
Treatment                   : Experimental treatment applied (e.g. "EGF stimulation 10 ng/mL 30 min").
TumorCellularity            : Percentage of tumor cells in sample (e.g. "80%").
TumorGrade                  : Histological tumor grade (e.g. "Grade II").
TumorSize                   : Physical size of tumor (e.g. "3 cm diameter").
TumorSite                   : Anatomical site of tumor (e.g. "breast", "pancreas").
TumorStage                  : Clinical staging (e.g. "Stage III").

TECHNICAL / MS ENTITY TYPES:
AcquisitionMethod           : MS acquisition scheme (e.g. "DDA", "DIA", "PRM"). Must be a scheme name, NOT an instrument name.
CollisionEnergy             : Collision energy in MS/MS (e.g. "27 eV", "normalized CE 28%", "25").
EnrichmentMethod            : Peptide/protein enrichment protocol before MS (e.g. "TiO2 phosphopeptide enrichment", "C18 StageTip desalting").
FlowRateChromatogram        : LC flow rate (e.g. "300 nL/min").
FractionationMethod         : OFF-LINE fractionation of the bulk PEPTIDE or PROTEIN sample before LC-MS runs, used to reduce sample complexity (e.g. "high-pH reverse-phase fractionation", "SCX", "IEF/OFFGEL"). This refers to technical fractionation of the digest or protein mixture. It does NOT include biological/cellular fractionation of tissue or cells into compartments (nucleus, cytoplasm, etc.) — that is sample preparation, not FractionationMethod.
FractionIdentifier          : ID label for each fraction (e.g. "F1", "F2", "fraction 3").
FragmentationMethod         : Ion-fragmentation technique (e.g. "HCD", "CID", "ETD").
FragmentMassTolerance       : Mass tolerance or tolerances for fragment ion matching in search (e.g. "0.02 Da", "20 mDa", "10 ppm", "20, 10, and 7 ppm"). A list of tolerance values stated together is valid.
GradientTime                : LC gradient time -- either the total gradient length (e.g. "120 min", "90-min gradient") OR individual gradient segment durations as extracted from the gradient description (e.g. "24 min", "8 min"). Statements like "5-45% B in 120 min" that include the total time ARE valid. Individual segment times extracted from a multi-step gradient description are also valid.
Instrument                  : Mass spectrometer make and model (e.g. "Thermo Q-Exactive Plus", "Orbitrap Fusion Lumos"). LC instruments are NOT this type.
IonizationType              : Ionization source type (e.g. "nano-ESI", "MALDI", "Electrospray").
MS2MassAnalyzer             : Analyzer used for MS2 scans (e.g. "Orbitrap", "ion trap").
NumberOfMissedCleavages     : Max missed cleavages allowed in database search (e.g. "2").
NumberOfFractions           : Total COUNT of fractions generated per sample.
PrecursorMassTolerance      : Mass tolerance for precursor matching in search (e.g. "10 ppm", "5 ppm").
Separation                  : ON-LINE LC separation method -- must describe the stationary phase, column chemistry, or chromatographic approach. LC instrument model names alone are NOT valid Separation values.
"""

_STATIC_SYSTEM_PROMPT = """\
You are an expert in proteomics and mass spectrometry experimental metadata.
Your task is to assess the correctness of annotation values against a paper text
and the annotation-type definitions provided below.

You will receive two inputs:
  1. A PAPER TEXT message (abstract + methods), prepended to the conversation.
  2. An assessment request as a JSON object with a "task" key.

=================================================================
TASK TYPE: "verify"
=================================================================
Determine whether a candidate value is a valid annotation for the
given annotation type, by reading the paper text and applying the type
definitions below.

JSON input fields:
  annotation_type   -- the annotation category to assess
  type_definition   -- precise definition of that category
  candidate_value   -- the value to assess (ONE value, not a set)

=================================================================
CORE EVALUATION QUESTIONS (answer in order)
=================================================================
1. TYPE CHECK   : Does the candidate value match the definition of
                  annotation_type?
                  -- Ask: "Is this value fundamentally the RIGHT CATEGORY
                     OF THING (chemical, organism, method, instrument, count,
                     etc.) for this annotation type?"
                  -- If the value is the right CATEGORY but the wrong
                     SPECIFIC KIND within that category: TYPE_CORRECT: yes,
                     VALUE_CORRECT: no, VERDICT: low (rule D -- wrong value).
                  -- Only set TYPE_CORRECT: no (rule B) when the value's
                     FUNDAMENTAL NATURE belongs to a completely different
                     annotation type. See TYPE CHECK guidance below.
2. SOURCE CHECK : Is the candidate value present in the paper text?
                  If absent -> HALLUCINATED: yes (rule A -> VERDICT: low).
                  HALLUCINATION means the value cannot be found anywhere
                  in the paper text (abstract or methods). If the value
                  IS in the paper text but is used in a different context,
                  that is rule D (wrong value), NOT hallucination.
3. TRUTH CHECK  : Is the candidate value factually correct and valid for
                  this experiment? Does it represent a real, accurate
                  entity/parameter of the correct type as described in the
                  paper? If not -> VALUE_CORRECT: no (rule D -> VERDICT: low).
4. COMPLETENESS : Is the value fully specified? Naming only one arm of a
                  confirmed multi-arm scheme -> MEDIUM (rule c).
                  Subtype or supertype drift -> MEDIUM (rules a/b).

=================================================================
FACTORVALUE-SPECIFIC EVALUATION GUIDANCE
=================================================================
When annotation_type is "FactorValue":

TYPE CHECK: A FactorValue must be the NAME of a study variable or
experimental factor that defines what differs between sample groups
in the proteomics experiment. Valid FactorValues include:
  - Specific biological conditions varied across sample groups
    (e.g. "prostate cancer", "benign prostate hyperplasia")
  - Specific dose values when they define sample groups (e.g. "1000 U/mL")
  - Specific time points when they define sample groups (e.g. "4 h", "24 h")
  - Specific treatment names that differ between groups
  - Specific tissue/organ names when they define distinct sample groups
    (e.g. "plume", "trophosome")

The following are NOT valid FactorValues:
  - Sample identifiers, batch codes, or manufacturer codes (e.g. "M1", "M2")
  - Species names used merely as sample descriptors, not varied conditions
  - Reference sample labels (e.g. "ref-DP")
  - Abstract category names (e.g. "tissue", "condition", "treatment")
  - SILAC/TMT/iTRAQ labeling strategies
  - Technical parameters (instrument settings, LC parameters)

A FactorValue MAY overlap with other annotation types (Time, Temperature,
Compound, CellType, CellLine, Disease, OrganismPart, Treatment, Strain,
DevelopmentalStage, GeneticModification, or any other type) -- this is
expected and correct. Do NOT downgrade because a value also appears
under another annotation type.

For studies with no controlled experimental design (descriptive studies,
characterization studies, single-condition profiling), having NO FactorValues
is correct.

SOURCE CHECK for FactorValue: The value (or a domain-equivalent form) must
appear in the paper text as a condition that differentiates proteomics sample
groups. A value mentioned only in non-proteomics sections (e.g. only in a
Western blot experiment with no connection to MS) is rule D, not rule A.

TRUTH CHECK for FactorValue: The value must genuinely define a systematically
varied condition across proteomics sample groups. A value present in the text
but NOT actually varied as an experimental factor (e.g. a constant condition
across all samples) is VALUE_CORRECT: no (rule D).

=================================================================
VERDICT RULES
=================================================================

LOW (most severe) -- use when ANY of:
  (A) HALLUCINATED: absent from paper text.
  (B) TYPE MISMATCH: the value's FUNDAMENTAL NATURE is that of a
      completely different annotation type.
      -- A method/process/protocol labeled with the WRONG KIND of method
         is NOT a type mismatch. It is rule (D): wrong value, TYPE_CORRECT: yes.
      -- A structural compartment (nucleus, cytoplasm) is a CellPart, NOT
         a FractionationMethod -- that IS a type mismatch (rule B).
      -- A PROCESS of separating cellular components labeled as
         FractionationMethod: this is rule (D), NOT rule (B), because it IS
         a separation/fractionation-type process, just the wrong kind for
         the technical off-line peptide fractionation that FractionationMethod
         requires. TYPE_CORRECT: yes, VALUE_CORRECT: no, VERDICT: low.
      Examples of TRUE type mismatches (rule B):
        OCT/paraffin labeled as Specimen (correct type: Compound);
        DDA labeled as Instrument; generic "Flag" tag labeled as Bait;
        numeric dose alone labeled as Compound (correct: ConcentrationOfCompound);
        "nucleus" labeled as FractionationMethod (correct: CellPart);
        A sample identifier like "M1" labeled as FactorValue (correct: none/identifier);
        A labeling strategy like "SILAC" labeled as FactorValue (correct: Label).
  (D) CONTRADICTION / INCORRECT: present in source text, type is correct,
      but the value is factually wrong, contradicts the paper, or is the
      wrong specific kind within the correct category.
      Examples:
        "Subcellular fractionation" for FractionationMethod: the paper did
        perform subcellular fractionation of cells, but FractionationMethod
        requires OFF-LINE PEPTIDE/PROTEIN fractionation. The candidate IS a
        fractionation process (right category), but the wrong kind. -> rule D.
        "Mus musculus" when the paper uses human samples -> rule D (and also
        hallucinated if "Mus musculus" does not appear in the paper).
        "tissue" as FactorValue when it is an abstract category name, not a
        specific condition varied across samples -> rule D.

MEDIUM -- use when type is correct, value is present, no LOW rule applies,
AND any of:
  (a) SUBTYPE DRIFT: more specific subtype of what the paper describes.
  (b) SUPERTYPE DRIFT: strictly broader (e.g. "cells" when paper says
      "cell line").
  (c) SINGLE CHANNEL: names one arm of a confirmed multi-arm scheme
      (e.g. only "heavy" when the paper uses triple SILAC).

HIGH -- use when: correct type, present in source text, factually accurate
and valid for this experiment, and complete.

=================================================================
IMPORTANT: THINGS THAT ARE NEVER A REASON TO DOWNGRADE
=================================================================
  - Verbose or sentence-form phrasing.
  - A list of values for the same annotation type (e.g. "20, 10, and 7 ppm").
  - Quality/purity qualifiers ("MS-grade trypsin" = "trypsin").
  - Vendor/manufacturer/geographic suffixes.
  - Parenthetical abbreviation expansion.
  - Leading articles or filler words.
  - Digit/word swaps ("3" = "three").
  - Unit spacing/case differences ("37 C" = "37C").
  - Common-name/scientific-name swaps ("human" = "Homo sapiens").
  - Descriptive label on a canonical value ("first search tolerance 20 ppm").
  - Gradient statements including total time.
  - Compound+concentration bundle.
  - Enzyme variant/configuration ("Trypsin/P" = trypsin with Pro-rule).
  - FactorValue overlapping with Time, Temperature, Compound, CellType,
    CellLine, Disease, OrganismPart, Treatment, Strain, or any other type.
  - A value that is valid for its annotation type AND also valid as FactorValue.
  - Individual gradient segment times extracted from gradient descriptions.

=================================================================
DOMAIN EQUIVALENCES (treat as identical)
=================================================================
  - Full IUPAC SILAC isotope names = shorthand pairs
  - "HCD" = "higher-energy collisional dissociation"
  - "IAA" = "iodoacetamide"; "DTT" = "dithiothreitol"
  - "Homo sapiens" = "human"; "Mus musculus" = "mouse"
  - Any abbreviation in parentheses that expands the other value.
  - "Trypsin/P" = "trypsin" (Pro-rule configuration).

NOTE: "SILAC" is NOT equivalent to "triple SILAC". "SILAC" is a supertype
of "triple SILAC" and should receive MEDIUM verdict for supertype drift
when the paper describes a triple SILAC experiment.

=================================================================
TYPE CHECK guidance
=================================================================
Ask: "What is the FUNDAMENTAL NATURE of this value?"
  -- Is it a CHEMICAL?               -> Compound / AlkylationReagent / ReductionReagent / etc.
  -- Is it a SPECIES NAME?           -> Organism
  -- Is it a STRUCTURAL COMPARTMENT? -> CellPart / OrganismPart
  -- Is it a PROCESS / METHOD?       -> depends which kind
  -- Is it a NUMERIC COUNT?          -> NumberOf...
  -- Is it an INSTRUMENT MODEL?      -> Instrument
  -- Is it a STUDY VARIABLE / EXPERIMENTAL FACTOR? -> FactorValue

FractionationMethod specifically: The value must describe OFF-LINE technical
fractionation of a peptide or protein mixture (high-pH RP, SCX, OFFGEL IEF
of the DIGEST). Biological/cellular fractionation (isolating nuclei, cytoplasm,
membranes from cells/tissue) is a sample preparation step -- it has the right
PROCESS FLAVOR (separation/fractionation) but is the wrong KIND. This is
VALUE_CORRECT: no, TYPE_CORRECT: yes, VERDICT: low (rule D). It is NOT
a type mismatch to CellPart because "subcellular fractionation" is a process,
not a structural compartment.

FactorValue specifically: The value must name a study variable or condition
that is systematically varied across proteomics sample groups. Sample
identifiers, manufacturer codes, and abstract category names are NOT valid
FactorValues. However, specific conditions (disease states, treatment doses,
time points, tissue names) ARE valid when they define distinct sample groups.

LC instrument names are Instrument not Separation.
DDA/DIA/PRM are AcquisitionMethod not instrument names.
Generic tags are GeneticModification not Bait.
Numeric dose alone is ConcentrationOfCompound not Compound.
OCT/paraffin/formalin are Compound not Specimen.
Structural compartments (nucleus, cytoplasm) are CellPart not FractionationMethod.
Labeling strategies (SILAC, TMT) are Label not FactorValue.
Sample identifiers (M1, M2, batch codes) are not FactorValue.

Strain includes ALL organism strains: animal strains (BALB/c), plant cultivars
(Nipponbare, Col-0), microbial strains. Do NOT reject plant cultivars.

OrganismPart includes life stages and morphological forms of organisms
(tachyzoites, sporozoites, oocysts) when they describe the biological source.
Do NOT mismatch these to CellType.

GradientTime accepts both total gradient time AND individual segment durations
from gradient descriptions. Individual segments are valid extractions.

Biological conditions like "sulfur-depleted" or "serum-starved" are NOT
Depletion methods. They may be FactorValue, Treatment, or Compound.

=================================================================
OUTPUT FORMAT for "verify"
=================================================================
TYPE CHECK:        <one sentence -- if wrong type, state the correct type>
SOURCE CHECK:      <one sentence: is the candidate value present in the paper text?>
TRUTH CHECK:       <one sentence: is the candidate value factually correct and valid
                   for this experiment as described in the paper?>
COMPLETENESS CHECK:<one sentence>
TYPE_CORRECT:      yes | no
CORRECT_TYPE_NAME: <the correct TypeName when TYPE_CORRECT is no, otherwise NONE>
VALUE_CORRECT:     yes | no
VALUE_COMPLETE:    yes | no
HALLUCINATED:      yes | no
VERDICT:           high | medium | low
CORRECTED_VALUE:   <when VERDICT is medium: the complete/correct value from the
                   paper text that this annotation SHOULD be. When VERDICT is
                   high or low: output NONE>

Field meanings:
  TYPE_CORRECT      = yes when the candidate value belongs to the correct annotation type.
                      no ONLY when its fundamental nature is that of a different type.
                      A method of the wrong kind -> TYPE_CORRECT: yes (not a type mismatch).
  CORRECT_TYPE_NAME = the TypeName the value actually belongs to, when TYPE_CORRECT is no.
                      Must be one of the defined annotation types.
                      Output NONE when TYPE_CORRECT is yes.
  VALUE_CORRECT     = yes when present in source text, type is correct, and factually valid.
                      no for hallucinations, type mismatches, contradictions, wrong kind.
  VALUE_COMPLETE    = yes when the value is fully specified.
                      no only for single-channel stand-ins in a multi-channel scheme.
  HALLUCINATED      = yes ONLY when the candidate value is ABSENT from the paper text.
                      If the value appears in the paper but in a different context,
                      set HALLUCINATED: no and VALUE_CORRECT: no (rule D).
  VERDICT           = overall high/medium/low per the rules above.
  CORRECTED_VALUE   = when VERDICT is medium, the complete or correct value
                      from the paper text. For supertype drift, give the more
                      specific term. For single-channel, give the full multi-
                      channel label. Output NONE for high and low verdicts.

Typical couplings:
  HIGH verdict              -> TYPE_CORRECT yes, VALUE_CORRECT yes, VALUE_COMPLETE yes, CORRECTED_VALUE NONE
  MEDIUM (drift)            -> TYPE_CORRECT yes, VALUE_CORRECT yes, VALUE_COMPLETE yes, CORRECTED_VALUE <corrected>
  MEDIUM (single chan)      -> TYPE_CORRECT yes, VALUE_CORRECT yes, VALUE_COMPLETE no,  CORRECTED_VALUE <corrected>
  LOW hallucination         -> TYPE_CORRECT yes,  VALUE_CORRECT no, HALLUCINATED yes,   CORRECTED_VALUE NONE
  LOW type mismatch         -> TYPE_CORRECT no,   VALUE_CORRECT no,                     CORRECTED_VALUE NONE
  LOW wrong kind / incorrect-> TYPE_CORRECT yes,  VALUE_CORRECT no, HALLUCINATED no,    CORRECTED_VALUE NONE

=================================================================
WORKED EXAMPLES (verify)
=================================================================

Example 1 -- HIGH: verbose phrasing, valid and present
annotation_type=AcquisitionMethod,
candidate_value="A data dependent mass spectrometric method"

TYPE CHECK:        Describes an MS acquisition scheme -- correct type.
SOURCE CHECK:      Present explicitly in the methods section.
TRUTH CHECK:       The paper used data-dependent acquisition; factually correct.
COMPLETENESS CHECK:Complete -- covers the full acquisition concept.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

Example 2 -- LOW (rule D, wrong kind): biological fractionation labeled as FractionationMethod
annotation_type=FractionationMethod,
candidate_value="Subcellular fractionation"

TYPE CHECK:        "Subcellular fractionation" is a process of isolating
                   cellular compartments -- it has a fractionation-like
                   character but FractionationMethod requires OFF-LINE
                   PEPTIDE/PROTEIN fractionation of the digest, not
                   biological separation of cell compartments.
                   TYPE_CORRECT: yes (it is a process, not an object/chemical),
                   but the wrong KIND -- this is rule D, not rule B.
SOURCE CHECK:      Present in the sample-preparation section.
TRUTH CHECK:       Factually the paper did perform subcellular fractionation,
                   but this does not satisfy the definition of FractionationMethod.
COMPLETENESS CHECK:Not applicable -- wrong kind.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           low

Example 3 -- LOW (rule B, true type mismatch): structural compartment labeled as FractionationMethod
annotation_type=FractionationMethod,
candidate_value="nucleus"

TYPE CHECK:        "nucleus" is a structural cellular compartment, not a
                   fractionation method or process. The correct type is CellPart.
SOURCE CHECK:      Present in the paper.
TRUTH CHECK:       Not applicable -- wrong fundamental type.
COMPLETENESS CHECK:Not applicable -- wrong fundamental type.
TYPE_CORRECT:      no
CORRECT_TYPE_NAME: CellPart
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           low

Example 4 -- LOW (rule A): hallucinated -- absent from paper text
annotation_type=EnrichmentMethod,
candidate_value="IMAC phosphopeptide enrichment"

TYPE CHECK:        Enrichment protocol -- correct type.
SOURCE CHECK:      No mention of IMAC or phosphopeptide enrichment in the paper.
TRUTH CHECK:       Cannot be valid -- not described in the paper.
COMPLETENESS CHECK:Not applicable -- hallucinated.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      yes
VERDICT:           low

Example 5 -- HIGH: list of values for same annotation type
annotation_type=FragmentMassTolerance,
candidate_value="20, 10, and 7 ppm"

TYPE CHECK:        All values describe fragment mass tolerances -- correct type.
SOURCE CHECK:      Present in the Data analysis section.
TRUTH CHECK:       The paper states these exact tolerance values; factually correct.
COMPLETENESS CHECK:Complete -- reports all tolerance values stated in the paper.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

Example 6 -- MEDIUM: single channel of a multi-channel scheme
annotation_type=Label, candidate_value="heavy"

TYPE CHECK:        Isotope label -- correct type.
SOURCE CHECK:      "heavy" channel is mentioned in the paper.
TRUTH CHECK:       Correct that a heavy channel was used, but the paper uses
                   triple SILAC -- only one arm named.
COMPLETENESS CHECK:Incomplete -- names only one channel of a three-channel scheme.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   triple SILAC

Example 7 -- LOW (rule B, true type mismatch): embedding medium labeled as Specimen
annotation_type=Specimen,
candidate_value="OCT (optimal cutting temperature compound) blocks"

TYPE CHECK:        OCT is an embedding medium / chemical compound, NOT a
                   biological specimen. The correct type is Compound.
SOURCE CHECK:      Present in the sample-prep section as the embedding medium.
TRUTH CHECK:       Not applicable -- wrong fundamental type.
COMPLETENESS CHECK:Not applicable -- wrong fundamental type.
TYPE_CORRECT:      no
CORRECT_TYPE_NAME: Compound
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           low

Example 8 -- HIGH: FactorValue correctly identifies experimental variable
annotation_type=FactorValue,
candidate_value="prostate cancer"

TYPE CHECK:        Names a disease condition that defines a distinct sample group
                   in a differential proteomics study -- correct type for FactorValue.
SOURCE CHECK:      Present in the paper as a sample group descriptor.
TRUTH CHECK:       The paper compares prostate cancer vs benign prostate hyperplasia
                   samples; "prostate cancer" is a systematically varied condition.
COMPLETENESS CHECK:Complete -- names a specific factor value.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

Example 9 -- LOW (rule D): sample identifier labeled as FactorValue
annotation_type=FactorValue,
candidate_value="M1"

TYPE CHECK:        "M1" is a sample identifier or manufacturer code, not a
                   study variable that defines what differs biologically between
                   sample groups. It names the sample, not the condition.
                   TYPE_CORRECT: yes (it could be a factor label in some context),
                   but VALUE_CORRECT: no -- it is not a meaningful experimental factor.
SOURCE CHECK:      Present in the paper as a sample label.
TRUTH CHECK:       M1 is a sample identifier, not a biologically meaningful experimental
                   variable. The study is descriptive with no controlled comparison.
COMPLETENESS CHECK:Not applicable -- wrong kind of value for FactorValue.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           low

Example 10 -- HIGH: FactorValue overlapping with Time annotation
annotation_type=FactorValue,
candidate_value="4 h"

TYPE CHECK:        A time point that defines a distinct proteomics sample group
                   in a time-course experiment -- correct type for FactorValue.
                   The fact that this also appears as a Time annotation is expected
                   and NOT a reason to downgrade.
SOURCE CHECK:      Present in the paper as a stimulation time point.
TRUTH CHECK:       The paper harvests cells at 4 h and 24 h post-stimulation for
                   proteomics -- these time points define sample groups.
COMPLETENESS CHECK:Complete.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

Example 10b -- MEDIUM: "SILAC" when paper describes "triple SILAC" (supertype drift)
annotation_type=Label, candidate_value="SILAC"

TYPE CHECK:        Metabolic labeling strategy -- correct type.
SOURCE CHECK:      "SILAC" is mentioned in the paper.
TRUTH CHECK:       The paper did use SILAC, but specifically describes a
                   "triple SILAC" approach. "SILAC" is the broader supertype.
COMPLETENESS CHECK:Incomplete -- "SILAC" does not specify the number of channels.
                   The paper uses "triple SILAC" which is more specific.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   triple SILAC

Example 10c -- HIGH: plant cultivar as Strain
annotation_type=Strain, candidate_value="Nipponbare"

TYPE CHECK:        "Nipponbare" is a rice cultivar/strain -- Strain includes
                   animal strains, plant cultivars, and microbial strains.
SOURCE CHECK:      Present in the methods section.
TRUTH CHECK:       The paper uses Nipponbare rice; factually correct.
COMPLETENESS CHECK:Complete.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

Example 10d -- HIGH: parasite life stage as OrganismPart
annotation_type=OrganismPart, candidate_value="tachyzoites"

TYPE CHECK:        "Tachyzoites" is a morphological form / life stage of
                   Toxoplasma gondii used as the biological source material.
                   OrganismPart includes life stages and morphological forms.
SOURCE CHECK:      Present throughout the paper.
TRUTH CHECK:       The paper uses tachyzoites as the source material; correct.
COMPLETENESS CHECK:Complete.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

Example 10e -- HIGH: individual gradient segment time
annotation_type=GradientTime, candidate_value="24 min"

TYPE CHECK:        A time duration from the gradient description -- correct type.
SOURCE CHECK:      Present in the LC-MS section as a gradient segment.
TRUTH CHECK:       The paper describes "24 min" as part of the gradient;
                   individual segment times are valid GradientTime values.
COMPLETENESS CHECK:Complete -- individual segments are valid extractions.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high

=================================================================
ANNOTATION TYPE DEFINITIONS
=================================================================
""" + _ENTITY_DEFINITIONS



_ANNOTATION_GEVAL_CRITERIA = (
    "Assess whether a candidate value is a valid annotation for the given type, "
    "based solely on the paper text and annotation-type definitions. "
    "Evaluate in order: "
    "(1) TYPE: does it match the annotation_type definition in FUNDAMENTAL NATURE? "
    "    A method of the wrong KIND is TYPE_CORRECT: yes but VALUE_CORRECT: no (rule D). "
    "    Only set TYPE_CORRECT: no when the value belongs to a completely different "
    "    annotation category (rule B). "
    "    For FactorValue: sample identifiers, abstract category names, and labeling "
    "    strategies are NOT valid FactorValues. Specific conditions (disease states, "
    "    doses, time points, tissue names, cell types) ARE valid when they define sample groups. "
    "    FactorValues commonly overlap with CellType, Disease, Compound, Time, etc. -- this is correct. "
    "    Strain includes animal strains, plant cultivars, and microbial strains. "
    "    OrganismPart includes life stages and morphological forms (tachyzoites, sporozoites). "
    "    GradientTime accepts individual gradient segment durations, not just total time. "
    "(2) SOURCE: is it present in the paper text? "
    "    Absent -> HALLUCINATED: yes -> low. "
    "    Present but in wrong context -> HALLUCINATED: no, VALUE_CORRECT: no (rule D). "
    "(3) TRUTH: is it factually correct and valid for this experiment per the paper "
    "    text AND the type definition? "
    "    Wrong kind within correct category -> VALUE_CORRECT: no -> low (rule D). "
    "    For FactorValue: value must genuinely be varied across proteomics sample groups. "
    "(4) COMPLETENESS: single-channel stand-in for multi-channel scheme -> medium. "
    "    Subtype or supertype drift -> medium. "
    "    IMPORTANT for Label: 'SILAC' alone when paper describes 'triple SILAC' is "
    "    supertype drift -> VALUE_COMPLETE: no -> VERDICT: medium. "
    "HIGH = correct type + present + factually valid + complete. "
    "MEDIUM = correct type + present + valid, but single-channel, supertype drift, or subtype drift. "
    "LOW = hallucinated (absent from text), true type mismatch (rule B), or "
    "      factually incorrect / wrong kind (rule D). "
    "NEVER downgrade for verbose phrasing, lists of values of the same type, "
    "abbreviation expansion, unit spacing, enzyme Pro-rule variants, "
    "individual gradient segment times, plant cultivar as Strain, "
    "life stages as OrganismPart, or "
    "FactorValue overlapping with Time/Temperature/Compound/CellType/Disease/etc."
)

_ANNOTATION_GEVAL_STEPS = [
    "TYPE CHECK: Ask 'What is the fundamental nature of this value (chemical, species, "
    "structural compartment, method/process, instrument, count, study variable, etc.)?' "
    "If its nature belongs to a completely different annotation category, state the correct "
    "type and set TYPE_CORRECT: no -> VERDICT: low (rule B). "
    "If it is a method/process of the RIGHT CATEGORY but wrong KIND for this annotation type "
    "(e.g. biological fractionation for FractionationMethod, which requires off-line peptide "
    "fractionation), set TYPE_CORRECT: yes -- this is rule D, not rule B. "
    "For FactorValue: check if the value names a study variable or experimental condition "
    "that is systematically varied across proteomics sample groups. "
    "FactorValues that overlap with CellType, Disease, Compound, Time, etc. are CORRECT. "
    "Strain includes ALL organism strains: animal, plant cultivars, microbial. "
    "OrganismPart includes life stages (tachyzoites, sporozoites). "
    "GradientTime accepts individual gradient segment durations. "
    "Biological conditions (sulfur-depleted) are NOT Depletion methods.",

    "SOURCE CHECK: Confirm the candidate value (or domain-equivalent phrasing) is present "
    "anywhere in the paper text. If ABSENT: HALLUCINATED: yes -> VERDICT: low (rule A). "
    "If PRESENT but used in a different context than this annotation type requires: "
    "HALLUCINATED: no, VALUE_CORRECT: no (rule D). "
    "A sentence-form value is present if all its factual content appears in the paper.",

    "TRUTH CHECK: Assess whether the candidate value is factually correct AND fits the "
    "precise definition of the annotation type for this experiment. "
    "A value that is the right CATEGORY but wrong KIND fails: VALUE_CORRECT: no -> low. "
    "For Label: 'SILAC' alone when paper uses 'triple SILAC' -> VALUE_CORRECT: yes but "
    "VALUE_COMPLETE: no (supertype drift -> MEDIUM). They are NOT equivalent.",

    "COMPLETENESS CHECK: Flag MEDIUM only for (a) subtype drift, (b) supertype drift, or "
    "(c) single-channel stand-in for a confirmed multi-channel scheme. "
    "IMPORTANT: 'SILAC' when paper describes 'triple SILAC' is supertype drift -> "
    "VALUE_COMPLETE: no -> VERDICT: medium. "
    "Do NOT flag medium for verbose phrasing or lists of values of the same type.",

    "Output TYPE CHECK / SOURCE CHECK / TRUTH CHECK / COMPLETENESS CHECK, "
    "TYPE_CORRECT (yes/no), CORRECT_TYPE_NAME (TypeName or NONE), "
    "VALUE_CORRECT (yes/no), VALUE_COMPLETE (yes/no), "
    "HALLUCINATED (yes/no), and VERDICT: high | medium | low. "
    "When VERDICT is medium, also output CORRECTED_VALUE: <the complete/correct value>.",
]


_METHOD_SIGNALS = re.compile(
    r"\b(fractionation|separation|digestion|enrichment|depletion|labeling|"
    r"staining|isolation|extraction|preparation|treatment|acquisition|"
    r"fragmentation|ionization|chromatography|electrophoresis|purification|"
    r"centrifugation|lysis|homogenization|solubilization|precipitation)\b",
    re.IGNORECASE,
)

_OBJECT_ENTITY_TYPES = {
    "cellpart", "organismpart", "celltype", "cellline", "specimen",
    "organism", "strain", "genotype", "disease", "materialtype",
    "anatomicsitetumor", "tumorsite", "originsitedisease",
    "sex", "age", "bmi", "developmentalstage", "ancestry",
}



#disk response cache
class DiskResponseCache:

    def __init__(self, cache_dir: str, enabled: bool = True):
        self._dir = Path(cache_dir)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()         
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(paper_id: str, task: str, annotation_type: str, primary_value: str) -> str:
        blob = f"{paper_id}|{task}|{annotation_type.lower()}|{primary_value}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, paper_id: str, task: str, annotation_type: str, primary_value: str) -> str | None:
        if not self._enabled:
            return None
        key = self._make_key(paper_id, task, annotation_type, primary_value)
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

    def put(
        self,
        paper_id: str,
        task: str,
        annotation_type: str,
        primary_value: str,
        response: str,
    ) -> None:
        if not self._enabled:
            return
        key = self._make_key(paper_id, task, annotation_type, primary_value)
        path = self._dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps({
                    "key": key,
                    "paper_id": paper_id,
                    "task": task,
                    "annotation_type": annotation_type,
                    "primary_value": primary_value,
                    "response": response,
                    "timestamp": time.time(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"  [cache-write-error] {exc}")

    def stats(self) -> dict:
        n_files = sum(1 for _ in self._dir.glob("*.json")) if self._enabled else 0
        total = self._hits + self._misses
        return {
            "enabled": self._enabled,
            "cache_dir": str(self._dir),
            "cached_responses_on_disk": n_files,
            "session_hits": self._hits,
            "session_misses": self._misses,
            "session_hit_rate": f"{self._hits / total * 100:.1f}%" if total > 0 else "n/a",
        }

#prompt catch message builder

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
                        "PAPER TEXT (abstract + methods) — use this as the source of truth "
                        "for all SOURCE CHECK and TRUTH CHECK assessments in this conversation:\n\n"
                        + paper_text
                    ),
                    "cache_control": {"type": "ephemeral"},  
                }
            ],
        },
        {
            "role": "assistant",
            "content": (
                "Understood. I have read the paper text and will assess all annotation "
                "values against it using the definitions and rules in my instructions."
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
    """
    Handles three cases:
      1. raw message is None            -> return ""
      2. raw message.content is a list  -> join all text type blocks
      3. raw message.content is a str   -> return it (or "" if None)
    """
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

class Gemma4OpenRouter(DeepEvalBaseLLM):
    def __init__(self):
        self._model_name = EVALUATION_MODEL
        self._client: openai.OpenAI | None = None
        self._paper_text: str = ""
        self._current_paper_id: str = ""
        self._response_cache = DiskResponseCache(CACHE_DIR, enabled=CACHE_ENABLED)
        self._call_count = 0
        self._api_call_count = 0
        self._disk_hit_count = 0
        self._lock = threading.Lock()        

    def set_paper_context(self, source_text: str, paper_id: str = "") -> None:
        self._paper_text = source_text
        self._current_paper_id = paper_id

    def set_entity_context(self, context: dict) -> None:
        _thread_local.current_context = context

    def load_model(self) -> openai.OpenAI:
        if self._client is None:
            self._client = _build_openrouter_client()
        return self._client

    def _generate_single(
        self,
        paper_text: str,
        paper_id: str,
        context: dict,
    ) -> str:
        with self._lock:
            self._call_count += 1
            call_num = self._call_count

        task            = context.get("task", "verify")
        annotation_type = context.get("annotation_type", "")
        primary_value   = context.get("candidate_value", "")

        cached = self._response_cache.get(
            paper_id, task, annotation_type, primary_value
        )
        if cached is not None:
            with self._lock:
                self._disk_hit_count += 1
            print(
                f"  [call #{call_num}] DISK CACHE HIT  "
                f"paper={paper_id!r}  task={task}  "
                f"type={annotation_type}  value={primary_value[:40]!r}"
            )
            return cached

        with self._lock:
            self._api_call_count += 1

        messages = _build_messages(paper_text, context)

        system_chars = len(_STATIC_SYSTEM_PROMPT)
        paper_chars  = len(paper_text)
        entity_chars = len(json.dumps(context))
        print(
            f"  [call #{call_num}] API CALL  "
            f"paper={paper_id!r}  task={task}  "
            f"type={annotation_type}  value={primary_value[:40]!r}  "
            f"[sys={system_chars:,}  paper={paper_chars:,}  entity={entity_chars:,} chars]"
        )

        content = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.load_model().chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=MODEL_TEMPERATURE,
                    max_tokens=32000,
                    extra_body=(
                        {"thinking": {"type": "enabled"}}
                        if MODEL_ENABLE_THINKING
                        else {}
                    ),
                    timeout=300,
                )

                if (
                    response is None
                    or not hasattr(response, "choices")
                    or not response.choices
                ):
                    if attempt < MAX_RETRIES:
                        print(
                            f"  [WARNING] API returned empty/null response "
                            f"(attempt {attempt}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s..."
                        )
                        time.sleep(RETRY_DELAY * attempt) 
                        continue
                    else:
                        print(
                            f"  [WARNING] API returned empty/null response "
                            f"(attempt {attempt}/{MAX_RETRIES}), giving up."
                        )
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
                        print(
                            f"  [tokens] prompt={usage.prompt_tokens:,}  "
                            f"completion={usage.completion_tokens:,}{cache_info}"
                        )

                    if content:
                        break  # success
                    elif attempt < MAX_RETRIES:
                        print(
                            f"  [WARNING] API returned empty text content "
                            f"(attempt {attempt}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s..."
                        )
                        time.sleep(RETRY_DELAY * attempt)
                        continue
                    else:
                        print(
                            f"  [WARNING] API returned empty text content "
                            f"(attempt {attempt}/{MAX_RETRIES}), giving up."
                        )

            except Exception as api_exc:
                if attempt < MAX_RETRIES:
                    print(
                        f"  [API ERROR] {type(api_exc).__name__}: {api_exc} "
                        f"(attempt {attempt}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s..."
                    )
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    print(
                        f"  [API ERROR] {type(api_exc).__name__}: {api_exc} "
                        f"(attempt {attempt}/{MAX_RETRIES}), giving up."
                    )

        #cache non empty responses
        if content:
            self._response_cache.put(
                paper_id, task, annotation_type,
                primary_value, content,
            )

        return content

    def generate(self, prompt: str, schema=None):
        context = getattr(_thread_local, "current_context", None) or {}
        content = self._generate_single(
            self._paper_text, self._current_paper_id, context
        )

        if not content:
            content = (
                "TYPE CHECK: Unable to evaluate (no model response).\n"
                "SOURCE CHECK: N/A\n"
                "TRUTH CHECK: N/A\n"
                "COMPLETENESS CHECK: N/A\n"
                "TYPE_CORRECT: yes\n"
                "CORRECT_TYPE_NAME: NONE\n"
                "VALUE_CORRECT: no\n"
                "VALUE_COMPLETE: no\n"
                "HALLUCINATED: no\n"
                "VERDICT: low"
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
        suffix = "-thinking" if MODEL_ENABLE_THINKING else ""
        return f"{self._model_name}{suffix}"

    def get_cache_stats(self) -> dict:
        stats = self._response_cache.stats()
        stats["total_generate_calls"] = self._call_count
        stats["actual_api_calls"]     = self._api_call_count
        stats["disk_cache_hits"]      = self._disk_hit_count
        stats["api_calls_saved"]      = self._disk_hit_count
        if self._call_count > 0:
            stats["overall_disk_hit_rate"] = (
                f"{self._disk_hit_count / self._call_count * 100:.1f}%"
            )
        return stats


#GEval metrics

_gemma4_model: Gemma4OpenRouter | None = None


def _get_judge_model() -> Gemma4OpenRouter:
    global _gemma4_model
    if _gemma4_model is None:
        _gemma4_model = Gemma4OpenRouter()
    return _gemma4_model


def _get_thread_metric() -> GEval:
    if not hasattr(_thread_local, "metric"):
        _thread_local.metric = GEval(
            name="AnnotationJudge",
            criteria=_ANNOTATION_GEVAL_CRITERIA,
            evaluation_steps=_ANNOTATION_GEVAL_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=_get_judge_model(),
            threshold=0.5,
        )
    return _thread_local.metric


def _type_def_for(annotation_type: str) -> str:
    at_lower = annotation_type.lower()
    for line in _ENTITY_DEFINITIONS.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(":", 1)
        if len(parts) >= 1 and parts[0].strip().lower() == at_lower:
            return stripped
    return f"Annotation type: {annotation_type}"


_KNOWN_TYPES_LOWER = {
    line.strip().split(":")[0].strip().lower()
    for line in _ENTITY_DEFINITIONS.strip().splitlines()
    if ":" in line
    and not line.strip().startswith("SAMPLE")
    and not line.strip().startswith("TECHNICAL")
    and line.strip().split(":")[0].strip()
}


def _parse_type_correct(reason: str) -> bool | None:
    m = re.search(r"TYPE_CORRECT\s*:\s*(yes|no)\b", reason, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "yes"
    m2 = re.search(r"CORRECT_TYPE\s*:\s*(\S+)", reason, re.IGNORECASE)
    if m2:
        val = m2.group(1).strip().upper()
        return val in ("SAME", "YES", "TRUE")
    return None


def _parse_correct_type_name(reason: str, annotation_type: str = "") -> str | None:
    at_lower = annotation_type.strip().lower()

    def _accept(name: str) -> str | None:
        n = name.strip().strip("'\"`,.()")
        if n.lower() in _KNOWN_TYPES_LOWER and n.lower() != at_lower:
            return n[0].upper() + n[1:] if n else None
        return None

    m = re.search(r"CORRECT_TYPE_NAME\s*:\s*(\S+)", reason, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip("'\"`,.")
        if val.upper() not in ("NONE", "N/A", "NULL", "SAME", "YES", "NO"):
            result = _accept(val)
            if result:
                return result

    prose_patterns = [
        r"correct\s+type\s+is\s+([A-Z][A-Za-z]+)",
        r"correct\s+type:\s*([A-Z][A-Za-z]+)",
        r"correct\s+type\s+(?:would\s+be|should\s+be)\s+([A-Z][A-Za-z]+)",
        r"belongs\s+to\s+(?:the\s+)?([A-Z][A-Za-z]+)",
        r"should\s+be\s+(?:labeled\s+as\s+|classified\s+as\s+|tagged\s+as\s+)?([A-Z][A-Za-z]+)",
        r"type\s+is\s+([A-Z][A-Za-z]+)",
        r"annotated\s+as\s+([A-Z][A-Za-z]+)",
    ]
    for pat in prose_patterns:
        m2 = re.search(pat, reason, re.IGNORECASE)
        if m2:
            result = _accept(m2.group(1))
            if result:
                return result

    type_check_block = ""
    tc_match = re.search(
        r"TYPE\s*CHECK\s*:\s*(.+?)(?=SOURCE\s*CHECK|TRUTH\s*CHECK|$)",
        reason, re.IGNORECASE | re.DOTALL,
    )
    if tc_match:
        type_check_block = tc_match.group(1)
    for paren_match in re.finditer(r"\(([A-Z][A-Za-z]+)\)", type_check_block):
        result = _accept(paren_match.group(1))
        if result:
            return result

    signal_pattern = re.compile(
        r"(?:correct\s+type|wrong\s+type|type\s+mismatch|belongs\s+to|should\s+be"
        r"|not\s+a|incorrect\s+type|right\s+type)",
        re.IGNORECASE,
    )
    for signal_match in signal_pattern.finditer(reason):
        window = reason[signal_match.start(): signal_match.start() + 80]
        for word_match in re.finditer(r"[A-Z][A-Za-z]+", window):
            result = _accept(word_match.group(0))
            if result:
                return result

    return None


def _parse_yes_no(reason: str, field: str) -> bool | None:
    m = re.search(rf"{field}\s*:\s*(yes|no)\b", reason, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower() == "yes"


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s.strip().lower())


def _apply_mismatch_veto(
    extracted_value: str,
    is_mismatch: bool,
    correct_type: str | None,
    annotation_type: str,
    reason: str,
) -> tuple[bool, str | None]:
    if not is_mismatch or correct_type is None:
        return is_mismatch, correct_type

    ct_lower = correct_type.lower()
    at_lower = annotation_type.strip().lower()

    if ct_lower == "factorvalue":
        print(
            f"  [mismatch-veto:factorvalue-overlap] '{extracted_value[:55]}' "
            f"flagged as FactorValue but values can be both {annotation_type} "
            f"and FactorValue -- demoting to wrong-value low (rule D)."
        )
        return False, None

    _OVERLAPPING_BIO_TYPES = {"organismpart", "celltype", "cellline", "specimen"}
    if at_lower in _OVERLAPPING_BIO_TYPES and ct_lower in _OVERLAPPING_BIO_TYPES:
        print(
            f"  [mismatch-veto:bio-overlap] '{extracted_value[:55]}' "
            f"both annotation_type={annotation_type} and correct_type={correct_type} "
            f"are biological entity types -- demoting to wrong-value low (rule D)."
        )
        return False, None

    if ct_lower in _OBJECT_ENTITY_TYPES and _METHOD_SIGNALS.search(extracted_value):
        print(
            f"  [mismatch-veto:method-as-object] '{extracted_value[:55]}' "
            f"flagged as {correct_type} but reads as a method/process "
            f"-- demoting to wrong-value low (rule D)."
        )
        return False, None

    _METHOD_ANN_TYPES = {
        "fractionationmethod", "enrichmentmethod", "separationmethod",
        "fragmentationmethod", "acquisitionmethod", "cleavageagent",
        "reductionreagent", "alkylationreagent", "depletion",
    }
    if at_lower in _METHOD_ANN_TYPES and ct_lower in _METHOD_ANN_TYPES:
        print(
            f"  [mismatch-veto:same-category] '{extracted_value[:55]}' "
            f"both annotation_type={annotation_type} and correct_type={correct_type} "
            f"are method types -- demoting to wrong-value low (rule D)."
        )
        return False, None

    if at_lower == "strain" and ct_lower in ("genotype", "organism", "cellline"):
        print(
            f"  [mismatch-veto:strain-broad] '{extracted_value[:55]}' "
            f"Strain includes plant cultivars/microbial strains "
            f"-- demoting to wrong-value low (rule D)."
        )
        return False, None

    return is_mismatch, correct_type


def _parse_corrected_value(reason: str) -> str | None:
    """
    Parse correct values
    """
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

    #focus on the completeness
    check_blocks = []
    for block_name in ["COMPLETENESS CHECK", "TRUTH CHECK"]:
        block_match = re.search(
            rf"{block_name}\s*:\s*(.+?)(?=TYPE_CORRECT|VALUE_CORRECT|VALUE_COMPLETE|HALLUCINATED|VERDICT|CORRECTED_VALUE|$)",
            reason, re.IGNORECASE | re.DOTALL,
        )
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


def _run_geval_semantic(annotation_type: str, extracted_value: str) -> dict:
    """
    evaluate single predicted annotation against the paper text only
    """
    model = _get_judge_model()

    context = {
        "task": "verify",
        "annotation_type": annotation_type,
        "type_definition": _type_def_for(annotation_type),
        "candidate_value": extracted_value,
    }

    #store context in thread local
    model.set_entity_context(context)

    expected_str = (
        "Assess the candidate_value against the paper text AND the type definition: "
        "correct annotation type in FUNDAMENTAL NATURE (TYPE_CORRECT yes/no), "
        "present in source (HALLUCINATED yes/no), "
        "factually valid and correct KIND for this type (VALUE_CORRECT yes/no), "
        "complete (VALUE_COMPLETE yes/no). "
        "A method of the wrong KIND is TYPE_CORRECT yes, VALUE_CORRECT no (rule D -- not mismatch). "
        "Only TYPE_CORRECT no when fundamental nature belongs to a different category (rule B). "
        "HALLUCINATED yes only when absent from paper text -- if present but wrong context, "
        "HALLUCINATED no, VALUE_CORRECT no. "
        "NEVER downgrade for verbose phrasing or lists of same-type values. "
        "Strain includes animal strains, plant cultivars, and microbial strains. "
        "OrganismPart includes life stages (tachyzoites, sporozoites). "
        "GradientTime accepts individual gradient segment durations. "
        "Biological conditions (sulfur-depleted) are NOT Depletion methods. "
        "For Label: 'SILAC' alone when paper uses 'triple SILAC' -> supertype drift -> MEDIUM. "
        "For FactorValue: value must name a systematically varied condition across proteomics "
        "sample groups. Overlap with CellType/Disease/Compound/Time/etc. is allowed and expected. "
        "Verdict: high (all pass) | medium (drift/single-channel) | low (absent/mismatch/wrong). "
        "When VERDICT is medium, also output CORRECTED_VALUE: <the complete/correct value "
        "from the paper text that this annotation should be>."
    )
    test_case = LLMTestCase(
        input=f"Assess annotation_type='{annotation_type}' candidate_value='{extracted_value}'",
        actual_output=f"annotation_type: {annotation_type}\ncandidate_value: {extracted_value}",
        expected_output=expected_str,
    )

    reason = ""
    try:
        metric = _get_thread_metric()
        metric.measure(test_case)
        reason = metric.reason or ""
    except RecursionError:
        print(
            f"  [GEval recursion] {annotation_type}: '{extracted_value[:40]}' "
            f"-- falling back to direct API call"
        )
        reason = model._generate_single(
            model._paper_text, model._current_paper_id, context
        )
        if not reason:
            reason = (
                "TYPE CHECK: Unable to evaluate (no model response).\n"
                "SOURCE CHECK: N/A\nTRUTH CHECK: N/A\nCOMPLETENESS CHECK: N/A\n"
                "TYPE_CORRECT: yes\nCORRECT_TYPE_NAME: NONE\n"
                "VALUE_CORRECT: no\nVALUE_COMPLETE: no\n"
                "HALLUCINATED: no\nVERDICT: low\nCORRECTED_VALUE: NONE"
            )
        print(f"  [direct-fallback response] {reason[:200]}...")
    except Exception as exc:
        print(f"  [GEval error] {annotation_type}: {type(exc).__name__}: {exc}")
        #try direct API as fallback for other GEval errors
        try:
            reason = model._generate_single(
                model._paper_text, model._current_paper_id, context
            )
            if reason:
                print(
                    f"  [GEval->direct fallback] Got response for {annotation_type}: "
                    f"'{extracted_value[:40]}'"
                )
            else:
                raise ValueError("Empty response from direct fallback")
        except Exception as fallback_exc:
            print(f"  [direct fallback also failed] {fallback_exc}")
            return {
                "verdict": None, "type_mismatch": None, "correct_type": None,
                "value_correct": None, "value_complete": None, "hallucination": None,
                "corrected_value": None,
                "issue_summary": f"GEval error: {exc}", "match_type": "ERROR",
            }

    #structured fields from reason
    verdict_match   = re.search(r"VERDICT:\s*(high|medium|low)", reason, re.IGNORECASE)
    verdict         = verdict_match.group(1).lower() if verdict_match else None
    hall_parsed     = _parse_yes_no(reason, "HALLUCINATED")
    is_hallucinated = bool(hall_parsed) if hall_parsed is not None else False

    type_correct_parsed = _parse_type_correct(reason)
    if type_correct_parsed is not None:
        is_mismatch = not type_correct_parsed
    else:
        is_mismatch = (verdict == "low") and not is_hallucinated and bool(
            re.search(r"type mismatch|wrong type|incorrect type|belongs to", reason, re.IGNORECASE)
        )

    correct_type = _parse_correct_type_name(reason, annotation_type) if is_mismatch else None
    is_mismatch, correct_type = _apply_mismatch_veto(
        extracted_value, is_mismatch, correct_type, annotation_type, reason
    )

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

    #Parse corrected value for medium
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

    return {
        "verdict": verdict,
        "type_mismatch": is_mismatch,
        "correct_type": correct_type,
        "value_correct": value_correct,
        "value_complete": value_complete,
        "hallucination": is_hallucinated,
        "corrected_value": corrected_value,
        "issue_summary": reason,
        "match_type": match_type,
    }

#evaluation with GEval

def evaluate_with_geval(
    paper_id: str,
    predicted_anns: dict,
    source_text: str,
) -> dict:
    """Evaluate all predicted annotations"""
    if not source_text or not predicted_anns:
        return {}

    _get_judge_model().set_paper_context(source_text, paper_id=paper_id)

    predicted_anns = {k.lower(): v for k, v in predicted_anns.items()}

    tasks = []
    for ann_type, values in predicted_anns.items():
        if not isinstance(values, list):
            values = [values]
        for val in values:
            val = str(val).strip()
            if val and val.lower() != "unknown":
                tasks.append((ann_type, val))

    if not tasks:
        return {}

    n_total = len(tasks)
    print(f"  Evaluating {n_total} annotations with {MAX_WORKERS} workers (GEval)...")

    results_map: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, (ann_type, val) in enumerate(tasks):
            future = executor.submit(_run_geval_semantic, ann_type, val)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            ann_type, val = tasks[idx]
            try:
                result = future.result()
                results_map[idx] = result
                if result.get("verdict") is not None:
                    cv_info = ""
                    if result.get("corrected_value"):
                        cv_info = f"  corrected={result['corrected_value'][:30]}"
                    print(
                        f"  [{idx+1}/{n_total}] {ann_type}: '{val[:40]}' "
                        f"-> verdict={result['verdict']}  correct={result['value_correct']}  "
                        f"complete={result['value_complete']}  mismatch={result['type_mismatch']}  "
                        f"halluc={result['hallucination']}{cv_info}"
                    )
                else:
                    print(f"  [{idx+1}/{n_total}] {ann_type}: '{val[:40]}' -> ERROR")
            except Exception as exc:
                print(f"  [{idx+1}/{n_total}] {ann_type}: '{val[:40]}' -> ERROR: {exc}")
                results_map[idx] = {
                    "verdict": None, "type_mismatch": None, "correct_type": None,
                    "value_correct": None, "value_complete": None, "hallucination": None,
                    "corrected_value": None,
                    "issue_summary": f"Thread error: {exc}", "match_type": "ERROR",
                }

    per_annotation_review = []
    n_halluc = n_mismatch = n_wrong = n_incompl = 0

    for idx in range(n_total):
        ann_type, val = tasks[idx]
        result = results_map.get(idx, {
            "verdict": None, "type_mismatch": None, "correct_type": None,
            "value_correct": None, "value_complete": None, "hallucination": None,
            "corrected_value": None,
            "issue_summary": "Missing result", "match_type": "ERROR",
        })

        if result.get("hallucination"):         n_halluc   += 1
        elif result.get("type_mismatch"):       n_mismatch += 1
        elif result.get("verdict") == "low":    n_wrong    += 1
        elif result.get("verdict") == "medium": n_incompl  += 1

        per_annotation_review.append({
            "annotation_type":    ann_type,
            "extracted_value":    val,
            "verdict":            result["verdict"],
            "match_type":         result["match_type"],
            "type_mismatch":      result["type_mismatch"],
            "correct_type":       result["correct_type"],
            "value_correct":      result["value_correct"],
            "value_complete":     result["value_complete"],
            "hallucination":      result["hallucination"],
            "corrected_value":    result.get("corrected_value"),
            "issue_summary":      result["issue_summary"],
        })

    n_clean         = sum(
        1 for r in per_annotation_review
        if r.get("verdict") == "high" and not r.get("type_mismatch") and not r.get("hallucination")
    )
    n_value_correct = sum(1 for r in per_annotation_review if r.get("value_correct"))
    n_high          = sum(1 for r in per_annotation_review if r.get("verdict") == "high")
    n_medium        = sum(1 for r in per_annotation_review if r.get("verdict") == "medium")
    n_low           = sum(1 for r in per_annotation_review if r.get("verdict") == "low")
    halluc_rate     = n_halluc / n_total if n_total > 0 else 0.0

    summary = (
        f"{n_total} annotations evaluated: "
        f"{n_high} high, {n_medium} medium, {n_low} low. "
        f"{n_halluc} hallucinated, {n_mismatch} type mismatch, "
        f"{n_wrong} low-verdict (wrong value), {n_incompl} medium-verdict."
    )
    print(
        f"  [summary] high={n_high} medium={n_medium} low={n_low}  "
        f"halluc={n_halluc}  mismatch={n_mismatch}  "
        f"low-verdict(wrong)={n_wrong}  medium-verdict={n_incompl}"
    )
    return {
        "qualitative_summary":   summary,
        "per_annotation_review": per_annotation_review,
        "hallucination_rate":    halluc_rate,
        "type_mismatch_count":   n_mismatch,
        "wrong_value_count":     n_wrong,
        "incomplete_count":      n_incompl,
        "overall_summary":       summary,
        "judge_n_total":         n_total,
        "judge_n_clean":         n_clean,
        "judge_n_value_correct": n_value_correct,
        "judge_n_high":          n_high,
        "judge_n_medium":        n_medium,
        "judge_n_low":           n_low,
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
                value = value.strip()
                if entity_type and value:
                    entities.append((entity_type, value, line_num))
    except Exception as exc:
        print(f"Error parsing {file_path}: {exc}")
    return entities


def should_exclude_entity(entity_type: str) -> bool:
    """No entity types are excluded --> FactorValue is now included in evaluation."""
    return False


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
        "fractionation": "fractionationmethod", "fractionation method": "fractionationmethod",
        "factor value": "factorvalue", "factor_value": "factorvalue",
        "experimentalfactor": "factorvalue", "experimental factor": "factorvalue",
        "experimental_factor": "factorvalue",
        "studyvariable": "factorvalue", "study variable": "factorvalue",
        "study_variable": "factorvalue",
        "experimentalvariable": "factorvalue", "experimental variable": "factorvalue",
        "experimental_variable": "factorvalue",
    }
    if "[" in entity_type:
        entity_type = entity_type.split("[")[0].strip()
    key = entity_type.strip().lower()
    if key in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[key]
    key_nospace = key.replace(" ", "").replace("-", "").replace("_", "")
    if key_nospace in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[key_nospace]
    return entity_type.strip()


def load_papers_dataset(json_path: str) -> dict:
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
                "abstract":      paper.get("abstract") or "",
                "methods":       paper.get("methods") or "",
                "supplementary": paper.get("supplementary") or "",
            }
        print(f"  Loaded {len(papers_dict)} papers")
        return papers_dict
    except FileNotFoundError:
        print(f"Warning: papers_dataset.json not found at: {json_path}")
        return {}
    except Exception as exc:
        print(f"Warning: Could not load papers dataset: {exc}")
        return {}


#CSV writers

_REVIEW_CSV_FIELDS = [
    "paper_id", "annotation_type", "extracted_value", "verdict",
    "match_type", "type_mismatch", "correct_type", "value_correct",
    "value_complete", "hallucination", "corrected_value", "issue_summary",
]
_STATS_CSV_FIELDS = [
    "file_name", "judge_n_total", "judge_n_clean", "judge_n_value_correct",
    "judge_n_high", "judge_n_medium", "judge_n_low",
    "judge_hallucination_rate", "judge_type_mismatch_count",
    "judge_wrong_value_count", "judge_incomplete_count",
]
_FILTER_CSV_FIELDS = ["file_name", "total", "kept", "removed", "corrected"]


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


def _filter_single_paper(
    file_name: str,
    judge_result: dict,
    pred_dir: str,
    filtered_dir: str,
) -> dict:
    """Filter one paper's .ann file immediately after judging. Returns stats dict."""
    low_pairs: set[tuple[str, str]] = set()
    medium_corrections: dict[tuple[str, str], str] = {}

    for review in judge_result.get("per_annotation_review", []):
        ann_type = review.get("annotation_type", "").strip().lower()
        ext_val  = review.get("extracted_value", "").strip()
        if not ann_type or not ext_val:
            continue
        if review.get("verdict") == "low":
            low_pairs.add((ann_type, ext_val))
        elif review.get("verdict") == "medium":
            corrected = review.get("corrected_value")
            if corrected and corrected.strip():
                medium_corrections[(ann_type, ext_val)] = corrected.strip()

    src_path = Path(pred_dir) / f"{file_name}.ann"
    if not src_path.exists():
        print(f"  [filter] WARNING: {src_path} not found, skipping.")
        return {"total": 0, "kept": 0, "removed": 0, "corrected": 0}

    kept_lines = []
    removed_lines = []
    corrected_count = 0
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n\r")
            stripped = raw.strip()
            if not stripped or ":" not in stripped:
                kept_lines.append(raw)
                continue

            entity_type, value = stripped.split(":", 1)
            entity_type = entity_type.strip()
            value = value.strip()

            if not entity_type or not value:
                kept_lines.append(raw)
                continue

            if should_exclude_entity(entity_type):
                kept_lines.append(raw)
                continue

            norm_type = normalize_entity_type(entity_type).lower()
            lookup = (norm_type, value)

            if lookup in low_pairs:
                removed_lines.append(raw)
            elif lookup in medium_corrections:
                corrected_val = medium_corrections[lookup]
                corrected_line = f"{entity_type}: {corrected_val}"
                kept_lines.append(corrected_line)
                corrected_count += 1
                print(
                    f"  [filter:correct] {file_name} {entity_type}: "
                    f"'{value[:40]}' -> '{corrected_val[:40]}'"
                )
            else:
                kept_lines.append(raw)

    out_path = Path(filtered_dir) / f"{file_name}.ann"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    n_total   = len(kept_lines) + len(removed_lines)
    n_kept    = len(kept_lines)
    n_removed = len(removed_lines)
    print(
        f"  [filter] {file_name}.ann: {n_total} lines -> "
        f"kept {n_kept}, removed {n_removed} low-verdict, "
        f"corrected {corrected_count} medium-verdict -> {out_path}"
    )

    return {
        "total": n_total,
        "kept": n_kept,
        "removed": n_removed,
        "corrected": corrected_count,
    }

# Main 

def run_geval_judge(pred_dir: str, papers_json: str):
    """
    Run the GEval judge 
    """
    pred_files = {f.stem: f for f in Path(pred_dir).glob("*.ann")}

    if not pred_files:
        print("No .ann prediction files found!")
        return None, None

    n_files = len(pred_files)
    print(f"Found {n_files} prediction files")
    papers_dict = load_papers_dataset(papers_json)

    os.makedirs(FILTERED_DIR, exist_ok=True)
    review_csv = _IncrementalCSV(
        os.path.join(RESULTS_DIR, "llm_judge_annotation_review.csv"),
        _REVIEW_CSV_FIELDS,
    )
    stats_csv = _IncrementalCSV(
        os.path.join(RESULTS_DIR, "llm_judge_per_file.csv"),
        _STATS_CSV_FIELDS,
    )
    filter_csv = _IncrementalCSV(
        os.path.join(FILTERED_DIR, "_filter_summary.csv"),
        _FILTER_CSV_FIELDS,
    )

    per_paper_judge: dict = {}
    per_file_stats: list  = []

    for file_idx, file_name in enumerate(sorted(pred_files.keys()), 1):
        t_start = time.time()
        print(f"  PAPER [{file_idx}/{n_files}]: {file_name}.ann")

        pred_entities = parse_ann_file(pred_files[file_name])
        pred_filtered = [(t, txt, ln) for t, txt, ln in pred_entities if not should_exclude_entity(t)]

        source_sections = papers_dict.get(file_name, {})
        source_text = "\n\n".join(
            f"### {section.upper()}\n{text}"
            for section, text in source_sections.items() if text
        )
        if not source_text:
            print(f"  WARNING: no source text for {file_name}, skipping.")
            continue

        pred_normalized = [(normalize_entity_type(t), txt, ln, file_name) for t, txt, ln in pred_filtered]

        pred_anns_dict: dict[str, list] = defaultdict(list)
        for et, txt, _, _ in pred_normalized:
            pred_anns_dict[et].append(txt)

        n_pred = sum(len(v) for v in pred_anns_dict.values())
        print(f"  {n_pred} annotations | {len(source_text):,} chars of source text")

        judge_scores = evaluate_with_geval(
            file_name, dict(pred_anns_dict), source_text
        )
        if not judge_scores:
            print(f"  No scores produced for {file_name}, skipping.")
            continue

        per_paper_judge[file_name] = judge_scores

        file_stat = {
            "file_name":                 file_name,
            "judge_n_total":             judge_scores.get("judge_n_total", 0),
            "judge_n_clean":             judge_scores.get("judge_n_clean", 0),
            "judge_n_value_correct":     judge_scores.get("judge_n_value_correct", 0),
            "judge_n_high":              judge_scores.get("judge_n_high", 0),
            "judge_n_medium":            judge_scores.get("judge_n_medium", 0),
            "judge_n_low":               judge_scores.get("judge_n_low", 0),
            "judge_hallucination_rate":  judge_scores.get("hallucination_rate"),
            "judge_type_mismatch_count": judge_scores.get("type_mismatch_count"),
            "judge_wrong_value_count":   judge_scores.get("wrong_value_count"),
            "judge_incomplete_count":    judge_scores.get("incomplete_count", 0),
        }
        per_file_stats.append(file_stat)

        review_rows = [
            {
                "paper_id": file_name,
                "annotation_type":   r.get("annotation_type"),
                "extracted_value":   r.get("extracted_value"),
                "verdict":           r.get("verdict", ""),
                "match_type":        r.get("match_type", ""),
                "type_mismatch":     r.get("type_mismatch"),
                "correct_type":      r.get("correct_type"),
                "value_correct":     r.get("value_correct"),
                "value_complete":    r.get("value_complete"),
                "hallucination":     r.get("hallucination"),
                "corrected_value":   r.get("corrected_value"),
                "issue_summary":     r.get("issue_summary"),
            }
            for r in judge_scores.get("per_annotation_review", [])
        ]
        review_csv.append_rows(review_rows)

        stats_csv.append_rows([file_stat])

        #write filtered .ann 
        filt_stats = _filter_single_paper(
            file_name, judge_scores, pred_dir, FILTERED_DIR,
        )
        filter_csv.append_rows([{"file_name": file_name, **filt_stats}])

        elapsed = time.time() - t_start
        n_tot = file_stat["judge_n_total"]
        print(f"\n  ✓ PAPER COMPLETE: {file_name}  ({elapsed:.1f}s)")
        print(
            f"    verdicts: high={file_stat['judge_n_high']}  "
            f"medium={file_stat['judge_n_medium']}  "
            f"low={file_stat['judge_n_low']}  (of {n_tot})"
        )
        print(
            f"    filter: kept={filt_stats['kept']}  "
            f"removed={filt_stats['removed']}  "
            f"corrected={filt_stats['corrected']}"
        )
        print(f"    results written to disk — you can inspect them now.")

    review_csv.close()
    stats_csv.close()
    filter_csv.close()

    cache_stats = _get_judge_model().get_cache_stats()
    print("CACHE STATISTICS")
    for k, v in cache_stats.items():
        print(f"  {k:<35} {v}")

    return per_paper_judge, per_file_stats

#Plot

def plot_geval_results(per_file_stats: list, output_dir: str):
    judge_stats = [s for s in per_file_stats if s.get("judge_n_total") and s["judge_n_total"] > 0]
    if not judge_stats:
        print("No GEval scores to plot.")
        return

    CLR = dict(clean="#27ae60", hall="#e74c3c", mismatch="#9b59b6",
               wrong="#e67e22", incompl="#3498db")
    names      = [s["file_name"] for s in judge_stats]
    n_total    = np.array([s["judge_n_total"] for s in judge_stats], float)
    n_hall     = np.array([round(s.get("judge_hallucination_rate", 0) * s["judge_n_total"]) for s in judge_stats], float)
    n_mismatch = np.array([s.get("judge_type_mismatch_count", 0) for s in judge_stats], float)
    n_wrong    = np.array([s.get("judge_wrong_value_count", 0) for s in judge_stats], float)
    n_incompl  = np.array([s.get("judge_incomplete_count", 0) for s in judge_stats], float)
    n_correct  = np.clip(n_total - n_hall - n_mismatch - n_wrong - n_incompl, 0, None)
    x = np.arange(len(names))
    w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("GEval -- Annotation Quality Analysis (Source-Text Only)", fontsize=15, fontweight="bold", y=0.98)

    stack_data = [
        (n_correct,  CLR["clean"],    "Correct (high)"),
        (n_incompl,  CLR["incompl"],  "Medium (drift/partial)"),
        (n_wrong,    CLR["wrong"],    "Low (wrong value)"),
        (n_mismatch, CLR["mismatch"], "Type Mismatch"),
        (n_hall,     CLR["hall"],     "Hallucinated"),
    ]

    def _stacked(ax, data_list, normalise=False):
        denom = np.where(n_total == 0, 1, n_total)
        b = np.zeros(len(names))
        for arr, col, lbl in data_list:
            data = arr / denom * 100 if normalise else arr
            bars = ax.bar(x, data, w, bottom=b, color=col, alpha=0.9,
                          edgecolor="white", lw=0.5, label=lbl)
            for bar, val, bot in zip(bars, data, b):
                if val >= (5 if normalise else 2):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.0f}%" if normalise else str(int(val)),
                        ha="center", va="center", fontsize=8, fontweight="bold", color="white",
                    )
            b += data

    for ax, norm, title in [
        (axes[0, 0], False, "Annotation Quality per Paper (raw counts)"),
        (axes[0, 1], True,  "Annotation Quality per Paper (normalised %)"),
    ]:
        _stacked(ax, stack_data, norm)
        if not norm:
            for i, tot in enumerate(n_total):
                ax.text(i, tot + 0.3, f"n={int(tot)}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#333")
        else:
            ax.set_ylim(0, 115)
            ax.axhline(100, color="#aaa", lw=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% of Annotations" if norm else "Number of Annotations", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    ax = axes[1, 0]
    wg = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * wg
    for (arr, col, lbl), off in zip([
        (n_hall,     CLR["hall"],     "Hallucinated"),
        (n_mismatch, CLR["mismatch"], "Type Mismatch"),
        (n_wrong,    CLR["wrong"],    "Wrong value"),
        (n_incompl,  CLR["incompl"],  "Medium-verdict"),
    ], offsets):
        bars = ax.bar(x + off, arr, wg, label=lbl, color=col, alpha=0.85)
        for bar, val in zip(bars, arr):
            if val >= 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(int(val)), ha="center", va="bottom",
                        fontsize=7, color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Error Type Breakdown per Paper", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    grand_total = int(n_total.sum())
    agg_labels  = ["Correct\n(high)", "Medium\n(drift)", "Low\n(wrong)",
                   "Type\nMismatch", "Hallucinated"]
    agg_vals    = [int(n_correct.sum()), int(n_incompl.sum()), int(n_wrong.sum()),
                   int(n_mismatch.sum()), int(n_hall.sum())]
    agg_colors  = [CLR["clean"], CLR["incompl"], CLR["wrong"],
                   CLR["mismatch"], CLR["hall"]]
    bars4 = ax.bar(agg_labels, agg_vals, color=agg_colors, alpha=0.88,
                   edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars4, agg_vals):
        pct = val / grand_total * 100 if grand_total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Count (all papers)", fontsize=10)
    ax.set_title(f"Aggregate Quality  (total = {grand_total})",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    n_high_all   = sum(s.get("judge_n_high", 0) for s in judge_stats)
    n_medium_all = sum(s.get("judge_n_medium", 0) for s in judge_stats)
    n_low_all    = sum(s.get("judge_n_low", 0) for s in judge_stats)
    total_all    = int(n_total.sum())
    ax.text(0.5, 0.97,
            f"Verdicts:  high={n_high_all}  medium={n_medium_all}  low={n_low_all}  (of {total_all})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold", color=CLR["clean"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                      edgecolor=CLR["clean"], alpha=0.8))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, "llm_judge_summary.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"GEval summary plot saved to: {out}")

    fig2, ax2 = plt.subplots(figsize=(max(12, len(names) * 1.2), 6))
    n_hi = np.array([s.get("judge_n_high", 0) for s in judge_stats], float)
    n_me = np.array([s.get("judge_n_medium", 0) for s in judge_stats], float)
    n_lo = np.array([s.get("judge_n_low", 0) for s in judge_stats], float)
    xb = np.arange(len(names))
    w2 = 0.6
    ax2.bar(xb, n_hi, w2, label="High",   color="#27ae60", alpha=0.9, edgecolor="white", lw=0.5)
    ax2.bar(xb, n_me, w2, bottom=n_hi,         label="Medium", color="#f39c12", alpha=0.9, edgecolor="white", lw=0.5)
    ax2.bar(xb, n_lo, w2, bottom=n_hi + n_me,  label="Low",    color="#e74c3c", alpha=0.9, edgecolor="white", lw=0.5)
    for i, (h, m, lo, tot) in enumerate(zip(n_hi, n_me, n_lo, n_total)):
        if h  >= 2: ax2.text(i, h / 2,          str(int(h)),  ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        if m  >= 2: ax2.text(i, h + m / 2,      str(int(m)),  ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        if lo >= 2: ax2.text(i, h + m + lo / 2, str(int(lo)), ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        ax2.text(i, h + m + lo + 0.3, f"n={int(tot)}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#333")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Number of Annotations", fontsize=12, fontweight="bold")
    ax2.set_title("LLM Judge -- Verdict Distribution per Paper  (high / medium / low)",
                  fontsize=13, fontweight="bold", pad=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=10)
    high_pct = n_high_all / total_all * 100 if total_all else 0
    ax2.text(0.98, 0.97, f"Overall: {high_pct:.1f}% high verdicts",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=11, fontweight="bold", color="#27ae60",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                       edgecolor="#27ae60", alpha=0.85))
    plt.tight_layout()
    out2 = os.path.join(output_dir, "llm_judge_verdict_distribution.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Verdict distribution plot saved to: {out2}")


def plot_match_distribution(per_paper_judge: dict, output_dir: str):
    KEYS   = ["CORRECT", "PARTIAL", "TYPE_MISMATCH", "HALLUCINATED", "NO_MATCH"]
    COLORS = ["#10B981", "#3B82F6", "#8B5CF6", "#EF4444", "#F59E0B"]
    LABELS = ["Correct", "Partial", "Type Mismatch", "Hallucinated", "Wrong Value"]

    paper_ids = sorted(per_paper_judge.keys())
    counts = {pid: {k: 0 for k in KEYS} for pid in paper_ids}
    for pid in paper_ids:
        for r in per_paper_judge[pid].get("per_annotation_review", []):
            mt = r.get("match_type", "NO_MATCH")
            counts[pid][mt if mt in KEYS else "NO_MATCH"] += 1

    names  = paper_ids
    x      = np.arange(len(names))
    w      = 0.6
    data   = {k: np.array([counts[p][k] for p in names], float) for k in KEYS}
    totals = sum(data[k] for k in KEYS)

    fig, axes = plt.subplots(1, 2, figsize=(max(18, len(names) * 1.4), 7))
    fig.suptitle("Match Category Distribution per Paper", fontsize=14, fontweight="bold")
    for ax, normalise in zip(axes, [False, True]):
        denom = np.where(totals == 0, 1, totals)
        b = np.zeros(len(names))
        for k, col, lbl in zip(KEYS, COLORS, LABELS):
            pd_ = data[k] / denom * 100 if normalise else data[k]
            bars = ax.bar(x, pd_, w, bottom=b, color=col, alpha=0.88,
                          edgecolor="white", linewidth=0.5, label=lbl)
            for bar, val, bot in zip(bars, pd_, b):
                if val >= (5 if normalise else 2):
                    ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                            f"{val:.0f}%" if normalise else str(int(val)),
                            ha="center", va="center", fontsize=7, fontweight="bold", color="white")
            b += pd_
        if not normalise:
            for i, tot in enumerate(totals):
                ax.text(i, tot + 0.3, f"n={int(tot)}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#333")
        else:
            ax.set_ylim(0, 115)
            ax.axhline(100, color="#aaa", lw=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("% of Annotations" if normalise else "Number of Annotations", fontsize=11)
        ax.set_title(
            "Match Distribution (normalised %)" if normalise else "Match Distribution (raw counts)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(output_dir, "match_distribution_per_paper.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: match_distribution_per_paper.png")

    agg   = {k: int(sum(data[k])) for k in KEYS}
    grand = sum(agg.values())
    fig, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Aggregate Match Category Distribution  (total = {grand})",
                 fontsize=14, fontweight="bold")
    ax3 = axes2[0]
    wlbls = [
        f"{lbl}\n{agg[k]} ({agg[k]/grand*100:.1f}%)" if grand else lbl
        for lbl, k in zip(LABELS, KEYS)
    ]
    wedges, _ = ax3.pie(
        [agg[k] for k in KEYS], colors=COLORS, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    ax3.legend(wedges, wlbls, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax3.set_title("Donut breakdown", fontsize=11)
    ax3.add_artist(plt.Circle((0, 0), 0.55, color="white"))
    matched_pct = (agg["CORRECT"] + agg["PARTIAL"]) / grand * 100 if grand else 0
    ax3.text(0, 0, f"{matched_pct:.1f}%\nmatched", ha="center", va="center",
             fontsize=11, fontweight="bold", color="#333")
    ax4 = axes2[1]
    bars4 = ax4.bar(LABELS, [agg[k] for k in KEYS], color=COLORS,
                    alpha=0.88, edgecolor="white", linewidth=1.0)
    for bar, val in zip(bars4, [agg[k] for k in KEYS]):
        pct = val / grand * 100 if grand else 0
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax4.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=9)
    ax4.set_ylabel("Total count (all papers)", fontsize=11)
    ax4.set_title("Bar breakdown", fontsize=11)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    ax4.set_axisbelow(True)
    plt.tight_layout()
    out2 = os.path.join(output_dir, "match_distribution_aggregate.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: match_distribution_aggregate.png")


if __name__ == "__main__":
    print("PART 3 -- GEVAL AS JUDGE  (verdicts: high / medium / low)")
    print("Mode: SOURCE TEXT + PREDICTIONS ONLY (no ground truth)")
    print("Hallucination = value ABSENT from paper text.")
    print("Type mismatch = value's fundamental nature belongs to a different annotation category.")
    print("Wrong value   = present in text, right category, but wrong kind or factually incorrect.")
    print("Correct       = present in text, right type, valid for this experiment.")
    print("FactorValue   = INCLUDED in evaluation.")
    print()
    print(f"Predictions Dir  : {PREDICTIONS_DIR}")
    print(f"Papers JSON      : {PAPERS_JSON}")
    print(f"Results Dir      : {RESULTS_DIR}")
    print(f"Filtered Dir     : {FILTERED_DIR}")
    print(f"Judge model      : {EVALUATION_MODEL} via OpenRouter (thinking={MODEL_ENABLE_THINKING})")
    print(f"Disk cache       : {'ENABLED' if CACHE_ENABLED else 'DISABLED'}  dir={CACHE_DIR}")
    print(f"Concurrency      : {MAX_WORKERS} workers per paper")
    print()
    print("Output mode: INCREMENTAL")
    print("  -> CSVs and filtered .ann files are written after EACH paper completes.")
    print("  -> You can inspect results on disk while the pipeline is still running.")
    print()
    print("Caching architecture:")
    print("  Layer 1: Disk cache     (hash-keyed JSON files)")
    print("  Layer 2: cache_control  (system prompt + paper text cached by provider)")
    print()

    per_paper_judge, per_file_stats = run_geval_judge(
        PREDICTIONS_DIR, PAPERS_JSON
    )
    if per_paper_judge:
        print("Generating summary plots...")
        plot_geval_results(per_file_stats, RESULTS_DIR)
        plot_match_distribution(per_paper_judge, RESULTS_DIR)

        print(f"FINAL SUMMARY  ({len(per_paper_judge)} papers)")
        for stat in per_file_stats:
            n_tot = stat.get("judge_n_total", 0)
            print(
                f"  {stat['file_name']:<30} total={n_tot}  "
                f"high={stat.get('judge_n_high', 0)}  "
                f"medium={stat.get('judge_n_medium', 0)}  "
                f"low={stat.get('judge_n_low', 0)}  "
                f"halluc_rate={stat.get('judge_hallucination_rate', 0):.2f}"
            )

        print(f"\nOutput files (available since each paper completed):")
        print(f"  Review CSV   : {RESULTS_DIR}/llm_judge_annotation_review.csv")
        print(f"  Per-file CSV : {RESULTS_DIR}/llm_judge_per_file.csv")
        print(f"  Filtered dir : {FILTERED_DIR}/")
        print(f"  Filter stats : {FILTERED_DIR}/_filter_summary.csv")
        print(f"  Plots        : {RESULTS_DIR}/llm_judge_summary.png")
        print("\nPART 3 COMPLETE.")
    else:
        print("No results produced.")
