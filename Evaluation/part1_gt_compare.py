import os
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from difflib import SequenceMatcher
import numpy as np
from scipy.stats import gaussian_kde
from fuzzywuzzy import fuzz
import json

RESULTS_DIR      = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/evaluation_results_gpt/GT_comparison"
GROUND_TRUTH_DIR = "/Users/fateme/Downloads/Ians_Annotations"
PREDICTIONS_DIR  = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/filtered_output"
PAPERS_JSON         = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/papers_dataset.json"

os.makedirs(RESULTS_DIR, exist_ok=True)

ENTITY_TYPE_ALIASES = {
    'diseasetreatment':        'treatment',
    'disease treatment':       'treatment',
    'experimentaltreatment':   'treatment',
    'experimental treatment':  'treatment',
    'condition':               'treatment',
    'enzyme':                  'cleavageagent',
    'protease':                'cleavageagent',
    'digestingenzyme':         'cleavageagent',
    'digestingagent':          'cleavageagent',
    'tissue':                  'organismpart',
    'organ':                   'organismpart',
    'bodyfluids':              'organismpart',
    'bodyfluid':               'organismpart',
    'sample source':           'organismpart',
    'samplesource':            'organismpart',
    'sampletype':              'materialtype',
    'sample type':             'materialtype',
    'biologicalmaterial':      'materialtype',
    'biological material':     'materialtype',
    'samplematrix':            'materialtype',
    'sample matrix':           'materialtype',
    'chromatography':          'separation',
    'lcmethod':                'separation',
    'lc method':               'separation',
    'liquidchromatography':    'separation',
    'liquid chromatography':   'separation',
    'massspectrometer':        'instrument',
    'mass spectrometer':       'instrument',
    'msplatform':              'instrument',
    'ms platform':             'instrument',
    'labelingmethod':          'label',
    'labeling method':         'label',
    'quantificationmethod':    'label',
    'quantification method':   'label',
    'isotopelabel':            'label',
    'acquisitionmode':         'acquisitionmethod',
    'acquisition mode':        'acquisitionmethod',
    'scanmode':                'acquisitionmethod',
    'scan mode':               'acquisitionmethod',
    'biologicalreplicates':    'numberofbiologicalreplicates',
    'biological replicates':   'numberofbiologicalreplicates',
    'technicalreplicates':     'numberoftechnicalreplicates',
    'technical replicates':    'numberoftechnicalreplicates',
    'replicates':              'numberofbiologicalreplicates',
    'enrichment':              'enrichmentmethod',
    'enrichment method':       'enrichmentmethod',
    'fractionation':           'enrichmentmethod',
    'fractionation method':    'enrichmentmethod',
}

C18_TERMS = {
    'c18', 'c-18', 'c 18', 'reversed-phase', 'reverse-phase', 'reverse phase',
    'rp-hplc', 'rp hplc', 'rp-lc', 'rplc', 'pepmap', 'pep map',
    'reprosil', 'acclaim', 'easy-spray', 'easysray',
}

ENRICHMENT_SEMANTIC_GROUPS = {
    'acetylpeptide enrichment': {
        'acetylpeptide enrichment', 'acetyl peptide enrichment',
        'acetylation enrichment', 'acetyl enrichment',
        'kac enrichment', 'kac peptide', 'kac peptides',
        'enriched kac', 'enrich kac',
        'anti-acetyllysine', 'anti acetyllysine', 'anti-kac', 'anti kac',
        'acetylation antibody', 'acetyl-lysine antibody', 'acetyl lysine antibody',
        'pan anti-acetyl-lysine', 'pan anti acetyl lysine',
        'antibody beads', 'antibody bead', 'ptm biolabs',
    },
    'phosphopeptide enrichment': {
        'phosphopeptide enrichment', 'phospho enrichment',
        'phosphopeptide', 'phosphorylation enrichment',
        'tio2', 'titanium dioxide', 'imac', 'fe-imac', 'ga-imac',
        'metal oxide affinity', 'moac', 'phospho antibody',
        'anti-phospho', 'anti phospho',
    },
    'ubiquitinpeptide enrichment': {
        'ubiquitin enrichment', 'ubiquitination enrichment',
        'ubiquitylation enrichment', 'ub enrichment',
        'anti-ubiquitin', 'anti ubiquitin', 'gg remnant',
        'diglycine remnant', 'ubiquitin antibody',
        'pt-gly-gly', 'ubiquitinated peptide',
    },
    'glycopeptide enrichment': {
        'glycopeptide enrichment', 'glycan enrichment',
        'lectin affinity', 'con a', 'wga', 'hydrazide chemistry',
        'boronic acid', 'hilic', 'graphite column',
    },
    'scx enrichment': {
        'scx', 'strong cation exchange', 'strong cation exchange enrichment',
        'scx fractionation',
    },
}

_ENRICHMENT_TERM_TO_CANONICAL: dict = {}
for _canonical, _terms in ENRICHMENT_SEMANTIC_GROUPS.items():
    for _term in _terms:
        _ENRICHMENT_TERM_TO_CANONICAL[_term] = _canonical

DOMAIN_SEMANTIC_MAP = {
    'tryptic': {'trypsin', 'tryptic', 'trypsin/lys-c', 'trypsin lys-c', 'lys-c trypsin', 'trypsin+lysc', 'tryp'},
    'trypsin': {'tryptic', 'trypsin', 'trypsin/lys-c', 'trypsin lys-c', 'trypsin+lysc', 'tryp'},
    'lysc':    {'lys-c', 'lysc', 'lys c', 'endopeptidase lys-c', 'lys-c/trypsin', 'lysc/trypsin'},
    'lys-c':   {'lys-c', 'lysc', 'lys c', 'endopeptidase lys-c'},
    'argc':    {'arg-c', 'argc', 'arg c'},
    'arg-c':   {'arg-c', 'argc', 'arg c'},
    'gluc':    {'glu-c', 'gluc', 'glu c', 'v8 protease'},
    'glu-c':   {'glu-c', 'gluc', 'glu c', 'v8 protease'},
    'cells':   {'cell lysate', 'lysate', 'whole cell lysate', 'cell extract', 'cellular extract', 'cell pellet', 'cell culture', 'cells', 'cell'},
    'cell':    {'cell lysate', 'lysate', 'whole cell lysate', 'cell extract', 'cellular extract', 'cell pellet', 'cell culture', 'cells', 'cell'},
    'tissue':  {'tissue lysate', 'tissue extract', 'biopsy', 'tissue', 'tissues'},
    'stimulated':   {'stimulation', 'stimulated', 'ifnb stimulation', 'ifn-b stimulation', 'ifn stimulation', 'cytokine stimulation', 'lps stimulation'},
    'unstimulated': {'unstimulated', 'untreated', 'control', 'no stimulation', 'mock'},
    'untreated':    {'untreated', 'unstimulated', 'control', 'vehicle', 'dmso', 'mock'},
    'reversed-phase c18':   C18_TERMS,
    'reversed-phase c-18':  C18_TERMS,
    'c18':                  C18_TERMS,
    'c-18':                 C18_TERMS,
    'slurry-packed with 3-um c18 packing': C18_TERMS,
    'affinity purification-mass spectrometry (ap-ms)': {'ap-ms', 'ap/ms', 'affinity purification ms', 'affinity purification mass spectrometry', 'tap-ms', 'co-ip ms'},
    'ap-ms': {'ap-ms', 'affinity purification mass spectrometry', 'affinity purification-ms', 'ap/ms', 'tap-ms'},
}

ABBREVIATIONS = {
    'dtt': 'dithiothreitol', 'iaa': 'iodoacetamide',
    'tcep': 'tris(2-carboxyethyl)phosphine', 'tfa': 'trifluoroacetic acid',
    'acn': 'acetonitrile', 'meoh': 'methanol', 'etoh': 'ethanol', 'fa': 'formic acid',
    'kac': 'acetyllysine', 'kme': 'methylated lysine', 'kub': 'ubiquitinated lysine',
    'ptm': 'post-translational modification',
    'anti-acetyllysine': 'acetylpeptide enrichment', 'anti-kac': 'acetylpeptide enrichment',
    'anti-phospho': 'phosphopeptide enrichment',
    'cells': 'cell', 'cell line': 'cell', 'cell lines': 'cell',
    'hesc': 'human embryonic stem cell', 'hescs': 'human embryonic stem cell',
    'ipsc': 'induced pluripotent stem cell', 'ipscs': 'induced pluripotent stem cell',
    'human': 'homo sapiens', 'humans': 'homo sapiens',
    'mouse': 'mus musculus', 'mice': 'mus musculus',
    'rat': 'rattus norvegicus', 'rats': 'rattus norvegicus',
    'male': 'male', 'men': 'male', 'female': 'female', 'women': 'female',
    'dda': 'data dependent acquisition', 'data-dependent acquisition': 'data dependent acquisition',
    'dia': 'data independent acquisition', 'data-independent acquisition': 'data independent acquisition',
    'srm': 'selected reaction monitoring', 'mrm': 'multiple reaction monitoring',
    'prm': 'parallel reaction monitoring',
    'ap-ms': 'affinity purification mass spectrometry',
    'tap-ms': 'tandem affinity purification mass spectrometry',
    'hplc': 'high performance liquid chromatography', 'lc': 'liquid chromatography',
    'uplc': 'ultra performance liquid chromatography',
    'rp-hplc': 'reverse phase high performance liquid chromatography',
    'ms': 'mass spectrometry', 'ms/ms': 'tandem mass spectrometry',
    'lc-ms': 'liquid chromatography mass spectrometry',
    'lc-ms/ms': 'liquid chromatography tandem mass spectrometry',
    'hcd': 'higher energy collisional dissociation',
    'higher-energy collisional dissociation': 'higher energy collisional dissociation',
    'cid': 'collision induced dissociation',
    'collision-induced dissociation': 'collision induced dissociation',
    'etd': 'electron transfer dissociation', 'ecd': 'electron capture dissociation',
    'esi': 'electrospray ionization', 'maldi': 'matrix assisted laser desorption ionization',
    'qtof': 'quadrupole time of flight', 'q-tof': 'quadrupole time of flight',
    'tof': 'time of flight', 'ft-icr': 'fourier transform ion cyclotron resonance',
    'fticr': 'fourier transform ion cyclotron resonance',
    'orbitrap': 'orbitrap', 'it': 'ion trap', 'lit': 'linear ion trap',
    'silac': 'stable isotope labeling amino acids cell culture',
    'tmt': 'tandem mass tag', 'tandem mass tags': 'tandem mass tag',
    'itraq': 'isobaric tags relative absolute quantitation',
    'lfq': 'label free', 'label free': 'lfq', 'label-free': 'lfq',
    'label free quantification': 'lfq', 'label-free quantification': 'lfq',
    'carbamidomethylation': 'carbamidomethyl', 'carbamidomethyl': 'carbamidomethylation',
    'oxidation': 'oxidized', 'phosphorylation': 'phosphorylated',
    'acetylation': 'acetylated', 'methylation': 'methylated', 'ubiquitination': 'ubiquitinated',
    'trypsin': 'trypsin', 'lysc': 'lys-c', 'lys-c': 'lysc',
    'argc': 'arg-c', 'arg-c': 'argc', 'gluc': 'glu-c', 'glu-c': 'gluc',
    'chymotrypsin': 'chymotrypsin',
    'samples': 'sample', 'replicates': 'replicate',
    'biological replicates': 'biological replicate', 'technical replicates': 'technical replicate',
    'proteins': 'protein', 'peptides': 'peptide', 'tissues': 'tissue', 'tissue': 'tissue',
    'fractions': 'fraction', 'modifications': 'modification', 'treatments': 'treatment',
    'organisms': 'organism', 'plasma': 'plasma', 'serum': 'serum',
    'lysate': 'lysate', 'lysates': 'lysate', 'cell lysate': 'lysate', 'whole cell lysate': 'lysate',
}

_NUM_TO_WORD = {
    '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
    '11': 'eleven', '12': 'twelve', '15': 'fifteen', '20': 'twenty',
}
_WORD_TO_NUM = {v: k for k, v in _NUM_TO_WORD.items()}

_LFQ_TERMS = {
    'lfq', 'label free', 'label-free', 'labelfree',
    'label free quantification', 'label-free quantification',
    'label free quantitation', 'label-free quantitation', 'lf', 'lf-ms',
}

_CONC_UNIT_PATTERNS = re.compile(
    r'\b(\d+(\.\d+)?\s*(u/ml|ng/ml|ug/ml|mg/ml|nm|um|mm|m\b|pm|fg/ml|pg/ml|iu/ml|'
    r'mol/l|mmol/l|nmol/l|pmol/l|percent|%|v/v|w/v|x\b))',
    re.IGNORECASE
)

_PURE_NUM_UNIT_RE = re.compile(
    r'^\d+(?:\.\d+)?\s*(%|ev|kev|mev|ppm|da|mda|mz|nm|um|mm|cm|ml|ul|nl|pl|'
    r'ng|ug|mg|g|kg|nm/min|nl/min|ul/min|ml/min|min|sec|h|hr|'
    r'u/ml|ng/ml|ug/ml|mg/ml|iu/ml|fg/ml|pg/ml|mol/l|mmol/l|nmol/l|pmol/l|v/v|w/v|[a-z]+)$',
    re.IGNORECASE
)


def normalize_entity_type(entity_type: str) -> str:
    if '[' in entity_type:
        entity_type = entity_type.split('[')[1].rstrip(']')
    key = entity_type.strip().lower()
    if key in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[key]
    key_nospace = key.replace(' ', '').replace('-', '').replace('_', '')
    if key_nospace in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[key_nospace]
    return entity_type.strip()


def should_exclude_entity(entity_type: str) -> bool:
    return entity_type.lower().strip().startswith('factorvalue')


def normalize_text(text: str) -> str:
    return ' '.join(text.split()).lower().strip()


def normalize_hyphen_variants(text: str) -> str:
    text = normalize_text(text)
    return re.sub(r'(\w)\s*-\s*(\w)', r'\1\2', text)


def expand_abbreviations(text: str) -> str:
    text = normalize_text(text)
    if text in ABBREVIATIONS:
        return ABBREVIATIONS[text]
    expanded = text
    for abbr, full in ABBREVIATIONS.items():
        expanded = re.sub(r'\b' + re.escape(abbr) + r'\b', full, expanded)
    return expanded


def normalize_for_semantic_matching(text: str) -> str:
    text = expand_abbreviations(normalize_text(text))
    stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'and', 'or', 'by', 'from', 'using', 'via', 'was', 'were', 'is', 'are'}
    words = text.split()
    if len(words) > 3:
        words = [w for w in words if w not in stop_words]
    return ' '.join(words)


def _resolve_enrichment_canonical(text: str):
    norm    = normalize_text(text)
    norm_nh = normalize_hyphen_variants(norm)
    if norm in _ENRICHMENT_TERM_TO_CANONICAL:
        return _ENRICHMENT_TERM_TO_CANONICAL[norm]
    if norm_nh in _ENRICHMENT_TERM_TO_CANONICAL:
        return _ENRICHMENT_TERM_TO_CANONICAL[norm_nh]
    for term, canonical in _ENRICHMENT_TERM_TO_CANONICAL.items():
        t_nh = normalize_hyphen_variants(term)
        if t_nh and (t_nh in norm_nh or norm_nh in t_nh):
            return canonical
    for kw in ['kac', 'acetyllysine', 'acetyl lysine', 'acetylation enrichment', 'acetylated peptide', 'kac peptide', 'acetylpeptide', 'acetyl peptide', 'anti acetyl', 'antiacetyl', 'pan anti acetyl', 'enrich kac', 'enriched kac', 'ptm biolabs']:
        if kw in norm_nh:
            return 'acetylpeptide enrichment'
    for kw in ['phosphopeptide', 'phospho peptide', 'phosphorylation enrichment', 'tio2', 'imac', 'feimac']:
        if kw in norm_nh:
            return 'phosphopeptide enrichment'
    for kw in ['ubiquitin', 'ubiquitination', 'ubiquitylation', 'diglycine', 'ggremnant']:
        if kw in norm_nh:
            return 'ubiquitinpeptide enrichment'
    for kw in ['glycopeptide', 'glycan', 'lectin', 'hilic']:
        if kw in norm_nh:
            return 'glycopeptide enrichment'
    return None


def enrichment_method_match(text1: str, text2: str) -> bool:
    c1 = _resolve_enrichment_canonical(text1)
    c2 = _resolve_enrichment_canonical(text2)
    if c1 is not None and c2 is not None:
        return c1 == c2
    norm1_nh = normalize_hyphen_variants(normalize_text(text1))
    norm2_nh = normalize_hyphen_variants(normalize_text(text2))
    for canonical, source_text_nh in [(c1, norm2_nh), (c2, norm1_nh)]:
        if canonical is not None:
            for term in ENRICHMENT_SEMANTIC_GROUPS.get(canonical, set()):
                t_nh = normalize_hyphen_variants(term)
                if t_nh and t_nh in source_text_nh:
                    return True
    return False


def _looks_like_pure_concentration(text: str) -> bool:
    return bool(re.match(r'^\d+(\.\d+)?\s*[\w%/]+$', normalize_text(text)))


def compound_concentration_match(text1: str, text2: str) -> bool:
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    if not norm1 or not norm2:
        return False
    if _looks_like_pure_concentration(norm1) and _looks_like_pure_concentration(norm2):
        return False
    if len(norm1) <= len(norm2):
        short, long = norm1, norm2
    else:
        short, long = norm2, norm1
    if len(short) / len(long) > 0.75 or len(short) < 2:
        return False
    if short in long:
        return True
    short_nh = normalize_hyphen_variants(short)
    long_nh  = normalize_hyphen_variants(long)
    if short_nh in long_nh:
        return True
    short_tokens = [t for t in re.findall(r'\b\w+\b', short_nh) if len(t) >= 2]
    long_tokens  = set(re.findall(r'\b\w+\b', long_nh))
    if short_tokens and sum(1 for t in short_tokens if t in long_tokens) / len(short_tokens) >= 0.80:
        return True
    return False


def _extract_leading_number(text: str):
    t = normalize_text(text)
    m = re.match(r'^(\d+)', t)
    if m:
        return m.group(1)
    for word, digit in _WORD_TO_NUM.items():
        if re.search(r'\b' + word + r'\b', t):
            return digit
    return None


def numeric_replicate_match(text1: str, text2: str) -> bool:
    n1 = _extract_leading_number(text1)
    n2 = _extract_leading_number(text2)
    return n1 is not None and n2 is not None and n1 == n2


def _is_lfq(text: str) -> bool:
    norm = normalize_hyphen_variants(normalize_text(text))
    return norm in _LFQ_TERMS or any(term in norm for term in _LFQ_TERMS)


def label_free_match(text1: str, text2: str) -> bool:
    return _is_lfq(text1) and _is_lfq(text2)


def semantic_similarity(text1: str, text2: str) -> bool:
    norm1 = normalize_for_semantic_matching(text1)
    norm2 = normalize_for_semantic_matching(text2)
    if norm1 == norm2:
        return True
    if normalize_hyphen_variants(text1) == normalize_hyphen_variants(text2):
        return True
    if len(norm1) > 2 and len(norm2) > 2 and (norm1 in norm2 or norm2 in norm1):
        return True
    return False


def domain_semantic_match(text1: str, text2: str) -> bool:
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    for gt_key, equiv_set in DOMAIN_SEMANTIC_MAP.items():
        if norm1 == gt_key or gt_key in norm1:
            if norm2 in equiv_set or any(e in norm2 for e in equiv_set):
                return True
        if norm2 == gt_key or gt_key in norm2:
            if norm1 in equiv_set or any(e in norm1 for e in equiv_set):
                return True
    return False


def morphological_match(text1: str, text2: str) -> bool:
    MORPHO_GROUPS = [
        {'trypsin', 'tryptic', 'trypsinize', 'trypsinized', 'tryp'},
        {'stimulated', 'stimulation', 'stimulate', 'stimulating'},
        {'unstimulated', 'untreated', 'unstimulate', 'no treatment', 'control', 'mock', 'vehicle'},
        {'cells', 'cell', 'cell lysate', 'lysate', 'whole cell lysate', 'cell extract', 'cellular extract', 'cell pellet'},
        {'lysc', 'lys-c', 'lys c', 'lysine c', 'endopeptidase lys-c', 'endolysine c', 'lysc/trypsin', 'lys-c/trypsin'},
        {'argc', 'arg-c', 'arg c'},
        {'gluc', 'glu-c', 'glu c', 'v8 protease'},
        {'c18', 'c-18', 'c 18', 'reversed-phase c18', 'reversed-phase c-18', 'reverse phase c18', 'reverse-phase c18', 'rp c18'},
        {'ap-ms', 'apms', 'affinity purification mass spectrometry', 'affinity purification-mass spectrometry', 'affinity purification ms', 'tap-ms'},
    ]
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    nh1   = normalize_hyphen_variants(norm1)
    nh2   = normalize_hyphen_variants(norm2)
    for group in MORPHO_GROUPS:
        nh_group = {normalize_hyphen_variants(g) for g in group}
        if (norm1 in group or nh1 in nh_group) and (norm2 in group or nh2 in nh_group):
            return True
        for member in group:
            if member in norm1 and member in norm2:
                return True
        for member in group:
            root = re.sub(r'(ing|tion|ate|ed)$', '', member)
            if len(root) >= 4 and root in norm1 and root in norm2:
                return True
    return False


def bait_containment_match(text1: str, text2: str) -> bool:
    BAIT_STOP_TOKENS = {'enrichment', 'method', 'analysis', 'approach', 'technique', 'protocol', 'assay', 'experiment', 'procedure', 'sample', 'peptide', 'peptides', 'protein', 'proteins', 'the', 'and', 'with', 'for', 'using', 'based', 'type', 'mode', 'level', 'purification', 'purified', 'isolation', 'extraction', 'capture', 'affinity', 'tagged', 'tagging', 'pull', 'pulldown', 'nanobody', 'antibody', 'bead', 'beads', 'resin'}
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    for tok1 in re.split(r'[-_\s]+', t1):
        if len(tok1) < 4 or tok1 in BAIT_STOP_TOKENS:
            continue
        for tok2 in re.split(r'[-_\s]+', t2):
            if tok2 in BAIT_STOP_TOKENS:
                continue
            if tok2.startswith(tok1) or tok1.startswith(tok2):
                return True
    return False


def key_term_in_long_text(short_text: str, long_text: str) -> bool:
    short_norm = normalize_text(short_text)
    long_norm  = normalize_text(long_text)
    if len(short_norm) / max(len(long_norm), 1) > 0.6:
        return False
    short_tokens = [t for t in re.findall(r'\b\w+\b', short_norm) if len(t) >= 3]
    if not short_tokens:
        return False
    long_nh = normalize_hyphen_variants(long_norm)
    matches = sum(1 for tok in short_tokens if tok in long_norm or normalize_hyphen_variants(tok) in long_nh)
    if matches / len(short_tokens) >= 0.8:
        return True
    c18_nh = {normalize_hyphen_variants(t) for t in C18_TERMS}
    long_words = set(re.findall(r'\b\w+\b', long_nh))
    if bool(c18_nh & long_words) and any(normalize_hyphen_variants(t) in normalize_hyphen_variants(short_norm) for t in C18_TERMS):
        return True
    return False


def substring_match(text1: str, text2: str) -> bool:
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    if len(norm1) < 3 or len(norm2) < 3:
        return False
    return norm1 in norm2 or norm2 in norm1


def extract_tokens(text: str) -> set:
    return {t for t in re.findall(r'\b\w+\b', normalize_text(text)) if len(t) > 1}


def extract_key_terms(text: str) -> set:
    norm = normalize_text(text)
    abbrevs = {a.lower() for a in re.findall(r'\(([A-Z0-9\-]+)\)', norm)}
    return abbrevs | extract_tokens(norm)


def token_overlap_ratio(text1: str, text2: str) -> float:
    t1, t2 = extract_tokens(text1), extract_tokens(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def key_terms_match(text1: str, text2: str) -> float:
    t1, t2 = extract_key_terms(text1), extract_key_terms(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / min(len(t1), len(t2))


def sequence_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()


def fuzzy_partial(text1: str, text2: str) -> float:
    return fuzz.partial_ratio(normalize_text(text1), normalize_text(text2)) / 100.0


def _is_pure_num_unit(text: str) -> bool:
    return bool(_PURE_NUM_UNIT_RE.match(normalize_text(text)))


def calculate_match_score(text1: str, text2: str) -> float:
    if _is_pure_num_unit(text1) and _is_pure_num_unit(text2):
        return 1.0 if normalize_text(text1) == normalize_text(text2) else 0.0
    if normalize_text(text1) == normalize_text(text2):
        return 1.0
    if normalize_hyphen_variants(text1) == normalize_hyphen_variants(text2):
        return 0.98
    if semantic_similarity(text1, text2):
        return 0.95
    if domain_semantic_match(text1, text2):
        return 0.93
    if label_free_match(text1, text2):
        return 0.92
    if enrichment_method_match(text1, text2):
        return 0.91
    if morphological_match(text1, text2):
        return 0.90
    if compound_concentration_match(text1, text2):
        return 0.89
    if bait_containment_match(text1, text2):
        return 0.88
    if numeric_replicate_match(text1, text2):
        return 0.87
    if key_term_in_long_text(text1, text2) or key_term_in_long_text(text2, text1):
        return 0.85
    fp = fuzzy_partial(text1, text2)
    if fp >= 0.85:
        return fp * 0.90
    return 0.40 * key_terms_match(text1, text2) + 0.35 * token_overlap_ratio(text1, text2) + 0.25 * sequence_similarity(text1, text2)


def get_match_type(text1: str, text2: str, score: float) -> str:
    if normalize_text(text1) == normalize_text(text2):
        return "Exact"
    if normalize_hyphen_variants(text1) == normalize_hyphen_variants(text2):
        return "HyphenVariant"
    if semantic_similarity(text1, text2):
        return "Semantic"
    if domain_semantic_match(text1, text2):
        return "DomainSemantic"
    if label_free_match(text1, text2):
        return "LabelFree"
    if enrichment_method_match(text1, text2):
        return "EnrichmentSemantic"
    if morphological_match(text1, text2):
        return "Morphological"
    if compound_concentration_match(text1, text2):
        return "CompoundConcentrationMerge"
    if bait_containment_match(text1, text2):
        return "TagContainment"
    if numeric_replicate_match(text1, text2):
        return "NumericReplicate"
    if key_term_in_long_text(text1, text2) or key_term_in_long_text(text2, text1):
        return "KeyTermInLongText"
    if substring_match(text1, text2):
        return "Substring"
    if score >= 0.85:
        return "Near-Exact"
    if score >= 0.70:
        return "High-Overlap"
    return "Partial"


def parse_ann_file(file_path):
    entities = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or ':' not in line:
                    continue
                entity_type, value = line.split(':', 1)
                entity_type = entity_type.strip()
                value       = value.strip()
                if entity_type and value:
                    entities.append((entity_type, value, line_num))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return entities


def calculate_metrics_per_type(gt_entities_by_type, pred_entities_by_type):
    results   = {}
    THRESHOLD = 0.5
    all_types = set(gt_entities_by_type.keys()) | set(pred_entities_by_type.keys())

    for gt_type in sorted(all_types):
        gt_entities   = gt_entities_by_type.get(gt_type, [])
        pred_entities = pred_entities_by_type.get(gt_type, [])

        if not gt_entities:
            fp = len(pred_entities)
            results[gt_type] = {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': fp, 'fn': 0,
                'gt_count': 0, 'pred_count': fp, 'matches': [], 'unmatched_gt': [],
                'unmatched_pred': [{'pred_text': txt, 'pred_line': ln, 'pred_file': fn} for _, txt, ln, fn in pred_entities]
            }
            continue

        if not pred_entities:
            results[gt_type] = {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': 0, 'fn': len(gt_entities),
                'gt_count': len(gt_entities), 'pred_count': 0, 'matches': [],
                'unmatched_gt': [{'gt_text': txt, 'gt_line': ln, 'gt_file': fn} for _, txt, ln, fn in gt_entities],
                'unmatched_pred': []
            }
            continue

        gt_matches   = defaultdict(list)
        pred_matches = defaultdict(list)

        for gi, (_, gt_text, gt_line, gt_file) in enumerate(gt_entities):
            for pi, (_, pred_text, pred_line, pred_file) in enumerate(pred_entities):
                if gt_file != pred_file:
                    continue
                score = calculate_match_score(gt_text, pred_text)
                if score >= THRESHOLD:
                    gt_matches[gi].append((pi, score, pred_text, pred_line, pred_file))
                    pred_matches[pi].append((gi, score, gt_text, gt_line, gt_file))

        tp = len(gt_matches)
        fn = len(gt_entities)   - len(gt_matches)
        fp = len(pred_entities) - len(pred_matches)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

        matches = [
            {'gt_text': gt_entities[gi][1], 'gt_line': gt_entities[gi][2], 'gt_file': gt_entities[gi][3],
             'pred_text': pred_text, 'pred_line': pred_line, 'pred_file': pred_file,
             'similarity': score, 'match_type': get_match_type(gt_entities[gi][1], pred_text, score)}
            for gi, match_list in gt_matches.items()
            for pi, score, pred_text, pred_line, pred_file in match_list
        ]
        unmatched_gt   = [{'gt_text': txt, 'gt_line': ln, 'gt_file': fn} for gi, (_, txt, ln, fn) in enumerate(gt_entities) if gi not in gt_matches]
        unmatched_pred = [{'pred_text': txt, 'pred_line': ln, 'pred_file': fn} for pi, (_, txt, ln, fn) in enumerate(pred_entities) if pi not in pred_matches]

        results[gt_type] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn,
            'gt_count': len(gt_entities), 'pred_count': len(pred_entities),
            'matches': matches, 'unmatched_gt': unmatched_gt, 'unmatched_pred': unmatched_pred
        }
    return results


def evaluate_gt(gt_dir, pred_dir):
    gt_files     = {f.stem: f for f in Path(gt_dir).glob('*.ann')}
    pred_files   = {f.stem: f for f in Path(pred_dir).glob('*.ann')}
    common_files = set(gt_files.keys()) & set(pred_files.keys())

    if not common_files:
        print("No matching .ann files found!")
        return None

    print(f"Found {len(common_files)} matching files")

    global_gt_by_type   = defaultdict(list)
    global_pred_by_type = defaultdict(list)
    per_file_stats      = []

    for file_name in sorted(common_files):
        print(f"\n{'='*80}\nProcessing: {file_name}.ann")

        gt_entities   = parse_ann_file(gt_files[file_name])
        pred_entities = parse_ann_file(pred_files[file_name])

        gt_filtered   = [(t, txt, ln) for t, txt, ln in gt_entities   if not should_exclude_entity(t)]
        pred_filtered = [(t, txt, ln) for t, txt, ln in pred_entities if not should_exclude_entity(t)]

        print(f"  GT: {len(gt_filtered)}  Pred: {len(pred_filtered)}")

        gt_normalized   = [(normalize_entity_type(t), txt, ln, file_name) for t, txt, ln in gt_filtered]
        pred_normalized = [(normalize_entity_type(t), txt, ln, file_name) for t, txt, ln in pred_filtered]

        file_gt   = defaultdict(list)
        file_pred = defaultdict(list)
        for entity in gt_normalized:
            file_gt[entity[0]].append(entity)
        for entity in pred_normalized:
            file_pred[entity[0]].append(entity)

        file_metrics = calculate_metrics_per_type(file_gt, file_pred)

        if file_metrics:
            ftp = sum(m['tp'] for m in file_metrics.values())
            ffp = sum(m['fp'] for m in file_metrics.values())
            ffn = sum(m['fn'] for m in file_metrics.values())
            fp_ = ftp / (ftp+ffp) if (ftp+ffp) > 0 else 0
            fr_ = ftp / (ftp+ffn) if (ftp+ffn) > 0 else 0
            ff_ = 2*fp_*fr_ / (fp_+fr_) if (fp_+fr_) > 0 else 0
            per_file_stats.append({'file_name': file_name, 'gt_count': len(gt_normalized), 'pred_count': len(pred_normalized), 'tp': ftp, 'fp': ffp, 'fn': ffn, 'precision': fp_, 'recall': fr_, 'f1': ff_})
            print(f"  P={fp_:.4f}  R={fr_:.4f}  F1={ff_:.4f}")

        for entity in gt_normalized:
            global_gt_by_type[entity[0]].append(entity)
        for entity in pred_normalized:
            global_pred_by_type[entity[0]].append(entity)

    metrics = calculate_metrics_per_type(global_gt_by_type, global_pred_by_type)
    return metrics, global_gt_by_type, global_pred_by_type, per_file_stats


def print_results_table(metrics):
    if not metrics:
        print("No metrics to display!")
        return
    print("\n" + "="*115)
    print(f"{'Entity Type':<35} {'Precision':>10} {'Recall':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'GT':>6} {'Pred':>6}  Note")
    print("-"*115)
    for et in sorted(metrics.keys()):
        m = metrics[et]
        note = ' [PRED-ONLY]' if m['gt_count']==0 else ' [GT-ONLY]' if m['pred_count']==0 else ''
        print(f"{et:<35} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['gt_count']:>6} {m['pred_count']:>6}{note}")
    total_tp = sum(metrics[t]['tp'] for t in metrics)
    total_fp = sum(metrics[t]['fp'] for t in metrics)
    total_fn = sum(metrics[t]['fn'] for t in metrics)
    ovp = total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0
    ovr = total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0
    mcp = sum(m['precision'] for m in metrics.values()) / len(metrics)
    mcr = sum(m['recall']    for m in metrics.values()) / len(metrics)
    print("="*115)
    print(f"{'MICRO-AVERAGED':<35} {ovp:>10.4f} {ovr:>10.4f} {total_tp:>6} {total_fp:>6} {total_fn:>6}")
    print(f"{'MACRO-AVERAGED':<35} {mcp:>10.4f} {mcr:>10.4f}")
    print("="*115 + "\n")


def save_gt_results(metrics, gt_by_type, pred_by_type, per_file_stats, output_dir):
    if not metrics:
        return

    df_data = []
    for et, m in sorted(metrics.items()):
        df_data.append({
            'Entity Type': et, 'Precision': f"{m['precision']:.4f}", 'Recall': f"{m['recall']:.4f}",
            'F1-Score': f"{m['f1']:.4f}", 'True Positives': m['tp'], 'False Positives': m['fp'],
            'False Negatives': m['fn'], 'GT Count': m['gt_count'], 'Pred Count': m['pred_count'],
            'GT Only': 'Yes' if m['gt_count']>0 and m['pred_count']==0 else 'No',
            'Pred Only': 'Yes' if m['gt_count']==0 and m['pred_count']>0 else 'No',
        })
    pd.DataFrame(df_data).to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False, escapechar='\\', doublequote=True)

    if per_file_stats:
        pf = pd.DataFrame(per_file_stats)
        for c in ['precision', 'recall', 'f1']:
            pf[c] = pf[c].apply(lambda x: f"{x:.4f}")
        pf.to_csv(os.path.join(output_dir, 'per_file_results.csv'), index=False, escapechar='\\', doublequote=True)

    total_tp = sum(metrics[t]['tp'] for t in metrics)
    total_fp = sum(metrics[t]['fp'] for t in metrics)
    total_fn = sum(metrics[t]['fn'] for t in metrics)
    ovp = total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0
    ovr = total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0
    ovf = 2*ovp*ovr/(ovp+ovr) if (ovp+ovr) > 0 else 0
    mcp = sum(m['precision'] for m in metrics.values()) / len(metrics)
    mcr = sum(m['recall']    for m in metrics.values()) / len(metrics)
    mcf = sum(m['f1']        for m in metrics.values()) / len(metrics)
    rows = [
        {'Metric': 'Micro-averaged Precision', 'Value': f"{ovp:.4f}"},
        {'Metric': 'Micro-averaged Recall',    'Value': f"{ovr:.4f}"},
        {'Metric': 'Micro-averaged F1',        'Value': f"{ovf:.4f}"},
        {'Metric': 'Macro-averaged Precision', 'Value': f"{mcp:.4f}"},
        {'Metric': 'Macro-averaged Recall',    'Value': f"{mcr:.4f}"},
        {'Metric': 'Macro-averaged F1',        'Value': f"{mcf:.4f}"},
        {'Metric': 'Total GT Entities',        'Value': str(total_tp+total_fn)},
        {'Metric': 'Total Pred Entities',      'Value': str(total_tp+total_fp)},
        {'Metric': 'True Positives',           'Value': str(total_tp)},
        {'Metric': 'False Positives',          'Value': str(total_fp)},
        {'Metric': 'False Negatives',          'Value': str(total_fn)},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False, escapechar='\\', doublequote=True)

    match_rows = [
        {'Entity_Type': et, 'GT_File': m['gt_file'], 'GT_Text': m['gt_text'], 'GT_Line': m['gt_line'],
         'Pred_File': m['pred_file'], 'Pred_Text': m['pred_text'], 'Pred_Line': m['pred_line'],
         'Similarity': f"{m['similarity']:.4f}", 'Match_Type': m['match_type']}
        for et, md in sorted(metrics.items()) for m in md['matches']
    ]
    if match_rows:
        pd.DataFrame(match_rows).to_csv(os.path.join(output_dir, 'match_details.csv'), index=False, escapechar='\\', doublequote=True)

    fn_rows = [{'Entity_Type': et, 'File': u['gt_file'], 'GT_Text': u['gt_text'], 'GT_Line': u['gt_line']} for et, md in sorted(metrics.items()) for u in md['unmatched_gt']]
    if fn_rows:
        pd.DataFrame(fn_rows).to_csv(os.path.join(output_dir, 'false_negatives.csv'), index=False, escapechar='\\', doublequote=True)

    fp_rows = [{'Entity_Type': et, 'File': u['pred_file'], 'Pred_Text': u['pred_text'], 'Pred_Line': u['pred_line']} for et, md in sorted(metrics.items()) for u in md['unmatched_pred']]
    if fp_rows:
        pd.DataFrame(fp_rows).to_csv(os.path.join(output_dir, 'false_positives.csv'), index=False, escapechar='\\', doublequote=True)

    print(f"Results saved to: {output_dir}")


def plot_gt_results(metrics, gt_by_type, pred_by_type, per_file_stats, output_dir):
    if not metrics:
        return

    entity_types = sorted(metrics.keys())
    precisions   = [metrics[t]['precision'] for t in entity_types]
    recalls      = [metrics[t]['recall']    for t in entity_types]
    f1_scores    = [metrics[t]['f1']        for t in entity_types]
    total_tp     = sum(metrics[t]['tp'] for t in entity_types)
    total_fp     = sum(metrics[t]['fp'] for t in entity_types)
    total_fn     = sum(metrics[t]['fn'] for t in entity_types)
    ovp = total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0
    ovr = total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0
    mcp = sum(precisions)/len(precisions) if precisions else 0
    mcr = sum(recalls)/len(recalls)       if recalls    else 0

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 18))
    gs  = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    x   = np.arange(len(entity_types))
    w   = 0.35
    ax1.bar(x-w/2, precisions, w, label='Precision', alpha=0.8, color='#2196F3')
    ax1.bar(x+w/2, recalls,    w, label='Recall',    alpha=0.8, color='#4CAF50')
    ax1.set_xlabel('Entity Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Precision and Recall by Entity Type', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(entity_types, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    x2  = np.arange(2)
    bars1 = ax2.bar(x2-w/2, [ovp, mcp], w, label='Precision', alpha=0.8, color='#2196F3')
    bars2 = ax2.bar(x2+w/2, [ovr, mcr], w, label='Recall',    alpha=0.8, color='#4CAF50')
    ax2.set_xlabel('Averaging Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Overall Performance Metrics', fontsize=13, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['Micro-Avg', 'Macro-Avg'], fontsize=10)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f'{h:.3f}', xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    si  = sorted(range(len(recalls)), key=lambda i: recalls[i], reverse=True)
    st  = [entity_types[i] for i in si]
    sr  = [recalls[i] for i in si]
    colors3 = ['#4CAF50' if r>=0.7 else '#FF9800' if r>=0.5 else '#f44336' for r in sr]
    ax3.barh(range(len(st)), sr, color=colors3, alpha=0.7)
    ax3.set_yticks(range(len(st)))
    ax3.set_yticklabels(st, fontsize=8)
    ax3.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax3.set_xlim([0, 1.1])
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    for i, v in enumerate(sr):
        ax3.text(v+0.02, i, f'{v:.3f}', va='center', fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    gt_c = [metrics[t]['gt_count']   for t in entity_types]
    pr_c = [metrics[t]['pred_count'] for t in entity_types]
    sc   = ax4.scatter(gt_c, pr_c, s=150, alpha=0.6, c=recalls, cmap='RdYlGn', vmin=0, vmax=1, edgecolors='black', linewidth=0.5)
    mv   = max(max(gt_c, default=1), max(pr_c, default=1))
    ax4.plot([0, mv], [0, mv], 'k--', alpha=0.5, label='Perfect Match', linewidth=2)
    for i, et in enumerate(entity_types):
        ax4.annotate(et, (gt_c[i], pr_c[i]), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points')
    ax4.set_xlabel('Ground Truth Count', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Prediction Count',   fontsize=11, fontweight='bold')
    ax4.set_title('Entity Counts: GT vs Predictions', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax4).set_label('Recall', fontsize=10)

    ax5 = fig.add_subplot(gs[2, 0])
    mtc = defaultdict(int)
    for et in entity_types:
        for m in metrics[et]['matches']:
            mtc[m['match_type']] += 1
    if mtc:
        mt_labels = sorted(mtc.keys())
        mt_counts = [mtc[mt] for mt in mt_labels]
        colors5   = ['#4CAF50','#8BC34A','#CDDC39','#FFC107','#FF9800','#F44336','#9C27B0','#2196F3','#00BCD4','#795548','#E91E63'][:len(mt_labels)]
        bars5     = ax5.bar(mt_labels, mt_counts, color=colors5, alpha=0.7)
        ax5.set_xlabel('Match Type', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax5.set_title('Distribution of Match Types', fontsize=13, fontweight='bold')
        ax5.set_xticklabels(mt_labels, rotation=30, ha='right', fontsize=8)
        ax5.grid(axis='y', alpha=0.3)
        for bar in bars5:
            h = bar.get_height()
            ax5.annotate(f'{int(h)}', xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    if per_file_stats and len(per_file_stats) > 1:
        ax6 = fig.add_subplot(gs[2, 1])
        ss  = sorted(per_file_stats, key=lambda x: x['recall'], reverse=True)
        fn_ = [s['file_name'] for s in ss]
        pp  = [s['precision'] for s in ss]
        rr  = [s['recall']    for s in ss]
        x6  = np.arange(len(fn_))
        ax6.bar(x6-w/2, pp, w, label='Precision', alpha=0.8, color='#2196F3')
        ax6.bar(x6+w/2, rr, w, label='Recall',    alpha=0.8, color='#4CAF50')
        ax6.set_xlabel('File / Paper', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax6.set_title('Performance by Paper', fontsize=13, fontweight='bold')
        ax6.set_xticks(x6)
        ax6.set_xticklabels(fn_, rotation=45, ha='right', fontsize=8)
        ax6.legend(fontsize=10)
        ax6.set_ylim([0, 1.1])
        ax6.grid(axis='y', alpha=0.3)

    plt.suptitle('GT Comparison — Evaluation Results', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(output_dir, 'gt_evaluation_results.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {out}")

    if per_file_stats and len(per_file_stats) >= 2:
        f1s        = np.array([s['f1'] for s in per_file_stats])
        names      = [s['file_name'] for s in per_file_stats]
        precs      = np.array([s['precision'] for s in per_file_stats])
        recs       = np.array([s['recall']    for s in per_file_stats])
        tps        = np.array([s['tp'] for s in per_file_stats], dtype=float)
        fps        = np.array([s['fp'] for s in per_file_stats], dtype=float)
        fns        = np.array([s['fn'] for s in per_file_stats], dtype=float)
        Q1, Q3     = np.percentile(f1s, 25), np.percentile(f1s, 75)
        fence      = Q1 - 1.5 * (Q3 - Q1)
        median     = np.median(f1s)
        mean_f1    = np.mean(f1s)
        is_outlier = f1s < fence
        outlier_idx = np.where(is_outlier)[0]
        normal_idx  = np.where(~is_outlier)[0]

        BG = '#ffffff'; PANEL_BG = '#f5f7fa'; TEAL = '#1a8f87'
        RED = '#c0392b'; GOLD = '#d4a017'; BLUE_DOT = '#2471a3'
        TEXT_CLR = '#1a1a1a'; GRID_CLR = '#d0d4db'
        GREEN_BAR = '#27ae60'; ORANGE_BAR = '#e67e22'; GREY_BAR = '#7f8c8d'

        fig2 = plt.figure(figsize=(20, 13), facecolor=BG)
        fig2.suptitle('Per-Paper Performance Distribution', color=TEXT_CLR, fontsize=18, fontweight='bold', y=0.98)
        gs2     = fig2.add_gridspec(3, 2, top=0.93, bottom=0.10, left=0.06, right=0.97, hspace=0.38, wspace=0.28, height_ratios=[1, 1, 0.01])
        ax_kde  = fig2.add_subplot(gs2[0, 0])
        ax_rank = fig2.add_subplot(gs2[0, 1])
        ax_pr   = fig2.add_subplot(gs2[1, 0])
        ax_stk  = fig2.add_subplot(gs2[1, 1])

        for ax in [ax_kde, ax_rank, ax_pr, ax_stk]:
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_CLR, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)

        x_grid = np.linspace(max(0, f1s.min()-0.15), min(1.05, f1s.max()+0.15), 400)
        if len(f1s) >= 3:
            kde_vals = gaussian_kde(f1s, bw_method=0.25)(x_grid)
            ax_kde.plot(x_grid, kde_vals, color=TEAL, lw=2, label='KDE')
            iqr_mask = (x_grid >= Q1) & (x_grid <= Q3)
            ax_kde.fill_between(x_grid, kde_vals, where=iqr_mask, alpha=0.25, color=TEAL, label=f'IQR [{Q1:.2f}-{Q3:.2f}]')
        ax_kde.axvline(median,  color=GOLD,     lw=1.5, linestyle='--', label=f'Median {median:.3f}')
        ax_kde.axvline(mean_f1, color=BLUE_DOT, lw=1.2, linestyle=':',  label=f'Mean {mean_f1:.3f}')
        ax_kde.axvline(fence,   color=RED,      lw=1.2, linestyle='-.', label=f'Fence ({fence:.3f})')
        ax_kde.scatter(f1s[normal_idx],  np.zeros(len(normal_idx)),  color=TEAL, s=40, marker='o', zorder=5, label='Normal')
        if len(outlier_idx):
            ax_kde.scatter(f1s[outlier_idx], np.zeros(len(outlier_idx)), color=RED, s=70, marker='D', zorder=6, label=f'Outlier ({len(outlier_idx)})')
            for i in outlier_idx:
                ax_kde.annotate(names[i], (f1s[i], 0), textcoords='offset points', xytext=(0, -14), color=RED, fontsize=6.5, ha='center', fontweight='bold')
        ax_kde.set_xlabel('F1 Score', fontsize=9)
        ax_kde.set_ylabel('Density', fontsize=9)
        ax_kde.set_title('F1 Score Distribution', fontsize=10, fontweight='bold', color=TEXT_CLR)
        ax_kde.set_xlim(0, 1.05)
        ax_kde.legend(fontsize=7, framealpha=0.25, facecolor=PANEL_BG, loc='upper left')

        sort_idx   = np.argsort(f1s)
        bar_colors = [RED if is_outlier[i] else TEAL for i in sort_idx]
        xb         = np.arange(len(names))
        bars_r     = ax_rank.bar(xb, f1s[sort_idx], color=bar_colors, width=0.7, edgecolor='none', alpha=0.9)
        non_out_f1 = f1s[normal_idx]
        non_out_med = np.median(non_out_f1) if len(non_out_f1) else median
        ax_rank.axhline(non_out_med, color=GOLD, lw=1.5, linestyle='--', label=f'Median ({non_out_med:.3f})')
        ax_rank.axhline(fence, color=RED, lw=1.2, linestyle='-.', alpha=0.7, label=f'Fence ({fence:.3f})')
        for bar, val, out in zip(bars_r, f1s[sort_idx], is_outlier[sort_idx]):
            ax_rank.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=7, color=RED if out else TEXT_CLR, fontweight='bold' if out else 'normal')
        ax_rank.set_xticks(xb)
        ax_rank.set_xticklabels([names[i] for i in sort_idx], rotation=45, ha='right', fontsize=7, color=TEXT_CLR)
        ax_rank.set_ylabel('F1 Score', fontsize=9)
        ax_rank.set_ylim(0, 1.15)
        ax_rank.set_title('Papers Ranked by F1', fontsize=10, fontweight='bold', color=TEXT_CLR)
        ax_rank.yaxis.grid(True, color=GRID_CLR, lw=0.5, linestyle='--')
        ax_rank.set_axisbelow(True)
        ax_rank.legend(fontsize=7.5, framealpha=0.25, facecolor=PANEL_BG, loc='lower right')

        for f1_val in [0.3, 0.5, 0.7, 0.9]:
            rr2 = np.linspace(0.01, 1.0, 300)
            pi2 = f1_val * rr2 / (2*rr2 - f1_val)
            v   = (pi2 > 0) & (pi2 <= 1)
            ax_pr.plot(rr2[v], pi2[v], color=GRID_CLR, lw=1.0, linestyle='--', alpha=0.7)
            if v.sum() > 0:
                mid = v.sum() // 2
                ax_pr.text(rr2[v][mid]+0.02, pi2[v][mid], f'F1={f1_val}', color='#888888', fontsize=7)
        sc2 = ax_pr.scatter(recs[normal_idx], precs[normal_idx], c=f1s[normal_idx], cmap='YlGn', vmin=0, vmax=1, s=80, zorder=5, edgecolors='#ffffff', linewidths=0.4)
        if len(outlier_idx):
            ax_pr.scatter(recs[outlier_idx], precs[outlier_idx], c=f1s[outlier_idx], cmap='YlGn', vmin=0, vmax=1, s=120, marker='D', zorder=6, edgecolors=RED, linewidths=1.5)
            for i in outlier_idx:
                ax_pr.annotate(names[i], (recs[i], precs[i]), textcoords='offset points', xytext=(4, 4), color=RED, fontsize=7, fontweight='bold')
        plt.colorbar(sc2, ax=ax_pr, pad=0.02).set_label('F1 Score', color=TEXT_CLR, fontsize=8)
        ax_pr.set_xlabel('Recall', fontsize=9)
        ax_pr.set_ylabel('Precision', fontsize=9)
        ax_pr.set_title('Precision vs Recall per Paper', fontsize=10, fontweight='bold', color=TEXT_CLR)
        ax_pr.set_xlim(-0.02, 1.05)
        ax_pr.set_ylim(-0.02, 1.1)
        ax_pr.set_axisbelow(True)

        totals = np.where(tps+fps+fns == 0, 1, tps+fps+fns)
        stk_order = np.argsort(f1s)
        xs = np.arange(len(names))
        ax_stk.bar(xs, tps[stk_order]/totals[stk_order]*100, color=GREEN_BAR,  alpha=0.9, label='TP',  edgecolor='none')
        ax_stk.bar(xs, fps[stk_order]/totals[stk_order]*100, bottom=tps[stk_order]/totals[stk_order]*100, color=ORANGE_BAR, alpha=0.9, label='FP', edgecolor='none')
        ax_stk.bar(xs, fns[stk_order]/totals[stk_order]*100, bottom=(tps[stk_order]+fps[stk_order])/totals[stk_order]*100, color=GREY_BAR, alpha=0.9, label='FN', edgecolor='none')
        ax_stk.set_xticks(xs)
        ax_stk.set_xticklabels([names[i] for i in stk_order], rotation=45, ha='right', fontsize=7, color=TEXT_CLR)
        ax_stk.set_ylabel('Percentage (%)', fontsize=9)
        ax_stk.set_ylim(0, 115)
        ax_stk.set_title('TP / FP / FN Breakdown per Paper', fontsize=10, fontweight='bold', color=TEXT_CLR)
        ax_stk.yaxis.grid(True, color=GRID_CLR, lw=0.4, linestyle='--')
        ax_stk.set_axisbelow(True)
        ax_stk.legend(fontsize=8, framealpha=0.25, facecolor=PANEL_BG, loc='upper right')

        out2 = os.path.join(output_dir, 'per_paper_distribution.png')
        plt.savefig(out2, dpi=250, bbox_inches='tight', facecolor=BG)
        plt.close()
        print(f"Per-paper distribution saved to: {out2}")


if __name__ == "__main__":
    print("PART 1 — GT COMPARISON")
    print(f"Ground Truth Dir : {GROUND_TRUTH_DIR}")
    print(f"Predictions Dir  : {PREDICTIONS_DIR}")
    print(f"Results Dir      : {RESULTS_DIR}\n")

    result = evaluate_gt(GROUND_TRUTH_DIR, PREDICTIONS_DIR)

    if result is not None:
        metrics, gt_by_type, pred_by_type, per_file_stats = result
        print_results_table(metrics)
        save_gt_results(metrics, gt_by_type, pred_by_type, per_file_stats, RESULTS_DIR)
        plot_gt_results(metrics, gt_by_type, pred_by_type, per_file_stats, RESULTS_DIR)
        print("\nPART 1 COMPLETE.")
    else:
        print("Evaluation failed.")
