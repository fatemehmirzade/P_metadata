import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import json

RESULTS_DIR      = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/evaluation_results_gpt/Source_verification"
#GROUND_TRUTH_DIR = "/Users/fateme/Downloads/Ians_Annotations"
#PREDICTIONS_DIR  = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/filtered_output"
#GROUND_TRUTH_DIR = "/Users/fateme/Downloads/Ians_Annotations/27"
#PREDICTIONS_DIR  = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run8_27/filtered_output"
#RESULTS_DIR      = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run8_27/evaluation_results_gpt/Source_verification"
GROUND_TRUTH_DIR = "/Users/fateme/Downloads/Ians_Annotations"
PREDICTIONS_DIR  = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/filtered_output"
PAPERS_JSON         = "/Users/fateme/Desktop/test_metadata/map1/Map_gpt5.2_N_prompt/run9/papers_dataset.json"
HALLUCINATION_THRESHOLD = 85

os.makedirs(RESULTS_DIR, exist_ok=True)

_GREEK_MAP = str.maketrans({
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e',
    'μ': 'u', 'µ': 'u', 'σ': 's', 'τ': 't', 'ω': 'w',
    'Α': 'a', 'Β': 'b', 'Γ': 'g', 'Δ': 'd',
})

_UNIT_SYNONYMS: dict[str, str] = {
    'micromolar': 'um', 'micromol/l': 'um', 'µm': 'um',
    'nanomolar': 'nm',  'nanomol/l': 'nm',
    'millimolar': 'mm', 'millimol/l': 'mm',
    'picomolar': 'pm',  'femtomolar': 'fm',
    'millilitre': 'ml', 'milliliter': 'ml',
    'microlitre': 'ul', 'microliter': 'ul', 'µl': 'ul',
    'nanolitre': 'nl',  'nanoliter': 'nl',
    'picolitre': 'pl',  'picoliter': 'pl',
    'microgram': 'ug',  'µg': 'ug',
    'nanogram': 'ng',   'milligram': 'mg',
    'nanolitres/minute': 'nl/min', 'nanoliters/minute': 'nl/min',
    'microlitres/minute': 'ul/min', 'microliters/minute': 'ul/min',
    'electron volt': 'ev', 'electron volts': 'ev', 'kiloelectron volt': 'kev',
    'minute': 'min', 'minutes': 'min', 'second': 'sec', 'seconds': 'sec',
    'hour': 'h', 'hours': 'h',
    'percent': '%', 'percentage': '%', 'parts per million': 'ppm',
}

_UNIT_SYN_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(_UNIT_SYNONYMS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def normalize_text(text: str) -> str:
    return ' '.join(text.split()).lower().strip()


def should_exclude_entity(entity_type: str) -> bool:
    return entity_type.lower().strip().startswith('factorvalue')


def _normalize_greek(text: str) -> str:
    return text.translate(_GREEK_MAP)


def _normalise_units(text: str) -> str:
    return _UNIT_SYN_RE.sub(lambda m: _UNIT_SYNONYMS[m.group(1).lower()], text)


def _extract_numeric_core(text: str) -> str | None:
    norm = normalize_text(text).strip()
    m = re.match(r'^(\d+(?:\.\d+)?)', norm)
    if m:
        num = m.group(1)
        digits_only = num.replace('.', '')
        return num if len(digits_only) >= 2 else None
    return None


def check_hallucination(
    predicted_value: str,
    source_text: str,
    threshold: int = HALLUCINATION_THRESHOLD,
    entity_type: str = "",
) -> tuple[bool, float]:
    if not source_text or not predicted_value:
        return False, 0.0

    pred_norm   = _normalise_units(_normalize_greek(normalize_text(predicted_value)))
    source_norm = _normalise_units(_normalize_greek(normalize_text(source_text)))

    if pred_norm in source_norm:
        return True, 1.0

    num_unit_match = re.match(
        r'^(\d+(?:\.\d+)?)\s*(%|[a-z][a-z0-9%µ/\-]*)$',
        pred_norm
    )
    if num_unit_match:
        num_part  = num_unit_match.group(1)
        unit_part = num_unit_match.group(2)
        unit_pat  = re.escape(unit_part)
        pattern   = re.compile(r'\b' + re.escape(num_part) + r'\b.{0,15}' + unit_pat, re.IGNORECASE)
        if pattern.search(source_norm):
            return True, 0.95
        pattern_rev = re.compile(unit_pat + r'.{0,15}\b' + re.escape(num_part) + r'\b', re.IGNORECASE)
        if pattern_rev.search(source_norm):
            return True, 0.93
        return False, 0.0

    numeric_core = _extract_numeric_core(pred_norm)
    if numeric_core:
        standalone = re.compile(r'(?<!\d)' + re.escape(numeric_core) + r'(?!\d)')
        if standalone.search(source_norm):
            return True, 0.88

    pred_words    = pred_norm.split()
    source_words  = source_norm.split()
    anchor_tokens = {tok for tok in pred_words if len(tok) >= 4}

    window_sizes = sorted({max(1, len(pred_words)), max(1, len(pred_words)+2), max(1, len(pred_words)+5)})

    max_score = 0
    for window_size in window_sizes:
        if window_size > len(source_words):
            break
        for i in range(len(source_words) - window_size + 1):
            chunk = ' '.join(source_words[i: i + window_size])
            if anchor_tokens and not any(tok in chunk for tok in anchor_tokens):
                continue
            score = fuzz.token_sort_ratio(pred_norm, chunk)
            if score > max_score:
                max_score = score
            if max_score >= threshold:
                return True, max_score / 100.0

    if len(pred_words) <= 2:
        for word in source_words:
            score = fuzz.ratio(pred_norm, _normalize_greek(word))
            if score > max_score:
                max_score = score
            if max_score >= threshold:
                return True, max_score / 100.0

    return max_score >= threshold, max_score / 100.0


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


def load_papers_dataset(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        papers_dict = {}
        for paper in papers:
            paper_id = paper.get('filename') or paper.get('stem', '')
            if not paper_id:
                continue
            key = paper_id.replace('.txt', '').strip()
            papers_dict[key] = {
                'abstract':      paper.get('abstract')      or '',
                'methods':       paper.get('methods')       or '',
                'supplementary': paper.get('supplementary') or '',
            }
        print(f"  Loaded {len(papers_dict)} papers")
        return papers_dict
    except FileNotFoundError:
        print(f"Warning: papers_dataset.json not found at: {json_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not load papers dataset: {e}")
        return {}


def run_source_verification(pred_dir, papers_json, threshold):
    pred_files  = {f.stem: f for f in Path(pred_dir).glob('*.ann')}
    papers_dict = load_papers_dataset(papers_json)

    stats = {
        'total_checked': 0, 'fully_supported': 0,
        'partially_supported': 0, 'hallucinated': 0,
        'hallucinated_items': [], 'partially_supported_items': []
    }
    per_file_rows = []

    for file_name in sorted(pred_files.keys()):
        pred_entities = parse_ann_file(pred_files[file_name])
        pred_filtered = [(t, txt, ln) for t, txt, ln in pred_entities if not should_exclude_entity(t)]

        source_sections = papers_dict.get(file_name, {})
        source_text = "\n\n".join([
            f"### {section.upper()}\n{text}"
            for section, text in source_sections.items() if text
        ])

        if not source_text:
            print(f"  WARNING: no source text for {file_name}")

        file_hallucinations  = 0
        file_partial_support = 0
        file_fully_supported = 0

        for entity_type, pred_text, pred_line in pred_filtered:
            is_found, confidence = check_hallucination(pred_text, source_text, threshold, entity_type)
            stats['total_checked'] += 1

            if confidence >= 0.85:
                stats['fully_supported'] += 1
                file_fully_supported += 1
            elif confidence >= 0.70:
                stats['partially_supported'] += 1
                file_partial_support += 1
                stats['partially_supported_items'].append(f"{file_name} - {entity_type}: {pred_text} (conf: {confidence:.2f})")
            else:
                stats['hallucinated'] += 1
                file_hallucinations += 1
                stats['hallucinated_items'].append(f"{file_name} - {entity_type}: {pred_text} (conf: {confidence:.2f})")

            per_file_rows.append({
                'file_name':   file_name,
                'entity_type': entity_type,
                'pred_text':   pred_text,
                'pred_line':   pred_line,
                'confidence':  round(confidence, 4),
                'verdict':     'supported' if confidence >= 0.85 else 'partial' if confidence >= 0.70 else 'hallucinated',
            })

        total_file = file_fully_supported + file_partial_support + file_hallucinations
        print(f"  {file_name}: {total_file} checked | supported={file_fully_supported} | partial={file_partial_support} | hallucinated={file_hallucinations}")

    return stats, per_file_rows


def save_verification_results(stats, per_file_rows, output_dir):
    hs    = stats
    total = hs['total_checked']

    rows = [
        {'Metric': 'Total Predictions Checked',    'Value': str(total)},
        {'Metric': 'Fully Supported (>=85%)',       'Value': str(hs['fully_supported'])},
        {'Metric': 'Partially Supported (70-84%)',  'Value': str(hs['partially_supported'])},
        {'Metric': 'Hallucinated (<70%)',           'Value': str(hs['hallucinated'])},
        {'Metric': 'Hallucination Rate',            'Value': f"{hs['hallucinated']/total:.2%}" if total > 0 else '0%'},
        {'Metric': 'Full Support Rate',             'Value': f"{hs['fully_supported']/total:.2%}" if total > 0 else '0%'},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'source_verification_summary.csv'), index=False, escapechar='\\', doublequote=True)

    if per_file_rows:
        pd.DataFrame(per_file_rows).to_csv(os.path.join(output_dir, 'source_verification_details.csv'), index=False, escapechar='\\', doublequote=True)

    rpt = os.path.join(output_dir, 'hallucination_report.txt')
    with open(rpt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\nHALLUCINATION DETECTION REPORT\n" + "="*80 + "\n\n")
        f.write(f"Total Predictions Checked: {total}\n")
        f.write(f"Fully Supported  (>=85%):  {hs['fully_supported']}\n")
        f.write(f"Partially Supported (70-84%): {hs['partially_supported']}\n")
        f.write(f"Hallucinated (<70%):  {hs['hallucinated']}\n")
        if total > 0:
            f.write(f"\nHallucination Rate: {hs['hallucinated']/total:.2%}\n")
        if hs['hallucinated_items']:
            f.write(f"\n{'='*80}\nHALLUCINATED ITEMS\n{'='*80}\n")
            for i, item in enumerate(hs['hallucinated_items'], 1):
                f.write(f"{i}. {item}\n")
        if hs['partially_supported_items']:
            f.write(f"\n{'='*80}\nPARTIALLY SUPPORTED ITEMS\n{'='*80}\n")
            for i, item in enumerate(hs['partially_supported_items'], 1):
                f.write(f"{i}. {item}\n")

    if hs['hallucinated_items']:
        pd.DataFrame([{'Item': i} for i in hs['hallucinated_items']]).to_csv(
            os.path.join(output_dir, 'hallucinated_items.csv'), index=False, escapechar='\\', doublequote=True)

    if hs['partially_supported_items']:
        pd.DataFrame([{'Item': i} for i in hs['partially_supported_items']]).to_csv(
            os.path.join(output_dir, 'partially_supported_items.csv'), index=False, escapechar='\\', doublequote=True)

    print(f"Verification results saved to: {output_dir}")


def plot_verification_results(stats, per_file_rows, output_dir):
    hs    = stats
    total = hs['total_checked']
    if total == 0:
        print("No data to plot.")
        return

    vals   = [hs['fully_supported'], hs['partially_supported'], hs['hallucinated']]
    cats   = ['Fully\nSupported\n(>=85%)', 'Partially\nSupported\n(70-84%)', 'Hallucinated\n(<70%)']
    colors = ['#4CAF50', '#FFC107', '#f44336']
    labels = [f'Fully Supported\n(>=85%)\n{vals[0]} items', f'Partially Supported\n(70-84%)\n{vals[1]} items', f'Hallucinated\n(<70%)\n{vals[2]} items']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    wedges, texts, autotexts = axes[0].pie(vals, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05, 0.1), textprops={'fontsize': 10})
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontsize(11)
    axes[0].set_title('Source Support Distribution', fontsize=13, fontweight='bold', pad=20)

    bars = axes[1].bar(cats, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.3)
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].set_title('Hallucination Detection Results', fontsize=13, fontweight='bold', pad=20)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars, vals):
        pct = v/total*100
        axes[1].text(bar.get_x()+bar.get_width()/2., bar.get_height(), f'{v}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].text(0.5, 0.95, f'Total: {total}\nHallucination Rate: {vals[2]/total*100:.1f}%\nFull Support Rate: {vals[0]/total*100:.1f}%', transform=axes[1].transAxes, fontsize=10, va='top', ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    if per_file_rows:
        df     = pd.DataFrame(per_file_rows)
        counts = df.groupby(['file_name', 'verdict']).size().unstack(fill_value=0)
        for col in ['supported', 'partial', 'hallucinated']:
            if col not in counts.columns:
                counts[col] = 0
        counts = counts[['supported', 'partial', 'hallucinated']]
        x  = np.arange(len(counts))
        w2 = 0.25
        axes[2].bar(x - w2,   counts['supported'],   w2, label='Supported',   color='#4CAF50', alpha=0.85)
        axes[2].bar(x,        counts['partial'],      w2, label='Partial',     color='#FFC107', alpha=0.85)
        axes[2].bar(x + w2,   counts['hallucinated'], w2, label='Hallucinated',color='#f44336', alpha=0.85)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
        axes[2].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[2].set_title('Per-Paper Verification Breakdown', fontsize=13, fontweight='bold', pad=20)
        axes[2].legend(fontsize=10)
        axes[2].grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Source Text Verification Results', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, 'source_verification.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {out}")


if __name__ == "__main__":
    print("PART 2 — SOURCE VERIFICATION")
    print(f"Predictions Dir : {PREDICTIONS_DIR}")
    print(f"Papers JSON     : {PAPERS_JSON}")
    print(f"Results Dir     : {RESULTS_DIR}\n")

    stats, per_file_rows = run_source_verification(PREDICTIONS_DIR, PAPERS_JSON, HALLUCINATION_THRESHOLD)

    hs    = stats
    total = hs['total_checked']
    print(f"\nTotal checked    : {total}")
    print(f"Fully supported  : {hs['fully_supported']}")
    print(f"Partially suppt  : {hs['partially_supported']}")
    print(f"Hallucinated     : {hs['hallucinated']}")
    if total > 0:
        print(f"Hallucination rate: {hs['hallucinated']/total:.2%}")

    save_verification_results(stats, per_file_rows, RESULTS_DIR)
    plot_verification_results(stats, per_file_rows, RESULTS_DIR)
    print("\nPART 2 COMPLETE.")
