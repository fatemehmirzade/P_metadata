import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


def clean_text_for_json(text: str) -> str:
    if not text:
        return ""
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', '  ', text)
    return text.strip()


def find_section_boundaries(text: str) -> Dict[str, Tuple[int, int]]:
    text_lower = text.lower()

    section_markers = {
        'abstract': [
            r'===\s*abstract\s*===',
            r'\n\*?\s*abstract\s*\n',
            r'\n\d+\.?\s*abstract\s*\n',
        ],
        'introduction': [
            r'===\s*intro(?:duction)?\s*===',
            r'\n\*?\s*introduction\s*\n',
            r'\n\d+\.?\s*introduction\s*\n',
            r'\n\*?\s*background\s*\n',
            r'\n\d+\.?\s*background\s*\n',
        ],
        'methods': [
            r'===\s*methods?\s*===',
            r'\n\*?\s*methods?\s*\n',
            r'\n\d+\.?\s*methods?\s*\n',
            r'\n\*?\s*online\s+methods?\s*\n',
            r'\n\d+\.?\s*online\s+methods?\s*\n',
            r'\n\*?\s*materials?\s+and\s+methods?\s*\n',
            r'\n\d+\.?\s*materials?\s+and\s+methods?\s*\n',
            r'\n\*?\s*experimental\s+(?:procedures?|design|section)\s*\n',
            r'\n\d+\.?\s*experimental\s+(?:procedures?|design|section)\s*\n',
        ],
        'results': [
            r'===\s*results?\s*===',
            r'\n\*?\s*results?\s*\n',
            r'\n\d+\.?\s*results?\s*\n',
        ],
        'discussion': [
            r'===\s*discussion\s*===',
            r'\n\*?\s*discussion\s*\n',
            r'\n\d+\.?\s*discussion\s*\n',
        ],
        'conclusions': [
            r'===\s*conclusions?\s*===',
            r'\n\*?\s*conclusions?\s*\n',
            r'\n\d+\.?\s*conclusions?\s*\n',
        ],
        'supplementary': [
            r'===\s*supplement(?:ary)?(?:\s+(?:information|material|data|methods?))?\s*===',
            r'\n\*?\s*supplementary\s+(?:information|material|data|methods?)\s*\n',
            r'\n\*?\s*supplementary\s*\n',
            r'\n\d+\.?\s*supplementary\s+(?:information|material|data|methods?)\s*\n',
            r'\n\*?\s*supporting\s+information\s*\n',
            r'\n\d+\.?\s*supporting\s+information\s*\n',
            r'\n\*?\s*appendix\s*\n',
            r'\n\d+\.?\s*appendix\s*\n',
        ],
        'references': [
            r'===\s*references?\s*===',
            r'\n\*?\s*references?\s*\n',
            r'\n\d+\.?\s*references?\s*\n',
            r'\n\*?\s*bibliography\s*\n',
        ],
        'acknowledgements': [
            r'\n\*?\s*acknowledgements?\s*\n',
            r'\n\*?\s*acknowledgments?\s*\n',
            r'\n\*?\s*funding\s*\n',
            r'\n\*?\s*author\s+contributions?\s*\n',
        ]
    }

    found_sections = {}

    for section_name, patterns in section_markers.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                match = matches[0]
                found_sections[section_name] = match.start()
                break

    sorted_sections = sorted(found_sections.items(), key=lambda x: x[1])

    boundaries = {}
    for i, (section_name, start_pos) in enumerate(sorted_sections):
        if i < len(sorted_sections) - 1:
            end_pos = sorted_sections[i + 1][1]
        else:
            end_pos = len(text)

        boundaries[section_name] = (start_pos, end_pos)

    return boundaries


def extract_section_by_boundaries(text: str, section_name: str, boundaries: Dict[str, Tuple[int, int]]) -> Optional[str]:
    if section_name not in boundaries:
        return None

    start_pos, end_pos = boundaries[section_name]
    section_text = text[start_pos:end_pos].strip()

    lines = section_text.split('\n')
    if len(lines) > 1:
        content_lines = []
        skip_header = True
        for line in lines:
            if skip_header:
                if line.strip() and not re.match(r'^(===|[\d\.]+\s*|\*\s*)?[A-Za-z\s]+$', line.strip()):
                    skip_header = False
                    content_lines.append(line)
            else:
                content_lines.append(line)

        section_text = '\n'.join(content_lines).strip()

    return section_text if len(section_text) > 50 else None


def extract_implicit_abstract(text: str, boundaries: Dict[str, Tuple[int, int]]) -> Optional[str]:
    major_sections = ['introduction', 'methods', 'results', 'discussion']
    earliest_pos = len(text)

    for section in major_sections:
        if section in boundaries:
            earliest_pos = min(earliest_pos, boundaries[section][0])

    if earliest_pos < len(text):
        abstract_text = text[:earliest_pos].strip()

        lines = abstract_text.split('\n')

        content_start = 0
        for i, line in enumerate(lines):
            if len(line.strip()) > 100:
                content_start = i
                break

        if content_start > 0 and len(lines) > content_start:
            abstract_text = '\n'.join(lines[content_start:]).strip()
        elif content_start == 0 and len(lines) > 1:
            abstract_text = '\n'.join(lines[1:]).strip()

        if len(abstract_text) > 200:
            return abstract_text

    return None


def extract_all_sections(text: str) -> Tuple[Dict[str, Optional[str]], Dict]:
    sections = {
        'abstract': None,
        'methods': None,
        'supplementary': None
    }

    extraction_info = {
        'sections_found': [],
        'sections_not_found': [],
        'abstract_chars': 0,
        'methods_chars': 0,
        'supplementary_chars': 0,
        'total_chars': 0,
        'methods_source': 'methods'
    }

    boundaries = find_section_boundaries(text)

    abstract = extract_section_by_boundaries(text, 'abstract', boundaries)

    if not abstract:
        abstract = extract_implicit_abstract(text, boundaries)

    if abstract:
        sections['abstract'] = abstract
        extraction_info['sections_found'].append('abstract')
        extraction_info['abstract_chars'] = len(abstract)
        extraction_info['total_chars'] += len(abstract)
    else:
        extraction_info['sections_not_found'].append('abstract')

    methods = extract_section_by_boundaries(text, 'methods', boundaries)
    if methods:
        sections['methods'] = methods
        extraction_info['sections_found'].append('methods')
        extraction_info['methods_chars'] = len(methods)
        extraction_info['total_chars'] += len(methods)
        extraction_info['methods_source'] = 'methods'
    else:
        results = extract_section_by_boundaries(text, 'results', boundaries)
        if results:
            sections['methods'] = results
            extraction_info['sections_found'].append('methods_from_results')
            extraction_info['methods_chars'] = len(results)
            extraction_info['total_chars'] += len(results)
            extraction_info['methods_source'] = 'results'
        else:
            extraction_info['sections_not_found'].append('methods')
            extraction_info['methods_source'] = 'none'

    supplementary = extract_section_by_boundaries(text, 'supplementary', boundaries)
    if supplementary:
        sections['supplementary'] = supplementary
        extraction_info['sections_found'].append('supplementary')
        extraction_info['supplementary_chars'] = len(supplementary)
        extraction_info['total_chars'] += len(supplementary)
    else:
        extraction_info['sections_not_found'].append('supplementary')

    return sections, extraction_info


def prepare_dataset(input_dir: str, output_file: str = "papers_dataset.json"):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    data = []
    txt_files = sorted(list(input_path.glob("*.txt")))

    print(f"PROTEOMICS PAPER DATASET PREPARATION")
    print(f"Input directory: {input_path}")
    print(f"Found {len(txt_files)} text files to process\n")

    stats = {
        'total': 0,
        'with_abstract': 0,
        'with_methods': 0,
        'with_methods_from_results': 0,
        'with_supplementary': 0,
        'with_all_three': 0,
        'with_abstract_methods': 0,
        'with_none': 0
    }

    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            text = clean_text_for_json(text)

            sections, extraction_info = extract_all_sections(text)

            stats['total'] += 1

            has_abstract = bool(sections['abstract'])
            has_methods = bool(sections['methods'])
            has_supplementary = bool(sections['supplementary'])
            methods_from_results = extraction_info.get('methods_source') == 'results'

            if has_abstract:
                stats['with_abstract'] += 1
            if has_methods:
                stats['with_methods'] += 1
                if methods_from_results:
                    stats['with_methods_from_results'] += 1
            if has_supplementary:
                stats['with_supplementary'] += 1

            if has_abstract and has_methods and has_supplementary:
                stats['with_all_three'] += 1
            elif has_abstract and has_methods:
                stats['with_abstract_methods'] += 1
            elif not (has_abstract or has_methods or has_supplementary):
                stats['with_none'] += 1

            doc = {
                "filename": file_path.name,
                "stem": file_path.stem,
                "abstract": clean_text_for_json(sections['abstract']) if sections['abstract'] else None,
                "methods": clean_text_for_json(sections['methods']) if sections['methods'] else None,
                "supplementary": clean_text_for_json(sections['supplementary']) if sections['supplementary'] else None,
                "extraction_info": extraction_info
            }

            data.append(doc)

            found_sections = extraction_info['sections_found']
            if found_sections:
                status_parts = []
                if 'abstract' in found_sections:
                    status_parts.append(f"Abstract({extraction_info['abstract_chars']:,} chars)")
                if 'methods' in found_sections:
                    status_parts.append(f"Methods({extraction_info['methods_chars']:,} chars)")
                elif 'methods_from_results' in found_sections:
                    status_parts.append(f"Methods[from Results]({extraction_info['methods_chars']:,} chars)")
                if 'supplementary' in found_sections:
                    status_parts.append(f"Supplementary({extraction_info['supplementary_chars']:,} chars)")
                status_msg = " + ".join(status_parts)
                symbol = 'OK'
            else:
                status_msg = "No sections found"
                symbol = 'FAIL'

            print(f"  [{symbol}] {file_path.name:<25s}: {status_msg}")

        except Exception as e:
            print(f"  [FAIL] {file_path.name:<25s}: Error - {e}")
            continue

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nEXTRACTION SUMMARY")
    print(f"Output file: {output_path}")
    print(f"\nDocuments processed: {stats['total']}")
    print(f"\nSection Coverage:")
    print(f"  Abstract:              {stats['with_abstract']:>3d} / {stats['total']:>3d} ({stats['with_abstract']/stats['total']*100:>5.1f}%)")
    print(f"  Methods:               {stats['with_methods']:>3d} / {stats['total']:>3d} ({stats['with_methods']/stats['total']*100:>5.1f}%)")
    if stats['with_methods_from_results'] > 0:
        print(f"    (from Results):      {stats['with_methods_from_results']:>3d} / {stats['with_methods']:>3d} ({stats['with_methods_from_results']/stats['with_methods']*100:>5.1f}%)")
    print(f"  Supplementary:         {stats['with_supplementary']:>3d} / {stats['total']:>3d} ({stats['with_supplementary']/stats['total']*100:>5.1f}%)")
    print(f"\nDocument Completeness:")
    print(f"  All three sections:    {stats['with_all_three']:>3d} / {stats['total']:>3d} ({stats['with_all_three']/stats['total']*100:>5.1f}%)")
    print(f"  Abstract + Methods:    {stats['with_abstract_methods']:>3d} / {stats['total']:>3d} ({stats['with_abstract_methods']/stats['total']*100:>5.1f}%)")
    print(f"  No sections found:     {stats['with_none']:>3d} / {stats['total']:>3d} ({stats['with_none']/stats['total']*100:>5.1f}%)")

    return output_path


if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "text_files_Papers_Ian"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "papers_dataset.json"

    try:
        output_path = prepare_dataset(input_dir, output_file)
        print(f"Dataset successfully created: {output_path}")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)