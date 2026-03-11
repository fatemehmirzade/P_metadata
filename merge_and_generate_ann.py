import json
import os
from pathlib import Path
from collections import defaultdict


def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_outputs(output_files):
    merged = defaultdict(dict)

    for category, filepath in output_files.items():
        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filepath}")
            continue

        data = load_json_file(filepath)

        for entry in data:
            filename = entry.get('filename')
            if not filename:
                continue

            cleaned_entry = {
                k: v for k, v in entry.items()
                if not k.startswith('_') and k != 'filename' and k != 'stem'
            }
            merged[filename][category] = cleaned_entry

    return dict(merged)


def generate_simple_annotations(paper_data, filename):
    annotations = []

    field_mappings = {
        'Organism': ('biological', 'Organism'),
        'Strain': ('biological', 'Strain'),
        'Age': ('biological', 'Age'),
        'Sex': ('biological', 'Sex'),
        'OrganismPart': ('biological', 'OrganismPart'),
        'MaterialType': ('biological', 'MaterialType'),
        'Specimen': ('biological', 'Specimen'),
        'AncestryCategory': ('biological', 'AncestryCategory'),
        'DevelopmentalStage': ('biological', 'DevelopmentalStage'),
        'Genotype': ('biological', 'Genotype'),
        'GeneticModification': ('biological', 'GeneticModification'),
        'Treatment': ('biological', 'Treatment'),
        'Instrument': ('ms_instruments', 'Instrument'),
        'AcquisitionMethod': ('ms_instruments', 'AcquisitionMethod'),
        'IonizationType': ('ms_instruments', 'IonizationType'),
        'FragmentationMethod': ('ms_instruments', 'FragmentationMethod'),
        'MS2MassAnalyzer': ('ms_instruments', 'MS2MassAnalyzer'),
        'CollisionEnergy': ('ms_instruments', 'CollisionEnergy'),
        'PrecursorMassTolerance': ('ms_instruments', 'PrecursorMassTolerance'),
        'FragmentMassTolerance': ('ms_instruments', 'FragmentMassTolerance'),
        'Label': ('sample_prep', 'Label'),
        'CleavageAgent': ('sample_prep', 'CleavageAgent'),
        'AlkylationReagent': ('sample_prep', 'AlkylationReagent'),
        'ReductionReagent': ('sample_prep', 'ReductionReagent'),
        'Depletion': ('sample_prep', 'Depletion'),
        'Modification': ('sample_prep', 'Modification'),
        'SpikedCompound': ('sample_prep', 'SpikedCompound'),
        'SyntheticPeptide': ('sample_prep', 'SyntheticPeptide'),
        'Staining': ('sample_prep', 'Staining'),
        'Separation': ('separation', 'Separation'),
        'FractionationMethod': ('separation', 'FractionationMethod'),
        'FractionIdentifier': ('separation', 'FractionIdentifier'),
        'NumberOfFractions': ('separation', 'NumberOfFractions'),
        'EnrichmentMethod': ('separation', 'EnrichmentMethod'),
        'Bait': ('separation', 'Bait'),
        'FlowRateChromatogram': ('separation', 'FlowRateChromatogram'),
        'GradientTime': ('separation', 'GradientTime'),
        'NumberOfMissedCleavages': ('data_analysis', 'NumberOfMissedCleavages'),
        'NumberOfTechnicalReplicates': ('data_analysis', 'NumberOfTechnicalReplicates'),
        'NumberOfBiologicalReplicates': ('data_analysis', 'NumberOfBiologicalReplicates'),
        'NumberOfSamples': ('data_analysis', 'NumberOfSamples'),
        'BiologicalReplicate': ('data_analysis', 'BiologicalReplicate'),
        'PooledSample': ('data_analysis', 'PooledSample'),
        'Disease': ('clinical', 'Disease'),
        'DiseaseTreatment': ('clinical', 'DiseaseTreatment'),
        'CellLine': ('clinical', 'CellLine'),
        'CellType': ('clinical', 'CellType'),
        'CellPart': ('clinical', 'CellPart'),
        'GrowthRate': ('clinical', 'GrowthRate'),
        'SamplingTime': ('clinical', 'SamplingTime'),
        'BMI': ('clinical', 'BMI'),
        'AnatomicSiteTumor': ('clinical', 'AnatomicSiteTumor'),
        'OriginSiteDisease': ('clinical', 'OriginSiteDisease'),
        'TumorCellularity': ('clinical', 'TumorCellularity'),
        'TumorGrade': ('clinical', 'TumorGrade'),
        'TumorStage': ('clinical', 'TumorStage'),
        'TumorSize': ('clinical', 'TumorSize'),
        'TumorSite': ('clinical', 'TumorSite'),
        'FactorValue': ('factor_values', 'FactorValue'),
        'Time': ('factor_values', 'Time'),
        'Temperature': ('factor_values', 'Temperature'),
        'Compound': ('factor_values', 'Compound'),
    }

    for entity_type, (category, field_name) in sorted(field_mappings.items()):
        category_data = paper_data.get(category, {})
        if not category_data:
            continue

        values = category_data.get(field_name, [])

        if not values:
            continue

        if not isinstance(values, list):
            values = [values]

        for value in values:
            if not value:
                continue

            if isinstance(value, str) and not value.strip():
                continue

            annotations.append(f"{entity_type}: {value}")

    return annotations


def save_simple_ann_files(merged_data, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats = {
        'total_papers': len(merged_data),
        'papers_with_annotations': 0,
        'total_annotations': 0,
        'empty_papers': []
    }

    for filename, paper_data in sorted(merged_data.items()):
        base_name = Path(filename).stem
        ann_path = os.path.join(output_dir, f"{base_name}.ann")

        annotations = generate_simple_annotations(paper_data, filename)

        if annotations:
            with open(ann_path, 'w', encoding='utf-8') as f:
                for ann in annotations:
                    f.write(ann + '\n')
            stats['papers_with_annotations'] += 1
            stats['total_annotations'] += len(annotations)
        else:
            with open(ann_path, 'w', encoding='utf-8') as f:
                pass
            stats['empty_papers'].append(filename)

    return stats


def print_annotation_statistics(merged_data, output_dir):
    entity_counts = defaultdict(int)
    papers_with_entity = defaultdict(set)
    total_papers = len(merged_data)

    for filename, paper_data in merged_data.items():
        annotations = generate_simple_annotations(paper_data, filename)

        for annotation in annotations:
            entity_type = annotation.split(':', 1)[0].strip()
            entity_counts[entity_type] += 1
            papers_with_entity[entity_type].add(filename)

    print("ANNOTATION STATISTICS BY ENTITY TYPE")
    print(f"{'Entity Type':<40s} {'Papers':>8s} {'Coverage':>10s} {'Total':>10s}")

    for entity_type in sorted(entity_counts.keys()):
        num_papers = len(papers_with_entity[entity_type])
        coverage = (num_papers / total_papers * 100) if total_papers > 0 else 0
        total_count = entity_counts[entity_type]
        print(f"{entity_type:<40s} {num_papers:>8d} {coverage:>9.1f}% {total_count:>10d}")

    critical_annotations = {
        'Organism': 80,
        'Instrument': 95,
        'CleavageAgent': 90,
        'AcquisitionMethod': 85,
        'FragmentationMethod': 85,
        'MS2MassAnalyzer': 85,
    }

    print("\nCRITICAL ANNOTATION HEALTH CHECK")
    print(f"{'Status':<8s} {'Entity Type':<40s} {'Actual':>8s} {'Expected':>8s}")

    warnings = []
    for ann_type, expected_rate in sorted(critical_annotations.items()):
        num_papers_with = len(papers_with_entity.get(ann_type, set()))
        actual_rate = (num_papers_with / total_papers * 100) if total_papers > 0 else 0
        status = "[OK]" if actual_rate >= expected_rate else "[WARN]"
        print(f"{status:<8s} {ann_type:<40s} {actual_rate:>7.1f}% {expected_rate:>7.0f}%")

        if actual_rate < expected_rate:
            warnings.append(f"  - {ann_type}: {actual_rate:.1f}% (expected >{expected_rate}%)")

    if warnings:
        print("\nWARNINGS - Low Coverage Detected:")
        for warning in warnings:
            print(warning)


def verify_input_files(output_files):
    print("VERIFYING INPUT FILES")

    all_exist = True
    for category, filepath in output_files.items():
        if os.path.exists(filepath):
            try:
                data = load_json_file(filepath)
                print(f"  [OK] {category:20s}: {len(data):4d} papers - {filepath}")
            except Exception as e:
                print(f"  [ERROR] {category:20s}: Cannot read file - {e}")
                all_exist = False
        else:
            print(f"  [MISSING] {category:20s}: {filepath}")
            all_exist = False

    return all_exist


def main():
    input_base_path = './output'
    output_base_path = './annotations'

    output_files = {
        'biological': os.path.join(input_base_path, 'biological_info_complete.json'),
        'ms_instruments': os.path.join(input_base_path, 'ms_instruments_complete.json'),
        'sample_prep': os.path.join(input_base_path, 'sample_prep_complete.json'),
        'separation': os.path.join(input_base_path, 'separation_complete.json'),
        'data_analysis': os.path.join(input_base_path, 'data_analysis_complete.json'),
        'clinical': os.path.join(input_base_path, 'clinical_experimental_complete.json'),
        'factor_values': os.path.join(input_base_path, 'factor_values_complete.json'),
    }

    print("PROTEOMICS METADATA ANNOTATION GENERATOR v7.0")
    print(f"Input directory:  {input_base_path}")
    print(f"Output directory: {output_base_path}")

    if not verify_input_files(output_files):
        print("\nERROR: Cannot proceed - missing or unreadable input files!")
        print("Please ensure all pipeline outputs are in the input directory.")
        print("Expected files:")
        for category, filepath in output_files.items():
            print(f"  - {os.path.basename(filepath)}")
        return

    print("\nMERGING PIPELINE OUTPUTS")
    merged_data = merge_outputs(output_files)
    print(f"  [OK] Successfully merged data for {len(merged_data)} papers")

    if not merged_data:
        print("\nERROR: No data to process!")
        return

    print("\nGENERATING .ANN FILES")
    stats = save_simple_ann_files(merged_data, output_base_path)
    print(f"  [OK] Generated .ann files for {stats['papers_with_annotations']} papers")

    print_annotation_statistics(merged_data, output_dir=output_base_path)

    if stats['empty_papers']:
        print(f"\nWARNING: {len(stats['empty_papers'])} PAPERS WITH ZERO ANNOTATIONS")
        print("These papers may have extraction issues:")
        for paper in stats['empty_papers'][:10]:
            print(f"  - {paper}")
        if len(stats['empty_papers']) > 10:
            print(f"  ... and {len(stats['empty_papers']) - 10} more")

    if merged_data:
        print("\nSAMPLE OUTPUT - First Paper")
        sample_filename = sorted(merged_data.keys())[0]
        base_name = Path(sample_filename).stem
        ann_path = os.path.join(output_base_path, f"{base_name}.ann")
        if os.path.exists(ann_path):
            with open(ann_path, encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:30], 1):
                    print(f"  {i:2d}. {line.rstrip()}")
                if len(lines) > 30:
                    print(f"  ... ({len(lines) - 30} more annotations)")
        print(f"\nFull file: {ann_path}")

    print("\nGENERATION SUMMARY")
    print(f"  Papers processed:      {stats['total_papers']}")
    print(f"  Papers with data:      {stats['papers_with_annotations']}")
    print(f"  Papers with no data:   {len(stats['empty_papers'])}")
    print(f"  Total annotations:     {stats['total_annotations']}")
    print(f"  Output directory:      {output_base_path}")
    print("\n[COMPLETE] Processing complete!")


if __name__ == '__main__':
    main()