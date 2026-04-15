import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FILES = [
    'for_labview_protein_/N_tensor.npy',
    'for_labview_protein_/bfactors_tensor.npy',
    'for_labview_protein_/buriedness_tensor.npy',
    'for_labview_protein_/charge_tensor.npy',
    'for_labview_protein_/radius_tensor.npy',
    'for_labview_protein_/hbdon_tensor.npy',
    'for_labview_protein_/hbac_tensor.npy',
    'for_labview_protein_/sasa_tensor.npy',
    'for_labview_pocket_/N_tensor_new_pocket.npy',
]

FEATURE_FILE_MAP = {
    'N': 'for_labview_protein_/N_tensor.npy',
    'bfactors': 'for_labview_protein_/bfactors_tensor.npy',
    'buriedness': 'for_labview_protein_/buriedness_tensor.npy',
    'charge': 'for_labview_protein_/charge_tensor.npy',
    'radius': 'for_labview_protein_/radius_tensor.npy',
    'hbdon': 'for_labview_protein_/hbdon_tensor.npy',
    'hbacc': 'for_labview_protein_/hbac_tensor.npy',
    'sasa': 'for_labview_protein_/sasa_tensor.npy',
}


def load_dataset_entries(dataset_file: Path):
    entries = []
    with dataset_file.open('r') as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith('#'):
                continue
            entries.append(raw)
    return entries


def derive_sample_id(entry: str, id_mode: str):
    stem = Path(entry).stem
    if id_mode == 'stem':
        return stem.lower()
    if len(stem) < 4:
        raise ValueError(f"Entry stem '{stem}' is too short to derive a 4-character PDB id.")
    return stem[:4].lower()


def sample_relpath(sample_id: str, processed_prefix: str):
    prefix = processed_prefix.strip('/ ')
    return f"{prefix}/{sample_id}/"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build VoxPocket evaluation file lists from P2Rank benchmark .ds files.')
    parser.add_argument('--dataset-file', type=str, required=True, help='Path to a P2Rank .ds benchmark list, e.g. coach420.ds')
    parser.add_argument('--processed-prefix', type=str, default='refined-set', help='Relative root under the project where processed VoxPocket sample folders live')
    parser.add_argument('--id-mode', choices=['pdb4', 'stem'], default='pdb4', help='How to map dataset entries to processed sample directories')
    parser.add_argument('--check-required-files', action='store_true', help='Require the full VoxPocket tensor set, not just the sample directory')
    parser.add_argument(
        '--features',
        type=str,
        default='',
        help='Optional comma-separated feature set to require when --check-required-files is enabled. Defaults to full VoxPocket tensor set.',
    )
    parser.add_argument('--output-file', type=str, required=True, help='Where to write the resulting VoxPocket file list')
    parser.add_argument('--missing-file', type=str, default='', help='Optional path for a detailed missing/coverage report')
    args = parser.parse_args()

    dataset_file = Path(args.dataset_file)
    entries = load_dataset_entries(dataset_file)
    if not entries:
        raise ValueError(f'No dataset entries found in {dataset_file}')

    usable = []
    missing_rows = []
    seen = set()

    for entry in entries:
        sample_id = derive_sample_id(entry, args.id_mode)
        relpath = sample_relpath(sample_id, args.processed_prefix)
        sample_dir = PROJECT_ROOT / relpath

        if relpath in seen:
            continue
        seen.add(relpath)

        if not sample_dir.is_dir():
            missing_rows.append((entry, sample_id, relpath, 'missing_directory'))
            continue

        if args.check_required_files:
            if args.features:
                requested = [x.strip() for x in args.features.split(',') if x.strip()]
                invalid = [name for name in requested if name not in FEATURE_FILE_MAP]
                if invalid:
                    raise ValueError(f'Invalid feature names: {invalid}. Valid: {sorted(FEATURE_FILE_MAP)}')
                required_files = [FEATURE_FILE_MAP[name] for name in requested] + [
                    'for_labview_pocket_/N_tensor_new_pocket.npy',
                ]
            else:
                required_files = REQUIRED_FILES
            missing_files = [rel for rel in required_files if not (sample_dir / rel).is_file()]
            if missing_files:
                missing_rows.append((entry, sample_id, relpath, 'missing_required_files:' + ';'.join(missing_files)))
                continue

        usable.append(relpath)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(f'{row}\n' for row in usable))

    report_lines = [
        f'dataset_file: {dataset_file}',
        f'entries_total: {len(entries)}',
        f'usable_samples: {len(usable)}',
        f'missing_samples: {len(missing_rows)}',
        f'processed_prefix: {args.processed_prefix}',
        f'id_mode: {args.id_mode}',
        f'check_required_files: {int(args.check_required_files)}',
        f'features: {args.features}',
        '',
    ]
    if missing_rows:
        report_lines.append('missing_details:')
        for entry, sample_id, relpath, reason in missing_rows:
            report_lines.append(f'- entry={entry} | sample_id={sample_id} | relpath={relpath} | reason={reason}')

    report_text = '\n'.join(report_lines) + '\n'
    if args.missing_file:
        missing_path = Path(args.missing_file)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text(report_text)

    print(f'Dataset list: {dataset_file}')
    print(f'Total entries: {len(entries)}')
    print(f'Usable samples: {len(usable)}')
    print(f'Missing samples: {len(missing_rows)}')
    print(f'Wrote VoxPocket file list to {output_path}')
    if args.missing_file:
        print(f'Wrote coverage report to {args.missing_file}')
