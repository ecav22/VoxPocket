import argparse
from pathlib import Path

WATER_NAMES = {'HOH', 'WAT', 'DOD'}
ION_NAMES = {
    'NA', 'K', 'CL', 'CA', 'MG', 'ZN', 'MN', 'FE', 'CU', 'CO', 'NI', 'CD',
    'IOD', 'BR', 'SO4', 'PO4'
}


def load_entries(dataset_file: Path):
    entries = []
    with dataset_file.open('r') as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith('#'):
                continue
            entries.append(raw)
    return entries


def infer_sample_id(entry: str, id_mode: str):
    stem = Path(entry).stem
    if id_mode == 'stem':
        return stem.lower()
    if len(stem) < 4:
        raise ValueError(f"Entry stem '{stem}' is too short to derive a 4-character PDB id.")
    return stem[:4].lower()


def infer_chain_id(entry: str, use_chain_from_stem: bool):
    stem = Path(entry).stem
    if not use_chain_from_stem:
        return None
    return stem[4] if len(stem) == 5 else None


def filter_pdb_lines(lines, record_type, chain_id=None, ligand_resnames=None, keep_all_ligand_chains=False):
    out = []
    allowed_ligands = {x.upper() for x in ligand_resnames} if ligand_resnames else None
    for line in lines:
        record = line[:6].strip().upper()
        if record != record_type:
            continue
        line_chain = line[21].strip() or None
        resname = line[17:20].strip().upper()

        if chain_id is not None and not keep_all_ligand_chains and line_chain != chain_id:
            continue

        if record_type == 'HETATM':
            if resname in WATER_NAMES or resname in ION_NAMES:
                continue
            if allowed_ligands is not None and resname not in allowed_ligands:
                continue

        out.append(line)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bootstrap P2Rank benchmark structures into VoxPocket-style sample folders.')
    parser.add_argument('--dataset-file', type=str, required=True, help='Path to a P2Rank .ds list')
    parser.add_argument('--source-root', type=str, required=True, help='Root directory containing the actual benchmark PDB files')
    parser.add_argument('--output-root', type=str, required=True, help='Destination root for generated sample directories')
    parser.add_argument('--id-mode', choices=['stem', 'pdb4'], default='stem', help='Folder naming mode for generated samples')
    parser.add_argument('--use-chain-from-stem', action='store_true', help='If entry stem is 5 chars like 1afkA, use the 5th char as a chain filter')
    parser.add_argument('--ligand-resnames', type=str, default='', help='Optional comma-separated ligand residue names to keep')
    parser.add_argument('--keep-all-ligand-chains', action='store_true', help='Do not filter ligand HETATM records by inferred chain')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing generated files')
    args = parser.parse_args()

    dataset_file = Path(args.dataset_file)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ligand_resnames = [x.strip() for x in args.ligand_resnames.split(',') if x.strip()]

    entries = load_entries(dataset_file)
    created = 0
    skipped = 0
    missing = 0

    for entry in entries:
        src_pdb = source_root / entry
        if not src_pdb.is_file():
            print(f'MISSING_SOURCE {entry}')
            missing += 1
            continue

        sample_id = infer_sample_id(entry, args.id_mode)
        chain_id = infer_chain_id(entry, args.use_chain_from_stem)
        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        holo_out = sample_dir / f'{sample_id}_holo.pdb'
        protein_out = sample_dir / f'{sample_id}_protein.pdb'
        ligand_out = sample_dir / f'{sample_id}_ligand.pdb'

        if (not args.overwrite) and holo_out.exists() and protein_out.exists() and ligand_out.exists():
            skipped += 1
            continue

        lines = src_pdb.read_text().splitlines(True)
        protein_lines = filter_pdb_lines(lines, 'ATOM', chain_id=chain_id)
        ligand_lines = filter_pdb_lines(
            lines,
            'HETATM',
            chain_id=chain_id,
            ligand_resnames=ligand_resnames,
            keep_all_ligand_chains=args.keep_all_ligand_chains,
        )

        if not protein_lines:
            print(f'NO_PROTEIN {entry} chain={chain_id}')
            missing += 1
            continue
        if not ligand_lines:
            print(f'NO_LIGAND {entry} chain={chain_id}')
            missing += 1
            continue

        holo_out.write_text(''.join(lines))
        protein_out.write_text(''.join(protein_lines) + 'END\n')
        ligand_out.write_text(''.join(ligand_lines) + 'END\n')
        created += 1

    print(f'Dataset file: {dataset_file}')
    print(f'Source root: {source_root}')
    print(f'Output root: {output_root}')
    print(f'Entries total: {len(entries)}')
    print(f'Samples created: {created}')
    print(f'Samples skipped: {skipped}')
    print(f'Entries missing/invalid: {missing}')
