import argparse
from pathlib import Path

import numpy as np


WATER_NAMES = {'HOH', 'WAT', 'DOD'}


def parse_pdb_coords(pdb_path: Path, record_types=('ATOM',), allowed_resnames=None, excluded_resnames=None):
    coords = []
    allowed = {x.upper() for x in allowed_resnames} if allowed_resnames else None
    excluded = {x.upper() for x in excluded_resnames} if excluded_resnames else set()

    with pdb_path.open('r') as f:
        for line in f:
            record = line[:6].strip().upper()
            if record not in record_types:
                continue
            resname = line[17:20].strip().upper()
            if allowed is not None and resname not in allowed:
                continue
            if resname in excluded:
                continue
            try:
                x = float(line[30:38]) / 10.0
                y = float(line[38:46]) / 10.0
                z = float(line[46:54]) / 10.0
            except ValueError:
                continue
            coords.append((x, y, z))

    arr = np.asarray(coords, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return arr


def get_new_pocket_coordinates(protein_coord, ligand_coord, cutoff_nm):
    pocket = []
    seen = set()
    for p in protein_coord:
        deltas = ligand_coord - p
        dists = np.linalg.norm(deltas, axis=1)
        if np.any(dists <= cutoff_nm):
            key = tuple(float(x) for x in p)
            if key not in seen:
                seen.add(key)
                pocket.append(key)
    arr = np.asarray(pocket, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return arr


def coord_to_bin(coord, x_bins, y_bins, z_bins):
    c_x, c_y, c_z = coord
    a = np.searchsorted(x_bins, c_x, side='right') - 1
    b = np.searchsorted(y_bins, c_y, side='right') - 1
    c = np.searchsorted(z_bins, c_z, side='right') - 1
    a = int(np.clip(a, 0, len(x_bins) - 2))
    b = int(np.clip(b, 0, len(y_bins) - 2))
    c = int(np.clip(c, 0, len(z_bins) - 2))
    return a, b, c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create VoxPocket pocket targets directly from a holo PDB.')
    parser.add_argument('--sample-dir', type=str, required=True, help='Sample directory containing for_labview_protein_/axis_bins.txt')
    parser.add_argument('--holo-pdb', type=str, required=True, help='Holo structure containing ligand HETATM records')
    parser.add_argument('--protein-pdb', type=str, default='', help='Protein structure to use for protein atom coordinates (default: <sample_id>_protein_cleaned.pdb in sample dir)')
    parser.add_argument('--ligand-resnames', type=str, default='', help='Optional comma-separated ligand residue names to keep from the holo PDB')
    parser.add_argument('--cutoff-nm', type=float, default=0.45, help='Protein-ligand distance cutoff in nm for target pocket construction')
    parser.add_argument('--output-pocket-dir', type=str, default='', help='Override output directory for for_labview_pocket_')
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    sample_id = sample_dir.name
    protein_pdb = Path(args.protein_pdb) if args.protein_pdb else sample_dir / f'{sample_id}_protein_cleaned.pdb'
    holo_pdb = Path(args.holo_pdb)
    output_pocket_dir = Path(args.output_pocket_dir) if args.output_pocket_dir else sample_dir / 'for_labview_pocket_'
    output_pocket_dir.mkdir(parents=True, exist_ok=True)

    axis_bins_path = sample_dir / 'for_labview_protein_/axis_bins.txt'
    if not axis_bins_path.is_file():
        raise FileNotFoundError(f'Missing axis bins: {axis_bins_path}')
    if not protein_pdb.is_file():
        raise FileNotFoundError(f'Missing protein PDB: {protein_pdb}')
    if not holo_pdb.is_file():
        raise FileNotFoundError(f'Missing holo PDB: {holo_pdb}')

    axis_bins = np.loadtxt(axis_bins_path)
    x_bins, y_bins, z_bins = axis_bins[0], axis_bins[1], axis_bins[2]
    nbins = len(x_bins) - 1

    protein_coord = parse_pdb_coords(protein_pdb, record_types=('ATOM',))
    allowed_resnames = [x.strip() for x in args.ligand_resnames.split(',') if x.strip()] or None
    ligand_coord = parse_pdb_coords(
        holo_pdb,
        record_types=('HETATM',),
        allowed_resnames=allowed_resnames,
        excluded_resnames=WATER_NAMES if allowed_resnames is None else None,
    )

    if protein_coord.shape[0] == 0:
        raise ValueError(f'No protein ATOM coordinates found in {protein_pdb}')
    if ligand_coord.shape[0] == 0:
        ligand_hint = f' with ligand_resnames={allowed_resnames}' if allowed_resnames else ''
        raise ValueError(f'No ligand HETATM coordinates found in {holo_pdb}{ligand_hint}')

    pocket_coordinates = get_new_pocket_coordinates(protein_coord, ligand_coord, cutoff_nm=args.cutoff_nm)
    if pocket_coordinates.shape[0] == 0:
        raise ValueError('No protein atoms fell within the ligand cutoff; pocket target would be empty.')

    pocket_tensor = np.zeros((nbins, nbins, nbins), dtype=np.float32)
    for coord in pocket_coordinates:
        i, j, k = coord_to_bin(coord, x_bins, y_bins, z_bins)
        pocket_tensor[i, j, k] += 1.0

    np.savetxt(output_pocket_dir / 'xyz_new_pocket.txt', pocket_coordinates, fmt='%1.8f')
    np.save(output_pocket_dir / 'N_tensor_new_pocket.npy', pocket_tensor)

    print(f'Sample dir: {sample_dir}')
    print(f'Protein atoms used: {protein_coord.shape[0]}')
    print(f'Ligand atoms used: {ligand_coord.shape[0]}')
    print(f'Pocket coordinates: {pocket_coordinates.shape[0]}')
    print(f'Wrote: {output_pocket_dir / "xyz_new_pocket.txt"}')
    print(f'Wrote: {output_pocket_dir / "N_tensor_new_pocket.npy"}')
