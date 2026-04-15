import argparse
from pathlib import Path

import mdtraj
import numpy as np


def atom_key(line):
    try:
        return (
            round(float(line[30:38]), 3),
            round(float(line[38:46]), 3),
            round(float(line[46:54]), 3),
        )
    except ValueError:
        return None


def write_lite_protein(input_pdb: Path, output_pdb: Path):
    seen_coords = set()
    kept = []
    dropped_duplicate_coords = 0
    dropped_bad_coords = 0
    with input_pdb.open('r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            key = atom_key(line)
            if key is None:
                dropped_bad_coords += 1
                continue
            if key in seen_coords:
                dropped_duplicate_coords += 1
                continue
            seen_coords.add(key)
            kept.append(line if line.endswith('\n') else line + '\n')
    if not kept:
        raise ValueError(f'No usable ATOM lines found in {input_pdb}')
    output_pdb.write_text(''.join(kept) + 'END\n')
    return len(kept), dropped_duplicate_coords, dropped_bad_coords


def parse_bfactors(clean_pdb: Path):
    vals = []
    with clean_pdb.open('r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


def bin_triplet(coord, x_bins, y_bins, z_bins):
    c_x, c_y, c_z = coord
    a = int(np.clip(np.searchsorted(x_bins, c_x, side='right') - 1, 0, len(x_bins) - 2))
    b = int(np.clip(np.searchsorted(y_bins, c_y, side='right') - 1, 0, len(y_bins) - 2))
    c = int(np.clip(np.searchsorted(z_bins, c_z, side='right') - 1, 0, len(z_bins) - 2))
    return a, b, c


def create_lite_tensors(sample_dir: Path, clean_pdb: Path, nbins: int):
    conf = mdtraj.load(str(clean_pdb))
    if conf.n_atoms > 50000:
        raise RuntimeError(f'{clean_pdb} too large ({conf.n_atoms} atoms) for this pipeline.')

    xyz = conf.xyz[0]
    bfactors = parse_bfactors(clean_pdb)
    if bfactors.shape[0] != xyz.shape[0]:
        print(
            f'WARNING {clean_pdb}: B-factor count mismatch '
            f'{bfactors.shape[0]} vs {xyz.shape[0]} atoms; using zero B-factors.'
        )
        bfactors = np.zeros(xyz.shape[0], dtype=np.float32)

    sasa = mdtraj.shrake_rupley(conf)[0]
    atom_indices = conf.topology.select('all')
    buriedness = []
    for ii in atom_indices:
        neigh = mdtraj.compute_neighbors(conf, 1.0, [ii])
        buriedness.append(len(neigh[0]))
    buriedness = np.asarray(buriedness, dtype=np.float32)

    small = 0.001
    min_x, max_x = np.min(xyz[:, 0]) - small, np.max(xyz[:, 0]) + small
    min_y, max_y = np.min(xyz[:, 1]) - small, np.max(xyz[:, 1]) + small
    min_z, max_z = np.min(xyz[:, 2]) - small, np.max(xyz[:, 2]) + small
    max_dist = np.max([max_x - min_x, max_y - min_y, max_z - min_z])

    x_bins = np.linspace(min_x, min_x + max_dist, nbins + 1)
    y_bins = np.linspace(min_y, min_y + max_dist, nbins + 1)
    z_bins = np.linspace(min_z, min_z + max_dist, nbins + 1)
    axis_bins = np.array([x_bins, y_bins, z_bins])

    out_dir = sample_dir / 'for_labview_protein_'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / 'axis_bins.txt', axis_bins, fmt='%1.8f')

    N_tensor = np.zeros((nbins, nbins, nbins), dtype=np.float32)
    bfactors_tensor = np.zeros_like(N_tensor)
    buriedness_tensor = np.zeros_like(N_tensor)
    sasa_tensor = np.zeros_like(N_tensor)

    bin_indices = []
    for coord in xyz:
        i, j, k = bin_triplet(coord, x_bins, y_bins, z_bins)
        bin_indices.append((i, j, k))

    for row, (i, j, k) in enumerate(bin_indices):
        N_tensor[i, j, k] += 1.0
        bfactors_tensor[i, j, k] += bfactors[row]
        buriedness_tensor[i, j, k] += buriedness[row]
        sasa_tensor[i, j, k] += sasa[row]

    with np.errstate(divide='ignore', invalid='ignore'):
        bfactors_tensor = np.nan_to_num(bfactors_tensor / N_tensor)
        buriedness_tensor = np.nan_to_num(buriedness_tensor / N_tensor)
        sasa_tensor = np.nan_to_num(sasa_tensor / N_tensor)

    np.savetxt(out_dir / 'xyz.txt', xyz, fmt='%1.8f')
    np.savetxt(out_dir / 'bfactors.txt', bfactors, fmt='%1.8f')
    np.savetxt(out_dir / 'buriedness.txt', buriedness, fmt='%i')
    np.savetxt(out_dir / 'sasa.txt', sasa, fmt='%1.8f')
    np.save(out_dir / 'N_tensor.npy', N_tensor)
    np.save(out_dir / 'bfactors_tensor.npy', bfactors_tensor)
    np.save(out_dir / 'buriedness_tensor.npy', buriedness_tensor)
    np.save(out_dir / 'sasa_tensor.npy', sasa_tensor)


def main():
    parser = argparse.ArgumentParser(description='Process benchmark sample folders without PyMOL/PDB2PQR dependencies.')
    parser.add_argument('--sample-root', required=True)
    parser.add_argument('--nbins', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start-at', default='')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    sample_root = Path(args.sample_root)
    sample_dirs = sorted([p for p in sample_root.iterdir() if p.is_dir()])
    if args.start_at:
        sample_dirs = [p for p in sample_dirs if p.name >= args.start_at]
    if args.limit > 0:
        sample_dirs = sample_dirs[:args.limit]

    processed = 0
    failed = 0
    skipped = 0
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        protein_pdb = sample_dir / f'{sample_id}_protein.pdb'
        holo_pdb = sample_dir / f'{sample_id}_holo.pdb'
        axis_bins = sample_dir / 'for_labview_protein_' / 'axis_bins.txt'
        if not protein_pdb.is_file() or not holo_pdb.is_file():
            print(f'SKIP {sample_id}: missing protein or holo pdb')
            skipped += 1
            continue
        if axis_bins.is_file() and not args.overwrite:
            print(f'SKIP {sample_id}: lite tensors already exist')
            skipped += 1
            continue
        try:
            lite_pdb = sample_dir / f'{sample_id}_protein_lite.pdb'
            kept, dropped_dupes, dropped_bad = write_lite_protein(protein_pdb, lite_pdb)
            create_lite_tensors(sample_dir, lite_pdb, nbins=args.nbins)
            processed += 1
            print(f'OK {sample_id}: atoms={kept} dropped_duplicate_coords={dropped_dupes} dropped_bad_coords={dropped_bad}')
        except Exception as exc:
            failed += 1
            print(f'FAIL {sample_id}: {type(exc).__name__}: {exc}')

    print(f'Sample root: {sample_root}')
    print(f'Processed: {processed}')
    print(f'Skipped: {skipped}')
    print(f'Failed: {failed}')


if __name__ == '__main__':
    main()
