import argparse
import os
from pathlib import Path

import mdtraj
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SILENCE = ' > /dev/null 2>&1'


def open_pqr(path_to_file):
    with open(path_to_file) as f:
        lines = f.readlines()

    x_pqr, y_pqr, z_pqr = [], [], []
    charge, radius = [], []
    for line in lines:
        if line.startswith("ATOM"):
            x_pqr.append(float(line[30:38]))
            y_pqr.append(float(line[38:46]))
            z_pqr.append(float(line[46:54]))
            charge.append(float(line.split()[-2]))
            radius.append(float(line.split()[-1]))

    xyz_pqr = np.column_stack((x_pqr, y_pqr, z_pqr))
    return xyz_pqr, charge, radius



def run_shell(command: str):
    rc = os.system(command)
    if rc != 0:
        raise RuntimeError(f'Command failed with exit code {rc}: {command}')


def run_cleaning(protein_pdb: Path):
    command = f"pymol -c {SCRIPT_DIR / 'pymol_cleaning.py'} {protein_pdb}{SILENCE}"
    run_shell(command)
    return protein_pdb.with_name(protein_pdb.stem + '_cleaned.pdb')


def run_feature_extractors(cleaned_pdb: Path):
    conf = mdtraj.load(str(cleaned_pdb))
    if conf.n_atoms > 50000:
        raise RuntimeError(f'{cleaned_pdb} too large ({conf.n_atoms} atoms) for this pipeline.')

    sasa = mdtraj.shrake_rupley(conf)
    np.savetxt(str(cleaned_pdb).rstrip('d.pdb') + 'd_sasa.txt', sasa[0])

    dssp = mdtraj.compute_dssp(conf, simplified=True)
    np.savetxt(str(cleaned_pdb).rstrip('d.pdb') + 'd_dssp.txt', dssp[0], fmt='%s')

    atom_indices = conf.topology.select('all')
    buriedness = []
    for ii in atom_indices:
        neigh = mdtraj.compute_neighbors(conf, 1, [ii])
        buriedness.append(len(neigh[0]))
    np.savetxt(str(cleaned_pdb).rstrip('d.pdb') + 'd_buriedness.txt', buriedness)

    run_shell(f"pdb2pqr --ff=amber {cleaned_pdb} {str(cleaned_pdb).rstrip('pdb') + 'pqr'}{SILENCE}")
    run_shell(f"pymol -c {SCRIPT_DIR / 'pymol_bfactors.py'} {cleaned_pdb}{SILENCE}")
    run_shell(f"pymol -c {SCRIPT_DIR / 'pymol_hdonors.py'} {cleaned_pdb}{SILENCE}")
    run_shell(f"pymol -c {SCRIPT_DIR / 'pymol_hacceptors.py'} {cleaned_pdb}{SILENCE}")

    return conf


def bin_triplet(coord, x_bins, y_bins, z_bins):
    c_x, c_y, c_z = coord
    a = int(np.clip(np.searchsorted(x_bins, c_x, side='right') - 1, 0, len(x_bins) - 2))
    b = int(np.clip(np.searchsorted(y_bins, c_y, side='right') - 1, 0, len(y_bins) - 2))
    c = int(np.clip(np.searchsorted(z_bins, c_z, side='right') - 1, 0, len(z_bins) - 2))
    return a, b, c


def create_protein_tensors(sample_dir: Path, cleaned_pdb: Path, conf, nbins: int):
    xyz = conf.xyz[0]
    bfactors = np.loadtxt(str(cleaned_pdb).rstrip('d.pdb') + 'd_bfactors.txt')
    buriedness = np.loadtxt(str(cleaned_pdb).rstrip('d.pdb') + 'd_buriedness.txt')
    hbac = np.loadtxt(str(cleaned_pdb).rstrip('d.pdb') + 'd_hbacceptors.txt')
    hbdon = np.loadtxt(str(cleaned_pdb).rstrip('d.pdb') + 'd_hbdonors.txt')
    sasa = np.loadtxt(str(cleaned_pdb).rstrip('d.pdb') + 'd_sasa.txt')
    xyz_pqr, charge, radius = open_pqr(str(cleaned_pdb).rstrip('pdb') + 'pqr')

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
    hbac_tensor = np.zeros_like(N_tensor)
    hbdon_tensor = np.zeros_like(N_tensor)
    sasa_tensor = np.zeros_like(N_tensor)
    charge_tensor = np.zeros_like(N_tensor)
    radius_tensor = np.zeros_like(N_tensor)

    bin_indices = []
    for coord in xyz:
        a, b, c = bin_triplet(coord, x_bins, y_bins, z_bins)
        bin_indices.append((a, b, c))

    for row, (i, j, k) in enumerate(bin_indices):
        N_tensor[i, j, k] += 1
        bfactors_tensor[i, j, k] += bfactors[row]
        buriedness_tensor[i, j, k] += buriedness[row]
        sasa_tensor[i, j, k] += sasa[row]

    bfactors_tensor = np.nan_to_num(bfactors_tensor / N_tensor)
    buriedness_tensor = np.nan_to_num(buriedness_tensor / N_tensor)
    sasa_tensor = np.nan_to_num(sasa_tensor / N_tensor)

    hbac = np.atleast_1d(hbac).astype(int)
    hbdon = np.atleast_1d(hbdon).astype(int)
    for g in hbac:
        idx = max(int(g) - 1, 0)
        if idx < len(bin_indices):
            i, j, k = bin_indices[idx]
            hbac_tensor[i, j, k] += 1
    for g in hbdon:
        idx = max(int(g) - 1, 0)
        if idx < len(bin_indices):
            i, j, k = bin_indices[idx]
            hbdon_tensor[i, j, k] += 1

    xyz_pqr = xyz_pqr / 10.0
    N_tensor_pqr = np.zeros_like(N_tensor)
    bin_indices_pqr = [bin_triplet(coord, x_bins, y_bins, z_bins) for coord in xyz_pqr]
    for row, (i, j, k) in enumerate(bin_indices_pqr):
        charge_tensor[i, j, k] += charge[row]
        radius_tensor[i, j, k] += radius[row]
        N_tensor_pqr[i, j, k] += 1
    radius_tensor = np.nan_to_num(radius_tensor / N_tensor_pqr)

    np.savetxt(out_dir / 'xyz.txt', xyz, fmt='%1.8f')
    np.savetxt(out_dir / 'bfactors.txt', bfactors, fmt='%1.8f')
    np.savetxt(out_dir / 'buriedness.txt', buriedness, fmt='%i')
    np.savetxt(out_dir / 'sasa.txt', sasa, fmt='%1.8f')
    np.savetxt(out_dir / 'xyz_pqr.txt', xyz_pqr, fmt='%1.8f')
    np.savetxt(out_dir / 'radius.txt', radius, fmt='%1.8f')
    np.savetxt(out_dir / 'charge.txt', charge, fmt='%1.8f')

    hbac_list = np.array([1 if i in set(hbac.tolist()) else 0 for i in range(len(buriedness))])
    hbdon_list = np.array([1 if i in set(hbdon.tolist()) else 0 for i in range(len(buriedness))])
    np.savetxt(out_dir / 'hbac.txt', hbac_list, fmt='%i')
    np.savetxt(out_dir / 'hbdon.txt', hbdon_list, fmt='%i')

    np.save(out_dir / 'N_tensor.npy', N_tensor)
    np.save(out_dir / 'bfactors_tensor.npy', bfactors_tensor)
    np.save(out_dir / 'buriedness_tensor.npy', buriedness_tensor)
    np.save(out_dir / 'sasa_tensor.npy', sasa_tensor)
    np.save(out_dir / 'hbac_tensor.npy', hbac_tensor)
    np.save(out_dir / 'hbdon_tensor.npy', hbdon_tensor)
    np.save(out_dir / 'N_tensor_pqr.npy', N_tensor_pqr)
    np.save(out_dir / 'charge_tensor.npy', charge_tensor)
    np.save(out_dir / 'radius_tensor.npy', radius_tensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process benchmark sample folders into VoxPocket protein tensors.')
    parser.add_argument('--sample-root', type=str, required=True, help='Root directory containing benchmark sample folders')
    parser.add_argument('--nbins', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start-at', type=str, default='')
    args = parser.parse_args()

    sample_root = Path(args.sample_root)
    sample_dirs = sorted([p for p in sample_root.iterdir() if p.is_dir()])
    if args.start_at:
        sample_dirs = [p for p in sample_dirs if p.name >= args.start_at]
    if args.limit > 0:
        sample_dirs = sample_dirs[: args.limit]

    processed = 0
    failed = 0
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        protein_pdb = sample_dir / f'{sample_id}_protein.pdb'
        holo_pdb = sample_dir / f'{sample_id}_holo.pdb'
        if not protein_pdb.is_file() or not holo_pdb.is_file():
            print(f'SKIP {sample_id}: missing protein or holo pdb')
            failed += 1
            continue
        try:
            cleaned_pdb = run_cleaning(protein_pdb)
            conf = run_feature_extractors(cleaned_pdb)
            create_protein_tensors(sample_dir, cleaned_pdb, conf, nbins=args.nbins)
            processed += 1
            print(f'OK {sample_id}')
        except Exception as exc:
            failed += 1
            print(f'FAIL {sample_id}: {type(exc).__name__}: {exc}')

    print(f'Sample root: {sample_root}')
    print(f'Processed: {processed}')
    print(f'Failed/skipped: {failed}')
