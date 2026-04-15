import argparse
import csv
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_filepaths(path):
    paths = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(line)
    return paths


def normalize_name(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def connected_components(binary):
    """Count 26-connected components in a small 3D boolean array without scipy."""
    arr = np.asarray(binary, dtype=bool)
    visited = np.zeros(arr.shape, dtype=bool)
    dims = arr.shape
    count = 0
    sizes = []
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    starts = np.argwhere(arr)
    for sx, sy, sz in starts:
        if visited[sx, sy, sz]:
            continue
        count += 1
        size = 0
        q = deque([(int(sx), int(sy), int(sz))])
        visited[sx, sy, sz] = True
        while q:
            x, y, z = q.popleft()
            size += 1
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (
                    0 <= nx < dims[0]
                    and 0 <= ny < dims[1]
                    and 0 <= nz < dims[2]
                    and arr[nx, ny, nz]
                    and not visited[nx, ny, nz]
                ):
                    visited[nx, ny, nz] = True
                    q.append((nx, ny, nz))
        sizes.append(size)
    return count, sizes


def target_pocket_count(sample_dir, min_voxels):
    target = sample_dir / "for_labview_pocket_" / "N_tensor_new_pocket.npy"
    if not target.is_file():
        return None, []
    arr = np.load(target)
    count, sizes = connected_components(arr > 0)
    kept = [s for s in sizes if s >= min_voxels]
    return len(kept), kept


def load_voxpocket_counts(candidate_csv):
    counts = Counter()
    if not candidate_csv:
        return counts
    path = Path(candidate_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Missing VoxPocket candidate CSV: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_path = row.get("sample_path", "").strip()
            sample_id = row.get("sample_id", "").strip()
            key = sample_path or sample_id
            if key:
                counts[normalize_name(Path(key).name)] += 1
    return counts


def prediction_csv_row_count(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0
        return sum(1 for _ in reader)


def load_p2rank_counts(predictions_dir):
    counts = defaultdict(int)
    if not predictions_dir:
        return counts
    root = Path(predictions_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing P2Rank predictions dir: {root}")
    for path in root.rglob("*predictions*.csv"):
        name = path.name
        sample = name.replace("_protein_lite.pdb_predictions.csv", "")
        sample = sample.replace("_protein_cleaned.pdb_predictions.csv", "")
        sample = sample.replace("_protein.pdb_predictions.csv", "")
        sample = sample.replace(".pdb_predictions.csv", "")
        counts[normalize_name(sample)] += prediction_csv_row_count(path)
    return counts


def fmt_float(value):
    return "" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare actual pocket-component counts with VoxPocket and P2Rank predicted-pocket counts."
    )
    parser.add_argument("--filepaths", required=True, help="VoxPocket sample directory file list")
    parser.add_argument("--voxpocket-candidates-csv", default="", help="Candidate table from build_candidate_rank_table.py")
    parser.add_argument("--p2rank-predictions-dir", default="", help="Directory containing P2Rank *_predictions.csv files")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-min-voxels", type=int, default=1, help="Minimum connected target voxels to count as an actual pocket")
    args = parser.parse_args()

    filepaths = load_filepaths(args.filepaths)
    vox_counts = load_voxpocket_counts(args.voxpocket_candidates_csv)
    p2_counts = load_p2rank_counts(args.p2rank_predictions_dir)

    rows = []
    actual_counts = []
    vox_pred_counts = []
    p2_pred_counts = []

    for rel in filepaths:
        sample_dir = PROJECT_ROOT / rel
        sample_id = sample_dir.name
        key = normalize_name(sample_id)
        actual_count, actual_sizes = target_pocket_count(sample_dir, args.target_min_voxels)
        vox_count = int(vox_counts.get(key, 0))
        p2_count = int(p2_counts.get(key, 0))

        if actual_count is not None:
            actual_counts.append(actual_count)
        vox_pred_counts.append(vox_count)
        p2_pred_counts.append(p2_count)

        rows.append({
            "sample_id": sample_id,
            "sample_path": rel,
            "actual_pocket_components": "" if actual_count is None else actual_count,
            "actual_component_sizes_voxels": ";".join(str(x) for x in actual_sizes),
            "voxpocket_predicted_pockets": vox_count,
            "p2rank_predicted_pockets": p2_count,
            "voxpocket_minus_actual": "" if actual_count is None else vox_count - actual_count,
            "p2rank_minus_actual": "" if actual_count is None else p2_count - actual_count,
        })

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Samples evaluated: {len(rows)}")
    print(f"Output CSV: {out}")
    print(f"Target min voxels: {args.target_min_voxels}")
    if actual_counts:
        print(f"Actual pockets mean: {np.mean(actual_counts):.6f}")
        print(f"Actual pockets median: {np.median(actual_counts):.6f}")
        print(f"Actual pockets total: {int(np.sum(actual_counts))}")
    print(f"VoxPocket predicted pockets mean: {np.mean(vox_pred_counts):.6f}")
    print(f"VoxPocket predicted pockets median: {np.median(vox_pred_counts):.6f}")
    print(f"VoxPocket predicted pockets total: {int(np.sum(vox_pred_counts))}")
    print(f"P2Rank predicted pockets mean: {np.mean(p2_pred_counts):.6f}")
    print(f"P2Rank predicted pockets median: {np.median(p2_pred_counts):.6f}")
    print(f"P2Rank predicted pockets total: {int(np.sum(p2_pred_counts))}")


if __name__ == "__main__":
    main()
