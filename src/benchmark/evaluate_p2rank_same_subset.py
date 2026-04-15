import argparse
import csv
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_sample_paths(filepaths):
    paths = []
    for raw in Path(filepaths).read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#"):
            paths.append(raw)
    return paths


def target_coords_nm(sample_dir):
    target = sample_dir / "for_labview_pocket_" / "xyz_new_pocket.txt"
    arr = np.loadtxt(target)
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return arr.reshape(-1, 3)


def normalize_name(s):
    return "".join(ch.lower() for ch in s if ch.isalnum())


def find_prediction_csv(predictions_dir, sample_id):
    pred_root = Path(predictions_dir)
    candidates = list(pred_root.rglob("*predictions*.csv")) + list(pred_root.rglob("*.csv"))
    sid = normalize_name(sample_id)
    scored = []
    for path in candidates:
        name = normalize_name(path.name)
        parent = normalize_name(path.parent.name)
        score = 0
        if sid in name:
            score += 10
        if sid in parent:
            score += 5
        if "prediction" in name or "predictions" in name:
            score += 1
        if score > 0:
            scored.append((score, len(path.name), path))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def parse_float(row, names):
    for name in names:
        if name in row and row[name] != "":
            try:
                return float(row[name])
            except ValueError:
                pass
    return None


def read_p2rank_centers_nm(csv_path):
    centers = []
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return centers
        fields = reader.fieldnames
        # Common P2Rank headers include center_x, center_y, center_z, but be tolerant.
        for row in reader:
            x = parse_float(row, ["center_x", "center x", "centerX", "x", "X"])
            y = parse_float(row, ["center_y", "center y", "centerY", "y", "Y"])
            z = parse_float(row, ["center_z", "center z", "centerZ", "z", "Z"])
            if x is None or y is None or z is None:
                # Fallback: use the first three numeric columns whose names look coordinate-like.
                numeric = []
                for key in fields:
                    try:
                        numeric.append((key, float(row[key])))
                    except Exception:
                        continue
                coord_like = [(k, v) for k, v in numeric if any(tok in k.lower() for tok in ["center", "coord", "x", "y", "z"])]
                if len(coord_like) >= 3:
                    x, y, z = coord_like[0][1], coord_like[1][1], coord_like[2][1]
                else:
                    continue
            # P2Rank/PDB coordinates are Angstrom. VoxPocket target coords are nm.
            centers.append(np.array([x, y, z], dtype=float) / 10.0)
    return centers


def min_dist_to_target(center_nm, target_nm):
    if len(target_nm) == 0:
        return math.inf
    return float(np.linalg.norm(target_nm - center_nm, axis=1).min())


def safe_mean(values):
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def safe_median(values):
    vals = [v for v in values if math.isfinite(v)]
    return float(np.median(vals)) if vals else None


def fmt_count(n, total):
    return f"{n}/{total} ({n / total:.6f})" if total else "0/0 (nan)"


def main():
    parser = argparse.ArgumentParser(description="Evaluate P2Rank predictions on the same VoxPocket sample subset.")
    parser.add_argument("--filepaths", required=True, help="VoxPocket sample directory file list")
    parser.add_argument("--predictions-dir", required=True, help="P2Rank output directory containing prediction CSVs")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--success-thresholds-nm", default="0.5,1.0,1.5")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.success_thresholds_nm.split(",") if x.strip()]
    sample_paths = read_sample_paths(args.filepaths)

    rows = []
    missing_predictions = []
    top1_distances = []
    topk_best_distances = []
    candidates_per_sample = []
    top1_present = 0
    topk_present = 0
    top1_success = {t: 0 for t in thresholds}
    topk_success = {t: 0 for t in thresholds}

    for rel in sample_paths:
        sample_dir = PROJECT_ROOT / rel
        sample_id = sample_dir.name
        pred_csv = find_prediction_csv(args.predictions_dir, sample_id)
        target = target_coords_nm(sample_dir)
        if pred_csv is None:
            missing_predictions.append(sample_id)
            centers = []
        else:
            centers = read_p2rank_centers_nm(pred_csv)
        candidates_per_sample.append(len(centers))

        dists = [min_dist_to_target(center, target) for center in centers]
        top1_dist = dists[0] if dists else math.inf
        topk_dist = min(dists[: args.top_k]) if dists[: args.top_k] else math.inf
        top1_distances.append(top1_dist)
        topk_best_distances.append(topk_dist)
        if math.isfinite(top1_dist):
            top1_present += 1
        if math.isfinite(topk_dist):
            topk_present += 1
        for t in thresholds:
            if top1_dist <= t:
                top1_success[t] += 1
            if topk_dist <= t:
                topk_success[t] += 1
        rows.append({
            "sample_id": sample_id,
            "sample_path": rel,
            "prediction_csv": "" if pred_csv is None else str(pred_csv),
            "num_candidates": len(centers),
            "top1_distance_nm": "" if not math.isfinite(top1_dist) else f"{top1_dist:.8f}",
            f"top{args.top_k}_best_distance_nm": "" if not math.isfinite(topk_dist) else f"{topk_dist:.8f}",
        })

    n = len(sample_paths)
    print(f"Samples evaluated: {n}")
    print(f"Predictions dir: {args.predictions_dir}")
    print(f"Missing prediction CSVs: {len(missing_predictions)}")
    print(f"Top-1 candidate present: {fmt_count(top1_present, n)}")
    print(f"Top-{args.top_k} candidate present: {fmt_count(topk_present, n)}")
    print(f"Mean candidates per sample: {float(np.mean(candidates_per_sample)) if candidates_per_sample else 0.0:.6f}")
    m = safe_mean(top1_distances)
    med = safe_median(top1_distances)
    print(f"Top-1 mean distance to reference pocket: {'N/A' if m is None else f'{m:.6f}'}")
    print(f"Top-1 median distance to reference pocket: {'N/A' if med is None else f'{med:.6f}'}")
    m = safe_mean(topk_best_distances)
    med = safe_median(topk_best_distances)
    print(f"Top-{args.top_k} best mean distance to reference pocket: {'N/A' if m is None else f'{m:.6f}'}")
    print(f"Top-{args.top_k} best median distance to reference pocket: {'N/A' if med is None else f'{med:.6f}'}")
    for t in thresholds:
        print(f"Top-1 success @ {t:.2f} nm: {fmt_count(top1_success[t], n)}")
        print(f"Top-{args.top_k} success @ {t:.2f} nm: {fmt_count(topk_success[t], n)}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote per-sample P2Rank comparison CSV: {out}")


if __name__ == "__main__":
    main()
