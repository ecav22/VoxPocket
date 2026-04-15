import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

import utilities  # noqa: E402


def parse_float_list(arg):
    values = [float(x.strip()) for x in arg.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def load_filepaths(path):
    with Path(path).open("r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def load_tensor(sample_dir, name, reference_shape=None):
    tensor_path = sample_dir / "for_labview_protein_" / f"{name}_tensor.npy"
    if tensor_path.is_file():
        return np.load(tensor_path)
    if reference_shape is None:
        raise FileNotFoundError(f"Missing tensor {tensor_path} and no reference shape was provided")
    return np.zeros(reference_shape, dtype=np.float32)


def component_tensor_stats(sample_dir, indices):
    stats = {}
    reference = load_tensor(sample_dir, "N")
    reference_shape = reference.shape
    feature_tensors = {
        "N": reference,
        "bfactors": load_tensor(sample_dir, "bfactors", reference_shape),
        "buriedness": load_tensor(sample_dir, "buriedness", reference_shape),
        "charge": load_tensor(sample_dir, "charge", reference_shape),
        "radius": load_tensor(sample_dir, "radius", reference_shape),
        "hbdon": load_tensor(sample_dir, "hbdon", reference_shape),
        "hbacc": load_tensor(sample_dir, "hbac", reference_shape),
        "sasa": load_tensor(sample_dir, "sasa", reference_shape),
    }
    for name, tensor in feature_tensors.items():
        vals = tensor[indices[:, 0], indices[:, 1], indices[:, 2]].astype(float)
        stats[f"{name}_mean"] = float(np.mean(vals)) if vals.size else 0.0
        stats[f"{name}_max"] = float(np.max(vals)) if vals.size else 0.0
        stats[f"{name}_sum"] = float(np.sum(vals)) if vals.size else 0.0
    return stats


def target_coords(sample_dir):
    arr = np.loadtxt(sample_dir / "for_labview_pocket_" / "xyz_new_pocket.txt")
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return arr.reshape(-1, 3)


def target_distance_stats(candidate_centroid, target_xyz):
    if target_xyz.shape[0] == 0:
        return float("inf"), float("inf")
    target_centroid = np.mean(target_xyz, axis=0)
    centroid_distance = float(np.linalg.norm(candidate_centroid - target_centroid))
    min_distance = float(np.linalg.norm(target_xyz - candidate_centroid, axis=1).min())
    return centroid_distance, min_distance


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build a tabular candidate-ranking dataset from VoxPocket predictions.")
    parser.add_argument("--filepaths", required=True)
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention_best.pt"))
    parser.add_argument("--thresholds", default="0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--merge-distance", type=float, default=0.35)
    parser.add_argument("--success-cutoff-nm", type=float, default=1.0)
    parser.add_argument("--distance-label", choices=["centroid", "min"], default="min")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_names = checkpoint.get("feature_names", utilities.FEATURE_ORDER)
    thresholds = parse_float_list(args.thresholds)

    paths = load_filepaths(args.filepaths)
    dataset = utilities.PocketDataset(paths, feature_names=feature_names, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = utilities.UNetAttention3D(
        in_channels=checkpoint.get("in_channels", len(feature_names)),
        dropout_rate=checkpoint.get("dropout_rate", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    candidate_counts = []
    oracle_success = 0

    with torch.no_grad():
        for idx, (features, target, pocket_present, centroid_target) in enumerate(loader):
            del target, pocket_present, centroid_target
            sample_path = dataset.filepaths[idx]
            sample_dir = PROJECT_ROOT / sample_path
            sample_id = sample_dir.name
            features = features.to(device)
            logits, presence_logits, centroid_logits = model(features)
            del centroid_logits
            presence_prob = float(torch.sigmoid(presence_logits).detach().cpu().item())
            pred = torch.sigmoid(logits).detach().cpu().numpy()

            candidates = utilities.extract_ranked_candidates_multithreshold(
                sample_path,
                pred,
                thresholds=thresholds,
                min_size=args.min_size,
                merge_distance=args.merge_distance,
            )
            candidate_counts.append(len(candidates))
            tgt = target_coords(sample_dir)
            sample_has_success = False

            for rank, cand in enumerate(candidates, start=1):
                centroid_distance, min_distance = target_distance_stats(cand["centroid_xyz"], tgt)
                label_distance = centroid_distance if args.distance_label == "centroid" else min_distance
                success = int(label_distance <= args.success_cutoff_nm)
                sample_has_success = sample_has_success or bool(success)
                stats = component_tensor_stats(sample_dir, cand["indices"])
                row = {
                    "sample_id": sample_id,
                    "sample_path": sample_path,
                    "candidate_rank": rank,
                    "candidate_count": len(candidates),
                    "source_threshold": cand["threshold"],
                    "presence_prob": presence_prob,
                    "size_voxels": cand["size"],
                    "prob_mean": cand["mean_prob"],
                    "prob_max": cand["max_prob"],
                    "prob_top5_mean": cand["top5_mean_prob"],
                    "prob_centroid": cand["centroid_prob"],
                    "prob_local_mean": cand["local_mean_prob"],
                    "heuristic_score": cand["score"],
                    "centroid_x": float(cand["centroid_xyz"][0]),
                    "centroid_y": float(cand["centroid_xyz"][1]),
                    "centroid_z": float(cand["centroid_xyz"][2]),
                    "bbox_extent_x": cand["bbox_extent_x"],
                    "bbox_extent_y": cand["bbox_extent_y"],
                    "bbox_extent_z": cand["bbox_extent_z"],
                    "bbox_volume": cand["bbox_volume"],
                    "fill_ratio": cand["fill_ratio"],
                    "distance_to_reference_centroid_nm": centroid_distance,
                    "distance_to_reference_min_nm": min_distance,
                    "label_distance_nm": label_distance,
                    "success_1nm": success,
                }
                row.update(stats)
                rows.append(row)

            if sample_has_success:
                oracle_success += 1

    write_rows(args.output_csv, rows)
    sample_count = len(dataset)
    print(f"File list: {args.filepaths}")
    print(f"Samples evaluated: {sample_count}")
    print(f"Candidates written: {len(rows)}")
    print(f"Mean candidates per sample: {float(np.mean(candidate_counts)) if candidate_counts else 0.0:.6f}")
    print(f"Oracle candidate success @ {args.success_cutoff_nm:.2f} nm: {oracle_success}/{sample_count} ({oracle_success / sample_count:.6f})")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
