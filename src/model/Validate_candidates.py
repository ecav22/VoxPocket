import argparse
from pathlib import Path

import numpy
import torch
from torch.utils.data import DataLoader

import utilities

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def summarize_distances(distances):
    if not distances:
        return None, None
    arr = numpy.asarray(distances, dtype=numpy.float32)
    return float(arr.mean()), float(numpy.median(arr))


def parse_float_list(arg):
    values = [float(x.strip()) for x in arg.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepaths",
        type=str,
        default=str(PROJECT_ROOT / "config/splits/test.txt"),
        help="Path to evaluation filepaths list",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature names. If omitted, uses checkpoint feature_names.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention_best.pt"),
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold for candidate extraction. Defaults to checkpoint best_val_threshold if present, else 0.5.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional comma-separated thresholds for multi-threshold candidate union",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k candidate window for best-candidate metrics")
    parser.add_argument("--min-size", type=int, default=3, help="Minimum voxel component size for a candidate")
    parser.add_argument("--merge-distance", type=float, default=0.35, help="Centroid distance (nm) used to merge duplicate candidates across thresholds")
    parser.add_argument(
        "--distance-cutoffs",
        type=str,
        default="0.5,1.0,1.5",
        help="Comma-separated centroid-distance cutoffs (nm) used to compute top-1 and top-k success rates",
    )
    parser.add_argument("--run-name", type=str, default="candidate_eval", help="Label for CSV tracking")
    parser.add_argument(
        "--results-csv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/torch/experiment_results.csv"),
        help="CSV path for experiment logging",
    )
    parser.add_argument(
        "--candidate-csv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/torch/candidate_results.csv"),
        help="CSV path for per-candidate logging",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("MPS detected but disabled: ConvTranspose3d is unsupported on MPS for this model.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_features = checkpoint.get("feature_names", utilities.FEATURE_ORDER)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("best_val_threshold", 0.5)
    thresholds = parse_float_list(args.thresholds) if args.thresholds is not None else [threshold]
    distance_cutoffs = parse_float_list(args.distance_cutoffs)
    if not distance_cutoffs:
        raise ValueError("At least one distance cutoff must be provided.")

    if args.features is None:
        feature_names = checkpoint_features
    else:
        feature_names = utilities.parse_feature_string(args.features)
        if feature_names != checkpoint_features:
            raise ValueError(
                f"Feature mismatch. Checkpoint was trained with {checkpoint_features}, "
                f"but requested {feature_names}. Retrain model with requested features first."
            )

    model = utilities.UNetAttention3D(
        in_channels=checkpoint.get("in_channels", len(feature_names)),
        dropout_rate=checkpoint.get("dropout_rate", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_file = Path(args.filepaths)
    with eval_file.open("r") as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    if not files:
        raise ValueError(f"Evaluation file list is empty: {eval_file}")

    dataset = utilities.PocketDataset(files, feature_names=feature_names, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    top1_detected = 0
    topk_candidate_present = 0
    top1_distances = []
    topk_best_distances = []
    candidate_counts = []
    top1_success = {cutoff: 0 for cutoff in distance_cutoffs}
    topk_success = {cutoff: 0 for cutoff in distance_cutoffs}

    with torch.no_grad():
        for idx, (features, target, pocket_present, centroid_target) in enumerate(loader):
            del target, pocket_present, centroid_target
            features = features.to(device)
            logits, presence_logits, centroid_logits = model(features)
            del presence_logits, centroid_logits

            pred = torch.sigmoid(logits).detach().cpu().numpy()
            sample_path = dataset.filepaths[idx]
            candidates = utilities.extract_ranked_candidates_multithreshold(
                sample_path,
                pred,
                thresholds=thresholds,
                min_size=args.min_size,
                merge_distance=args.merge_distance,
            )
            distances = utilities.candidate_distances_to_reference(sample_path, candidates)
            candidate_counts.append(len(candidates))

            if candidates:
                top1_detected += 1
                top1_distance = distances[0]
                top1_distances.append(top1_distance)
                for cutoff in distance_cutoffs:
                    if top1_distance <= cutoff:
                        top1_success[cutoff] += 1

                window = distances[: args.top_k]
                best_topk_distance = min(window)
                topk_candidate_present += 1
                topk_best_distances.append(best_topk_distance)
                for cutoff in distance_cutoffs:
                    if best_topk_distance <= cutoff:
                        topk_success[cutoff] += 1
            else:
                top1_distance = None

            pdb_id = Path(sample_path.rstrip("/")).name
            for rank, candidate in enumerate(candidates[: args.top_k], start=1):
                utilities.append_candidate_row(
                    args.candidate_csv,
                    {
                        "run_name": args.run_name,
                        "sample_path": sample_path,
                        "pdb_id": pdb_id,
                        "threshold": ",".join(str(x) for x in thresholds),
                        "candidate_rank": rank,
                        "candidate_score": candidate["score"],
                        "candidate_size": candidate["size"],
                        "candidate_mean_prob": candidate["mean_prob"],
                        "candidate_max_prob": candidate["max_prob"],
                        "candidate_centroid_x": candidate["centroid_xyz"][0],
                        "candidate_centroid_y": candidate["centroid_xyz"][1],
                        "candidate_centroid_z": candidate["centroid_xyz"][2],
                        "distance_to_reference": distances[rank - 1],
                        "is_top1": int(rank == 1),
                        "is_top3": int(rank <= min(args.top_k, 3)),
                    },
                )

    sample_count = len(loader)
    top1_mean, top1_median = summarize_distances(top1_distances)
    topk_mean, topk_median = summarize_distances(topk_best_distances)
    mean_candidates = float(numpy.mean(candidate_counts)) if candidate_counts else 0.0

    print("Samples evaluated:", sample_count)
    print("Evaluation file list:", eval_file)
    print("Features:", feature_names)
    print(f"Thresholds: {thresholds}")
    print(f"Merge distance: {args.merge_distance}")
    print(f"Top-1 candidate present: {top1_detected}/{sample_count} ({top1_detected / sample_count:.6f})")
    print(f"Top-{args.top_k} candidate present: {topk_candidate_present}/{sample_count} ({topk_candidate_present / sample_count:.6f})")
    print(f"Mean candidates per sample: {mean_candidates:.6f}")
    if top1_mean is not None:
        print(f"Top-1 mean distance to reference pocket: {top1_mean:.6f}")
        print(f"Top-1 median distance to reference pocket: {top1_median:.6f}")
        print(f"Top-{args.top_k} best mean distance to reference pocket: {topk_mean:.6f}")
        print(f"Top-{args.top_k} best median distance to reference pocket: {topk_median:.6f}")
    else:
        print("Top-1 mean distance to reference pocket: N/A (no candidates)")
    for cutoff in distance_cutoffs:
        print(
            f"Top-1 success @ {cutoff:.2f} nm: "
            f"{top1_success[cutoff]}/{sample_count} ({top1_success[cutoff] / sample_count:.6f})"
        )
        print(
            f"Top-{args.top_k} success @ {cutoff:.2f} nm: "
            f"{topk_success[cutoff]}/{sample_count} ({topk_success[cutoff] / sample_count:.6f})"
        )

    utilities.append_experiment_row(
        args.results_csv,
        {
            "phase": "candidate_eval",
            "run_name": args.run_name,
            "features": ",".join(feature_names),
            "device": str(device),
            "toy_mode": "",
            "epochs": "",
            "steps_per_epoch": "",
            "batch_size": 1,
            "learning_rate": "",
            "dropout_rate": checkpoint.get("dropout_rate", ""),
            "raw_samples": dataset.raw_count,
            "usable_samples": dataset.usable_count,
            "skipped_samples": dataset.skipped_count,
            "corrupt_samples": dataset.corrupt_count,
            "mean_loss": "",
            "mean_custom_metrics": "",
            "threshold": ",".join(str(x) for x in thresholds),
            "voxel_accuracy": "",
            "dice": "",
            "iou": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "detected_count": top1_detected,
            "not_detected_count": sample_count - top1_detected,
            "detection_rate": top1_detected / sample_count,
            "mean_distance_to_reference": "" if top1_mean is None else top1_mean,
            "median_distance_to_reference": "" if top1_median is None else top1_median,
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(PROJECT_ROOT / "artifacts/torch/history_unet_bn_attention.pkl"),
            "notes": (
                f"candidate_eval_complete;top_k={args.top_k};min_size={args.min_size};"
                f"merge_distance={args.merge_distance};"
                f"topk_candidate_present_rate={topk_candidate_present / sample_count:.6f};"
                f"topk_mean_distance={'' if topk_mean is None else topk_mean};"
                f"topk_median_distance={'' if topk_median is None else topk_median};"
                f"mean_candidates={mean_candidates:.6f};"
                + ";".join(
                    [
                        f"top1_success_at_{str(cutoff).replace('.', 'p')}nm={top1_success[cutoff] / sample_count:.6f};"
                        f"top{args.top_k}_success_at_{str(cutoff).replace('.', 'p')}nm={topk_success[cutoff] / sample_count:.6f}"
                        for cutoff in distance_cutoffs
                    ]
                )
            ),
        },
    )
    print(f"Appended candidate eval results to {args.results_csv}")
    print(f"Appended per-candidate rows to {args.candidate_csv}")


if __name__ == "__main__":
    main()
