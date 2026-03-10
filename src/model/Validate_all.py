import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import utilities

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature names. If omitted, uses checkpoint feature_names.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for segmentation metrics (accuracy, Dice, IoU, etc.)",
    )
    parser.add_argument("--run-name", type=str, default="default", help="Label for CSV tracking")
    parser.add_argument(
        "--results-csv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/torch/experiment_results.csv"),
        help="CSV path for experiment logging",
    )
    parser.add_argument(
        "--detection-csv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/torch/detection_results.csv"),
        help="CSV path for per-sample detection/distance logging",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("MPS detected but disabled: ConvTranspose3d is unsupported on MPS for this model.")
    checkpoint = torch.load(PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention.pt", map_location=device)
    checkpoint_features = checkpoint.get("feature_names", utilities.FEATURE_ORDER)

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

    with open(PROJECT_ROOT / "config/filepaths.txt", "r") as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    if len(files) == 0:
        raise ValueError("filepaths.txt is empty. Run prepare_filenames.py first.")

    dataset = utilities.PocketDataset(files, feature_names=feature_names)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_metric = 0.0
    total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    detected_count = 0
    not_detected_count = 0
    distances = []

    with torch.no_grad():
        for idx, (features, target) in enumerate(loader):
            features = features.to(device)
            target = target.to(device)

            logits = model(features)
            pred = torch.sigmoid(logits)
            loss = criterion(logits, target)
            metric = utilities.custom_metrics(target, pred)
            counts = utilities.segmentation_counts(target, pred, threshold=args.threshold)

            total_loss += loss.item()
            total_metric += metric.item()
            total_counts["tp"] += counts["tp"]
            total_counts["fp"] += counts["fp"]
            total_counts["fn"] += counts["fn"]
            total_counts["tn"] += counts["tn"]

            sample_path = dataset.filepaths[idx]
            pred_np = pred.detach().cpu().numpy()
            detected, distance = utilities.pocket_detected_and_distance(sample_path, pred_np, threshold=args.threshold)
            if detected:
                detected_count += 1
                distances.append(distance)
            else:
                not_detected_count += 1

            utilities.append_detection_row(
                args.detection_csv,
                {
                    "run_name": args.run_name,
                    "sample_path": sample_path,
                    "pdb_id": sample_path.rstrip("/").split("/")[-1],
                    "threshold": args.threshold,
                    "detected": int(detected),
                    "distance_to_reference": "" if distance is None else distance,
                },
            )

    mean_loss = total_loss / len(loader)
    mean_metric = total_metric / len(loader)
    segm = utilities.segmentation_metrics_from_counts(total_counts)
    detection_rate = detected_count / len(loader)
    mean_distance = float(sum(distances) / len(distances)) if len(distances) > 0 else None
    median_distance = float(torch.tensor(distances).median().item()) if len(distances) > 0 else None

    print("Samples evaluated:", len(loader))
    print("Features:", feature_names)
    print(f"Threshold: {args.threshold}")
    print(f"Mean BCE loss: {mean_loss:.6f}")
    print(f"Mean custom_metrics: {mean_metric:.6f}")
    print(f"Voxel accuracy: {segm['voxel_accuracy']:.6f}")
    print(f"Dice: {segm['dice']:.6f}")
    print(f"IoU: {segm['iou']:.6f}")
    print(f"Precision: {segm['precision']:.6f}")
    print(f"Recall: {segm['recall']:.6f}")
    print(f"F1: {segm['f1']:.6f}")
    print(f"Pocket detected: {detected_count}/{len(loader)} ({detection_rate:.6f})")
    if mean_distance is not None:
        print(f"Mean distance to reference pocket: {mean_distance:.6f}")
        print(f"Median distance to reference pocket: {median_distance:.6f}")
    else:
        print("Mean distance to reference pocket: N/A (no detections)")

    utilities.append_experiment_row(
        args.results_csv,
        {
            "phase": "eval",
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
            "mean_loss": mean_loss,
            "mean_custom_metrics": mean_metric,
            "threshold": args.threshold,
            "voxel_accuracy": segm["voxel_accuracy"],
            "dice": segm["dice"],
            "iou": segm["iou"],
            "precision": segm["precision"],
            "recall": segm["recall"],
            "f1": segm["f1"],
            "detected_count": detected_count,
            "not_detected_count": not_detected_count,
            "detection_rate": detection_rate,
            "mean_distance_to_reference": "" if mean_distance is None else mean_distance,
            "median_distance_to_reference": "" if median_distance is None else median_distance,
            "checkpoint_path": str(PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention.pt"),
            "history_path": str(PROJECT_ROOT / "artifacts/torch/history_unet_bn_attention.pkl"),
            "notes": "eval_complete",
        },
    )
    print(f"Appended eval results to {args.results_csv}")
    print(f"Appended per-sample detections to {args.detection_csv}")


if __name__ == "__main__":
    main()
