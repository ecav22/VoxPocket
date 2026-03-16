import argparse
from pathlib import Path

import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset

import utilities

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CandidateRanker(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=16, dropout_rate=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_float_list(arg):
    values = [float(x.strip()) for x in arg.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def load_split_paths(path_str):
    split_file = Path(path_str)
    with split_file.open("r") as f:
        paths = [line.strip() for line in f.readlines() if line.strip()]
    if not paths:
        raise ValueError(f"Split file is empty: {split_file}")
    return split_file, paths


def candidate_feature_vector(candidate, heuristic_rank, candidate_count, presence_prob):
    size = float(candidate["size"])
    mean_prob = float(candidate["mean_prob"])
    max_prob = float(candidate["max_prob"])
    top5_mean_prob = float(candidate["top5_mean_prob"])
    centroid_prob = float(candidate["centroid_prob"])
    local_mean_prob = float(candidate["local_mean_prob"])
    bbox_extent_x = float(candidate["bbox_extent_x"])
    bbox_extent_y = float(candidate["bbox_extent_y"])
    bbox_extent_z = float(candidate["bbox_extent_z"])
    bbox_volume = float(candidate["bbox_volume"])
    fill_ratio = float(candidate["fill_ratio"])
    heuristic_score = float(candidate["score"])
    rank_feature = 1.0 / float(heuristic_rank)
    count_feature = numpy.log1p(float(candidate_count))
    relative_rank = float(heuristic_rank) / float(max(candidate_count, 1))
    return numpy.asarray(
        [
            numpy.log1p(size),
            size,
            mean_prob,
            max_prob,
            top5_mean_prob,
            centroid_prob,
            local_mean_prob,
            heuristic_score,
            max_prob - mean_prob,
            mean_prob * numpy.log1p(size),
            bbox_extent_x,
            bbox_extent_y,
            bbox_extent_z,
            bbox_volume,
            fill_ratio,
            count_feature,
            rank_feature,
            relative_rank,
            presence_prob,
        ],
        dtype=numpy.float32,
    )


def build_candidate_examples(model, checkpoint_features, filepaths, device, thresholds, min_size, merge_distance, positive_cutoff):
    dataset = utilities.PocketDataset(filepaths, feature_names=checkpoint_features, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for idx, (features, target, pocket_present, centroid_target) in enumerate(loader):
            del target, pocket_present, centroid_target
            features = features.to(device)
            logits, presence_logits, centroid_logits = model(features)
            presence_prob = float(torch.sigmoid(presence_logits).detach().cpu().item())
            del centroid_logits

            pred = torch.sigmoid(logits).detach().cpu().numpy()
            sample_path = dataset.filepaths[idx]
            candidates = utilities.extract_ranked_candidates_multithreshold(
                sample_path,
                pred,
                thresholds=thresholds,
                min_size=min_size,
                merge_distance=merge_distance,
            )
            distances = utilities.candidate_distances_to_reference(sample_path, candidates)
            candidate_count = len(candidates)

            for rank, (candidate, distance) in enumerate(zip(candidates, distances), start=1):
                rows.append(
                    {
                        "sample_path": sample_path,
                        "pdb_id": Path(sample_path.rstrip("/")).name,
                        "heuristic_rank": rank,
                        "features": candidate_feature_vector(candidate, rank, candidate_count, presence_prob),
                        "distance": float(distance),
                        "label": 1.0 if float(distance) <= positive_cutoff else 0.0,
                        "heuristic_score": float(candidate["score"]),
                        "presence_prob": presence_prob,
                    }
                )
    return rows, dataset


def standardize_features(train_rows, other_row_groups):
    train_x = numpy.stack([row["features"] for row in train_rows], axis=0)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0

    def apply(rows):
        for row in rows:
            row["features_std"] = (row["features"] - mean) / std

    apply(train_rows)
    for rows in other_row_groups:
        apply(rows)

    return mean, std


def make_tensor_dataset(rows):
    x = torch.tensor(numpy.stack([row["features_std"] for row in rows], axis=0), dtype=torch.float32)
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    return TensorDataset(x, y)


def build_pairwise_examples(rows, max_pairs_per_sample=32, distance_margin=1e-6):
    by_sample = {}
    for row in rows:
        by_sample.setdefault(row["sample_path"], []).append(row)

    left_features = []
    right_features = []
    targets = []

    for sample_rows in by_sample.values():
        ordered = sorted(sample_rows, key=lambda row: row["distance"])
        sample_pairs = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                better = ordered[i]
                worse = ordered[j]
                if worse["distance"] - better["distance"] <= distance_margin:
                    continue
                sample_pairs.append((better["features_std"], worse["features_std"], 1.0))
        if not sample_pairs:
            continue
        if len(sample_pairs) > max_pairs_per_sample:
            sample_pairs = sample_pairs[:max_pairs_per_sample]
        for left, right, target in sample_pairs:
            left_features.append(left)
            right_features.append(right)
            targets.append(target)

    if not left_features:
        raise ValueError("No valid pairwise ranking examples were created. Check candidate generation or distance margins.")

    return (
        torch.tensor(numpy.stack(left_features, axis=0), dtype=torch.float32),
        torch.tensor(numpy.stack(right_features, axis=0), dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def score_rows(model, rows, device):
    if not rows:
        return
    x = torch.tensor(numpy.stack([row["features_std"] for row in rows], axis=0), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    for row, prob in zip(rows, probs):
        row["ranker_prob"] = float(prob)


def format_metric(value):
    return "N/A" if value is None else f"{value:.6f}"


def pairwise_ranking_loss(score_left, score_right, target):
    # target=1 means left should rank above right.
    margin = score_left - score_right
    return torch.nn.functional.binary_cross_entropy_with_logits(margin, target)


def evaluate_ranked_rows(rows, sample_paths, top_k, cutoffs):
    by_sample = {}
    for row in rows:
        by_sample.setdefault(row["sample_path"], []).append(row)

    sample_count = len(sample_paths)
    top1_success = {cutoff: 0 for cutoff in cutoffs}
    topk_success = {cutoff: 0 for cutoff in cutoffs}
    top1_distances = []
    topk_distances = []
    candidate_counts = []

    for sample_path in sample_paths:
        sample_rows = by_sample.get(sample_path, [])
        if not sample_rows:
            candidate_counts.append(0)
            continue
        ranked = sorted(
            sample_rows,
            key=lambda row: (row.get("ranker_prob", row["heuristic_score"]), row["heuristic_score"]),
            reverse=True,
        )
        candidate_counts.append(len(ranked))
        top1_distance = ranked[0]["distance"]
        best_topk_distance = min(row["distance"] for row in ranked[:top_k])
        top1_distances.append(top1_distance)
        topk_distances.append(best_topk_distance)
        for cutoff in cutoffs:
            if top1_distance <= cutoff:
                top1_success[cutoff] += 1
            if best_topk_distance <= cutoff:
                topk_success[cutoff] += 1

    metrics = {
        "sample_count": sample_count,
        "mean_candidates": float(numpy.mean(candidate_counts)) if candidate_counts else 0.0,
        "top1_mean_distance": float(numpy.mean(top1_distances)) if top1_distances else None,
        "top1_median_distance": float(numpy.median(top1_distances)) if top1_distances else None,
        "topk_mean_distance": float(numpy.mean(topk_distances)) if topk_distances else None,
        "topk_median_distance": float(numpy.median(topk_distances)) if topk_distances else None,
        "top1_success": {cutoff: top1_success[cutoff] / sample_count for cutoff in cutoffs},
        "topk_success": {cutoff: topk_success[cutoff] / sample_count for cutoff in cutoffs},
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-filepaths", type=str, default=str(PROJECT_ROOT / "config/splits/train.txt"))
    parser.add_argument("--val-filepaths", type=str, default=str(PROJECT_ROOT / "config/splits/val.txt"))
    parser.add_argument("--test-filepaths", type=str, default=str(PROJECT_ROOT / "config/splits/test.txt"))
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention_best.pt"))
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--thresholds", type=str, default=None, help="Optional comma-separated thresholds for multi-threshold candidate union")
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--merge-distance", type=float, default=0.35)
    parser.add_argument("--positive-cutoff", type=float, default=1.0, help="Candidates within this distance are labeled positive")
    parser.add_argument("--eval-cutoffs", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--ranking-mode", type=str, default="pairwise", choices=["pairwise", "binary"])
    parser.add_argument("--max-pairs-per-sample", type=int, default=32)
    parser.add_argument("--run-name", type=str, default="candidate_ranker")
    parser.add_argument("--results-csv", type=str, default=str(PROJECT_ROOT / "artifacts/torch/experiment_results.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("MPS detected but disabled: ConvTranspose3d is unsupported on MPS for this model.")

    eval_cutoffs = parse_float_list(args.eval_cutoffs)
    thresholds = parse_float_list(args.thresholds) if args.thresholds is not None else [args.threshold]
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_names = checkpoint.get("feature_names", utilities.FEATURE_ORDER)

    train_file, train_paths = load_split_paths(args.train_filepaths)
    val_file, val_paths = load_split_paths(args.val_filepaths)
    test_file, test_paths = load_split_paths(args.test_filepaths)

    segmentation_model = utilities.UNetAttention3D(
        in_channels=checkpoint.get("in_channels", len(feature_names)),
        dropout_rate=checkpoint.get("dropout_rate", 0.1),
    ).to(device)
    segmentation_model.load_state_dict(checkpoint["model_state_dict"])
    segmentation_model.eval()

    print("Building candidate tables from checkpoint predictions...")
    train_rows, train_dataset = build_candidate_examples(
        segmentation_model, feature_names, train_paths, device, thresholds, args.min_size, args.merge_distance, args.positive_cutoff
    )
    val_rows, val_dataset = build_candidate_examples(
        segmentation_model, feature_names, val_paths, device, thresholds, args.min_size, args.merge_distance, args.positive_cutoff
    )
    test_rows, test_dataset = build_candidate_examples(
        segmentation_model, feature_names, test_paths, device, thresholds, args.min_size, args.merge_distance, args.positive_cutoff
    )

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Candidate generation produced an empty split. Adjust threshold/min-size before training a ranker.")

    standardize_features(train_rows, [val_rows, test_rows])

    ranker = CandidateRanker(in_dim=train_rows[0]["features_std"].shape[0], hidden_dim=args.hidden_dim, dropout_rate=args.dropout_rate).to(device)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=args.learning_rate)
    if args.ranking_mode == "binary":
        train_ds = make_tensor_dataset(train_rows)
        val_ds = make_tensor_dataset(val_rows)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        train_pos = sum(row["label"] for row in train_rows)
        train_neg = len(train_rows) - train_pos
        pos_weight = float(train_neg / max(train_pos, 1.0))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        train_left, train_right, train_targets = build_pairwise_examples(
            train_rows, max_pairs_per_sample=args.max_pairs_per_sample
        )
        val_left, val_right, val_targets = build_pairwise_examples(
            val_rows, max_pairs_per_sample=args.max_pairs_per_sample
        )
        train_ds = TensorDataset(train_left, train_right, train_targets)
        val_ds = TensorDataset(val_left, val_right, val_targets)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        criterion = pairwise_ranking_loss

    best_state = None
    best_metrics = None
    best_score = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        ranker.train()
        epoch_loss = 0.0
        if args.ranking_mode == "binary":
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = ranker(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        else:
            for left_x, right_x, yb in train_loader:
                left_x = left_x.to(device)
                right_x = right_x.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                left_score = ranker(left_x)
                right_score = ranker(right_x)
                loss = criterion(left_score, right_score, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        epoch_loss /= max(len(train_loader), 1)

        ranker.eval()
        val_loss = 0.0
        with torch.no_grad():
            if args.ranking_mode == "binary":
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    val_loss += criterion(ranker(xb), yb).item()
            else:
                for left_x, right_x, yb in val_loader:
                    left_x = left_x.to(device)
                    right_x = right_x.to(device)
                    yb = yb.to(device)
                    val_loss += criterion(ranker(left_x), ranker(right_x), yb).item()
        val_loss /= max(len(val_loader), 1)

        score_rows(ranker, val_rows, device)
        val_metrics = evaluate_ranked_rows(val_rows, val_dataset.filepaths, args.top_k, eval_cutoffs)
        primary_cutoff = args.positive_cutoff
        val_score = (
            val_metrics["top1_success"].get(primary_cutoff, 0.0),
            val_metrics["topk_success"].get(primary_cutoff, 0.0),
            -1e9 if val_metrics["top1_mean_distance"] is None else -val_metrics["top1_mean_distance"],
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs}"
            f" - train_loss: {epoch_loss:.6f}"
            f" - val_loss: {val_loss:.6f}"
            f" - val_top1_success@{primary_cutoff:.2f}: {val_metrics['top1_success'].get(primary_cutoff, 0.0):.6f}"
            f" - val_top{args.top_k}_success@{primary_cutoff:.2f}: {val_metrics['topk_success'].get(primary_cutoff, 0.0):.6f}"
            f" - val_top1_mean_dist: {format_metric(val_metrics['top1_mean_distance'])}"
        )

        if best_score is None or val_score > best_score:
            best_score = val_score
            best_metrics = val_metrics
            best_state = {
                "model_state_dict": ranker.state_dict(),
                "feature_mean": numpy.stack([row["features"] for row in train_rows], axis=0).mean(axis=0),
                "feature_std": numpy.stack([row["features"] for row in train_rows], axis=0).std(axis=0),
                "candidate_thresholds": thresholds,
                "min_size": args.min_size,
                "merge_distance": args.merge_distance,
                "positive_cutoff": args.positive_cutoff,
                "eval_cutoffs": eval_cutoffs,
                "top_k": args.top_k,
                "feature_names": feature_names,
                "hidden_dim": args.hidden_dim,
                "dropout_rate": args.dropout_rate,
                "ranking_mode": args.ranking_mode,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs (patience={args.early_stop_patience}).")
                break

    ranker_ckpt = PROJECT_ROOT / "artifacts/torch/candidate_ranker_best.pt"
    if best_state is None:
        raise RuntimeError("Ranker training did not produce a best state.")
    torch.save(best_state, ranker_ckpt)
    print(f"Saved best candidate ranker checkpoint to {ranker_ckpt}")

    ranker.load_state_dict(best_state["model_state_dict"])
    ranker.eval()
    score_rows(ranker, test_rows, device)
    test_metrics = evaluate_ranked_rows(test_rows, test_dataset.filepaths, args.top_k, eval_cutoffs)

    print("Test split:", test_file)
    print("Candidate thresholds:", thresholds)
    print("Candidate min_size:", args.min_size)
    print("Merge distance:", args.merge_distance)
    print("Positive cutoff:", args.positive_cutoff)
    print(f"Mean candidates per sample: {test_metrics['mean_candidates']:.6f}")
    print(f"Top-1 mean distance: {format_metric(test_metrics['top1_mean_distance'])}")
    print(f"Top-1 median distance: {format_metric(test_metrics['top1_median_distance'])}")
    print(f"Top-{args.top_k} best mean distance: {format_metric(test_metrics['topk_mean_distance'])}")
    print(f"Top-{args.top_k} best median distance: {format_metric(test_metrics['topk_median_distance'])}")
    for cutoff in eval_cutoffs:
        print(f"Top-1 success @ {cutoff:.2f} nm: {test_metrics['top1_success'][cutoff]:.6f}")
        print(f"Top-{args.top_k} success @ {cutoff:.2f} nm: {test_metrics['topk_success'][cutoff]:.6f}")

    utilities.append_experiment_row(
        args.results_csv,
        {
            "phase": "candidate_ranker",
            "run_name": args.run_name,
            "features": ",".join(feature_names),
            "device": str(device),
            "toy_mode": "",
            "epochs": args.epochs,
            "steps_per_epoch": len(train_loader),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout_rate": args.dropout_rate,
            "raw_samples": train_dataset.raw_count + val_dataset.raw_count + test_dataset.raw_count,
            "usable_samples": train_dataset.usable_count + val_dataset.usable_count + test_dataset.usable_count,
            "skipped_samples": train_dataset.skipped_count + val_dataset.skipped_count + test_dataset.skipped_count,
            "corrupt_samples": train_dataset.corrupt_count + val_dataset.corrupt_count + test_dataset.corrupt_count,
            "mean_loss": "",
            "mean_custom_metrics": "",
            "threshold": ",".join(str(x) for x in thresholds),
            "voxel_accuracy": "",
            "dice": "",
            "iou": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "detected_count": int(round(test_metrics["top1_success"][args.positive_cutoff] * test_metrics["sample_count"])),
            "not_detected_count": test_metrics["sample_count"] - int(round(test_metrics["top1_success"][args.positive_cutoff] * test_metrics["sample_count"])),
            "detection_rate": test_metrics["top1_success"][args.positive_cutoff],
            "mean_distance_to_reference": test_metrics["top1_mean_distance"],
            "median_distance_to_reference": test_metrics["top1_median_distance"],
            "checkpoint_path": str(ranker_ckpt),
            "history_path": str(checkpoint_path),
            "notes": (
                f"candidate_ranker_complete;segmentation_checkpoint={checkpoint_path};"
                f"candidate_thresholds={','.join(str(x) for x in thresholds)};min_size={args.min_size};merge_distance={args.merge_distance};"
                f"positive_cutoff={args.positive_cutoff};top_k={args.top_k};ranking_mode={args.ranking_mode};"
                f"train_candidates={len(train_rows)};val_candidates={len(val_rows)};test_candidates={len(test_rows)};"
                + ";".join(
                    [
                        f"top1_success_at_{str(cutoff).replace('.', 'p')}nm={test_metrics['top1_success'][cutoff]:.6f};"
                        f"top{args.top_k}_success_at_{str(cutoff).replace('.', 'p')}nm={test_metrics['topk_success'][cutoff]:.6f}"
                        for cutoff in eval_cutoffs
                    ]
                )
            ),
        },
    )
    print(f"Appended candidate ranker results to {args.results_csv}")


if __name__ == "__main__":
    main()
