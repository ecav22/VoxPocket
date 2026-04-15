import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "candidate_rank", "candidate_count", "source_threshold", "presence_prob",
    "size_voxels", "prob_mean", "prob_max", "prob_top5_mean", "prob_centroid", "prob_local_mean",
    "heuristic_score", "bbox_extent_x", "bbox_extent_y", "bbox_extent_z", "bbox_volume", "fill_ratio",
    "N_mean", "N_max", "N_sum", "bfactors_mean", "bfactors_max", "bfactors_sum",
    "buriedness_mean", "buriedness_max", "buriedness_sum", "charge_mean", "charge_max", "charge_sum",
    "radius_mean", "radius_max", "radius_sum", "hbdon_mean", "hbdon_max", "hbdon_sum",
    "hbacc_mean", "hbacc_max", "hbacc_sum", "sasa_mean", "sasa_max", "sasa_sum",
]


def load_table(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + ["success_1nm"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def evaluate_rows(df, score_col, top_k, cutoffs):
    top1_dist = []
    topk_dist = []
    cand_counts = []
    top1_success = {c: 0 for c in cutoffs}
    topk_success = {c: 0 for c in cutoffs}
    samples = list(df["sample_path"].drop_duplicates())
    for sample in samples:
        sdf = df[df["sample_path"] == sample].sort_values([score_col, "heuristic_score"], ascending=False)
        cand_counts.append(len(sdf))
        if len(sdf) == 0:
            continue
        d1 = float(sdf.iloc[0]["label_distance_nm"])
        dk = float(sdf.head(top_k)["label_distance_nm"].min())
        top1_dist.append(d1)
        topk_dist.append(dk)
        for cutoff in cutoffs:
            if d1 <= cutoff:
                top1_success[cutoff] += 1
            if dk <= cutoff:
                topk_success[cutoff] += 1
    n = len(samples)
    return {
        "samples": n,
        "mean_candidates": float(np.mean(cand_counts)) if cand_counts else 0.0,
        "top1_mean_distance": float(np.mean(top1_dist)) if top1_dist else None,
        "top1_median_distance": float(np.median(top1_dist)) if top1_dist else None,
        "topk_mean_distance": float(np.mean(topk_dist)) if topk_dist else None,
        "topk_median_distance": float(np.median(topk_dist)) if topk_dist else None,
        "top1_success": {c: top1_success[c] / n if n else 0.0 for c in cutoffs},
        "topk_success": {c: topk_success[c] / n if n else 0.0 for c in cutoffs},
    }


def print_metrics(title, metrics, top_k, cutoffs):
    print(title)
    print(f"Samples evaluated: {metrics['samples']}")
    print(f"Mean candidates per sample: {metrics['mean_candidates']:.6f}")
    print(f"Top-1 mean distance: {metrics['top1_mean_distance']:.6f}")
    print(f"Top-1 median distance: {metrics['top1_median_distance']:.6f}")
    print(f"Top-{top_k} best mean distance: {metrics['topk_mean_distance']:.6f}")
    print(f"Top-{top_k} best median distance: {metrics['topk_median_distance']:.6f}")
    for cutoff in cutoffs:
        print(f"Top-1 success @ {cutoff:.2f} nm: {metrics['top1_success'][cutoff]:.6f}")
        print(f"Top-{top_k} success @ {cutoff:.2f} nm: {metrics['topk_success'][cutoff]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train a RandomForest candidate ranker from candidate CSV tables.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cutoffs", default="0.5,1.0,1.5")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise SystemExit("scikit-learn is required. Install it in pocket_torch or run with an env that has sklearn.") from exc

    cutoffs = [float(x.strip()) for x in args.cutoffs.split(",") if x.strip()]
    train = load_table(args.train_csv)
    val = load_table(args.val_csv)
    test = load_table(args.test_csv)

    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["success_1nm"].to_numpy(dtype=int)
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training labels contain only one class; adjust thresholds/cutoff or use a larger training set.")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    for df in [train, val, test]:
        df["ranker_score"] = model.predict_proba(df[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]

    print(f"Train candidates: {len(train)} positives: {int(train['success_1nm'].sum())}")
    print(f"Val candidates: {len(val)} positives: {int(val['success_1nm'].sum())}")
    print(f"Test candidates: {len(test)} positives: {int(test['success_1nm'].sum())}")
    print()
    print_metrics("=== Heuristic Test Ranking ===", evaluate_rows(test, "heuristic_score", args.top_k, cutoffs), args.top_k, cutoffs)
    print()
    print_metrics("=== RandomForest Test Ranking ===", evaluate_rows(test, "ranker_score", args.top_k, cutoffs), args.top_k, cutoffs)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "top_k": args.top_k, "cutoffs": cutoffs}, f)
    print(f"Saved model: {out}")


if __name__ == "__main__":
    main()
