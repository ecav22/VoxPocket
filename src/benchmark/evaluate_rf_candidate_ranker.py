import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


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
    print(f"Samples evaluated: {n}")
    print(f"Mean candidates per sample: {float(np.mean(cand_counts)) if cand_counts else 0.0:.6f}")
    print(f"Top-1 mean distance: {float(np.mean(top1_dist)):.6f}")
    print(f"Top-1 median distance: {float(np.median(top1_dist)):.6f}")
    print(f"Top-{top_k} best mean distance: {float(np.mean(topk_dist)):.6f}")
    print(f"Top-{top_k} best median distance: {float(np.median(topk_dist)):.6f}")
    for cutoff in cutoffs:
        print(f"Top-1 success @ {cutoff:.2f} nm: {top1_success[cutoff]}/{n} ({top1_success[cutoff] / n:.6f})")
        print(f"Top-{top_k} success @ {cutoff:.2f} nm: {topk_success[cutoff]}/{n} ({topk_success[cutoff] / n:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved RandomForest candidate ranker on a candidate CSV table.")
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cutoffs", default="0.5,1.0,1.5")
    parser.add_argument("--scored-csv", default="")
    args = parser.parse_args()

    cutoffs = [float(x.strip()) for x in args.cutoffs.split(",") if x.strip()]
    with Path(args.model).open("rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    feature_columns = payload["feature_columns"]
    df = pd.read_csv(args.candidate_csv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ranker_score"] = model.predict_proba(df[feature_columns].to_numpy(dtype=float))[:, 1]
    evaluate_rows(df, "ranker_score", args.top_k, cutoffs)
    if args.scored_csv:
        out = Path(args.scored_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote scored candidates: {out}")


if __name__ == "__main__":
    main()
