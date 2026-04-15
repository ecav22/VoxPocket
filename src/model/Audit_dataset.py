import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "for_labview_protein_/N_tensor.npy",
    "for_labview_protein_/bfactors_tensor.npy",
    "for_labview_protein_/buriedness_tensor.npy",
    "for_labview_protein_/charge_tensor.npy",
    "for_labview_protein_/radius_tensor.npy",
    "for_labview_protein_/hbdon_tensor.npy",
    "for_labview_protein_/hbac_tensor.npy",
    "for_labview_protein_/sasa_tensor.npy",
    "for_labview_pocket_/N_tensor_new_pocket.npy",
]


def load_paths(list_file):
    with Path(list_file).open("r") as f:
        return [line.strip() for line in f if line.strip()]


def load_numeric_npy(path):
    try:
        arr = numpy.load(path)
    except ValueError:
        arr = numpy.load(path, allow_pickle=True)

    arr = numpy.asarray(arr)
    if arr.dtype == object:
        arr = arr.astype(numpy.float32)
    return arr


def mean_max_scaler(non_zero_indices, tensor):
    max_ = numpy.max(tensor[non_zero_indices])
    min_ = numpy.min(tensor[non_zero_indices])
    mean_ = numpy.mean(tensor[non_zero_indices])
    denom = max_ - min_
    if denom == 0:
        return tensor
    tensor = tensor.copy()
    tensor[non_zero_indices] = (tensor[non_zero_indices] - mean_) / denom
    return tensor


def open_files_minimal(path):
    nzv = 0.0
    sample_dir = PROJECT_ROOT / path

    N = load_numeric_npy(sample_dir / "for_labview_protein_/N_tensor.npy")
    non_zeros = numpy.nonzero(N)
    zeros = numpy.nonzero(N == 0)

    N = N.astype(numpy.float32)
    N[non_zeros] = 1.0
    N[zeros] = nzv
    N = numpy.expand_dims(N, axis=3)

    feature_arrays = [N]
    for rel in [
        "for_labview_protein_/bfactors_tensor.npy",
        "for_labview_protein_/buriedness_tensor.npy",
        "for_labview_protein_/charge_tensor.npy",
        "for_labview_protein_/radius_tensor.npy",
        "for_labview_protein_/hbdon_tensor.npy",
        "for_labview_protein_/hbac_tensor.npy",
        "for_labview_protein_/sasa_tensor.npy",
    ]:
        arr = load_numeric_npy(sample_dir / rel).astype(numpy.float32)
        arr = mean_max_scaler(non_zeros, arr)
        arr[zeros] = nzv
        arr = numpy.expand_dims(arr, axis=3)
        feature_arrays.append(arr)

    features = numpy.concatenate(feature_arrays, axis=3)

    target = load_numeric_npy(sample_dir / "for_labview_pocket_/N_tensor_new_pocket.npy").astype(numpy.float32)
    non_zeros_target = numpy.nonzero(target)
    eta = 1e-11
    target[non_zeros] += eta
    target[non_zeros_target] = 1.0
    target[zeros] = nzv
    target = numpy.expand_dims(target, axis=3)

    return features.astype(numpy.float32), target.astype(numpy.float32)


def audit_sample(rel_path):
    sample_dir = PROJECT_ROOT / rel_path
    result = {
        "sample_path": rel_path,
        "sample_dir": str(sample_dir),
        "status": "usable",
        "reason": "",
        "details": "",
    }

    missing = []
    for rel in REQUIRED_FILES:
        path = sample_dir / rel
        if not path.is_file():
            missing.append(rel)

    if missing:
        result["status"] = "missing_files"
        result["reason"] = "missing_required_files"
        result["details"] = ";".join(missing)
        return result

    for rel in REQUIRED_FILES:
        path = sample_dir / rel
        try:
            arr = load_numeric_npy(path)
        except Exception as exc:
            result["status"] = "corrupt_tensor"
            result["reason"] = "load_failed"
            result["details"] = f"{rel}: {type(exc).__name__}: {exc}"
            return result

        arr = numpy.asarray(arr)
        if arr.size == 0:
            result["status"] = "invalid_tensor"
            result["reason"] = "empty_array"
            result["details"] = rel
            return result

        if not numpy.issubdtype(arr.dtype, numpy.number):
            result["status"] = "invalid_tensor"
            result["reason"] = "non_numeric_dtype"
            result["details"] = f"{rel}: dtype={arr.dtype}"
            return result

    try:
        features, target = open_files_minimal(rel_path)
    except Exception as exc:
        result["status"] = "pipeline_failure"
        result["reason"] = type(exc).__name__
        result["details"] = str(exc)
        return result

    if features.ndim != 4 or target.ndim != 4:
        result["status"] = "invalid_tensor"
        result["reason"] = "unexpected_rank"
        result["details"] = f"features_ndim={features.ndim};target_ndim={target.ndim}"
        return result

    if features.shape[:3] != target.shape[:3]:
        result["status"] = "invalid_tensor"
        result["reason"] = "shape_mismatch"
        result["details"] = f"features={features.shape};target={target.shape}"
        return result

    if numpy.isnan(features).any() or numpy.isnan(target).any():
        result["status"] = "invalid_tensor"
        result["reason"] = "nan_values"
        result["details"] = ""
        return result

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepaths",
        type=str,
        default=str(PROJECT_ROOT / "config/filepaths.txt"),
        help="Path to dataset filepaths list",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick audits; 0 means audit all samples",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/torch/dataset_audit.txt"),
        help="Where to save the detailed text report",
    )
    args = parser.parse_args()

    filepaths = load_paths(args.filepaths)
    if args.limit > 0:
        filepaths = filepaths[: args.limit]

    results = [audit_sample(path) for path in filepaths]
    status_counts = Counter(r["status"] for r in results)
    reason_counts = Counter(r["reason"] for r in results if r["reason"])
    by_status = defaultdict(list)
    for row in results:
        by_status[row["status"]].append(row)

    report_lines = []
    report_lines.append(f"Audit file list: {args.filepaths}")
    report_lines.append(f"Samples checked: {len(results)}")
    report_lines.append("")
    report_lines.append("Status counts:")
    for status, count in status_counts.most_common():
        report_lines.append(f"- {status}: {count}")
    report_lines.append("")
    report_lines.append("Reason counts:")
    for reason, count in reason_counts.most_common():
        report_lines.append(f"- {reason}: {count}")
    report_lines.append("")

    for status in sorted(by_status.keys()):
        if status == "usable":
            continue
        report_lines.append(f"{status}:")
        for row in by_status[status][:50]:
            detail_suffix = f" | {row['details']}" if row["details"] else ""
            report_lines.append(f"- {row['sample_path']} | {row['reason']}{detail_suffix}")
        if len(by_status[status]) > 50:
            report_lines.append(f"- ... {len(by_status[status]) - 50} more")
        report_lines.append("")

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"Audit file list: {args.filepaths}")
    print(f"Samples checked: {len(results)}")
    print("Status counts:")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}")
    print("Reason counts:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")
    print(f"Wrote detailed report to {report_path}")


if __name__ == "__main__":
    main()
