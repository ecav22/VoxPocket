from __future__ import division
import os
import csv
import numpy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset

from DANet_attention3D import DANet3D

FEATURE_ORDER = ["N", "bfactors", "buriedness", "charge", "radius", "hbdon", "hbacc", "sasa"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_feature_string(features_arg):
    if features_arg is None:
        return FEATURE_ORDER.copy()

    requested = [x.strip() for x in features_arg.split(",") if x.strip()]
    if len(requested) == 0:
        raise ValueError("No features specified.")

    invalid = [x for x in requested if x not in FEATURE_ORDER]
    if invalid:
        raise ValueError(f"Invalid features: {invalid}. Valid options: {FEATURE_ORDER}")

    seen = set()
    ordered_unique = []
    for feat in requested:
        if feat not in seen:
            ordered_unique.append(feat)
            seen.add(feat)

    return ordered_unique


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


def append_experiment_row(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    base_row.update(row)

    fieldnames = [
        "timestamp_utc",
        "slurm_job_id",
        "phase",
        "run_name",
        "features",
        "device",
        "toy_mode",
        "epochs",
        "steps_per_epoch",
        "batch_size",
        "learning_rate",
        "dropout_rate",
        "raw_samples",
        "usable_samples",
        "skipped_samples",
        "corrupt_samples",
        "mean_loss",
        "mean_custom_metrics",
        "threshold",
        "voxel_accuracy",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "detected_count",
        "not_detected_count",
        "detection_rate",
        "mean_distance_to_reference",
        "median_distance_to_reference",
        "checkpoint_path",
        "history_path",
        "notes",
    ]

    def ensure_experiment_csv_schema(path, expected_fields):
        if not path.exists():
            return
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            current_fields = reader.fieldnames or []
            existing_rows = list(reader)
        if current_fields == expected_fields:
            return
        # Rewrite the file with the expected schema, preserving overlapping columns.
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=expected_fields, extrasaction="ignore")
            writer.writeheader()
            for old_row in existing_rows:
                new_row = {k: old_row.get(k, "") for k in expected_fields}
                writer.writerow(new_row)

    ensure_experiment_csv_schema(csv_path, fieldnames)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(base_row)


def append_detection_row(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    base_row.update(row)

    fieldnames = [
        "timestamp_utc",
        "slurm_job_id",
        "run_name",
        "sample_path",
        "pdb_id",
        "threshold",
        "detected",
        "distance_to_reference",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(base_row)


def mean_max_scaler(non_zero_indices, tensor):
    max_ = numpy.max(tensor[non_zero_indices])
    min_ = numpy.min(tensor[non_zero_indices])
    mean_ = numpy.mean(tensor[non_zero_indices])
    denom = max_ - min_
    if denom == 0:
        return tensor
    tensor[non_zero_indices] = (tensor[non_zero_indices] - mean_) / denom
    return tensor


def load_numeric_npy(path):
    try:
        arr = numpy.load(path)
    except ValueError:
        # Some legacy files are stored as pickled numpy objects.
        arr = numpy.load(path, allow_pickle=True)

    arr = numpy.asarray(arr)
    if arr.dtype == object:
        # Try best-effort conversion to numeric array.
        arr = arr.astype(numpy.float32)
    return arr


def open_files(path, feature_names=None):
    if feature_names is None:
        feature_names = FEATURE_ORDER

    nzv = 0.0
    sample_dir = PROJECT_ROOT / path

    N = load_numeric_npy(sample_dir / "for_labview_protein_/N_tensor.npy")
    non_zeros = numpy.nonzero(N)
    zeros = numpy.nonzero(N == 0)

    N = N.astype(numpy.float32)
    N[non_zeros] = 1.0
    N[zeros] = nzv
    N = numpy.expand_dims(N, axis=3)

    bfactors = load_numeric_npy(sample_dir / "for_labview_protein_/bfactors_tensor.npy").astype(numpy.float32)
    bfactors = mean_max_scaler(non_zeros, bfactors)
    bfactors[zeros] = nzv
    bfactors = numpy.expand_dims(bfactors, axis=3)

    buriedness = load_numeric_npy(sample_dir / "for_labview_protein_/buriedness_tensor.npy").astype(numpy.float32)
    buriedness = mean_max_scaler(non_zeros, buriedness)
    buriedness[zeros] = nzv
    buriedness = numpy.expand_dims(buriedness, axis=3)

    charge = load_numeric_npy(sample_dir / "for_labview_protein_/charge_tensor.npy").astype(numpy.float32)
    charge = mean_max_scaler(non_zeros, charge)
    charge[zeros] = nzv
    charge = numpy.expand_dims(charge, axis=3)

    radius = load_numeric_npy(sample_dir / "for_labview_protein_/radius_tensor.npy").astype(numpy.float32)
    radius = mean_max_scaler(non_zeros, radius)
    radius[zeros] = nzv
    radius = numpy.expand_dims(radius, axis=3)

    hbdon = load_numeric_npy(sample_dir / "for_labview_protein_/hbdon_tensor.npy").astype(numpy.float32)
    hbdon = mean_max_scaler(non_zeros, hbdon)
    hbdon[zeros] = nzv
    hbdon = numpy.expand_dims(hbdon, axis=3)

    hbacc = load_numeric_npy(sample_dir / "for_labview_protein_/hbac_tensor.npy").astype(numpy.float32)
    hbacc = mean_max_scaler(non_zeros, hbacc)
    hbacc[zeros] = nzv
    hbacc = numpy.expand_dims(hbacc, axis=3)

    sasa = load_numeric_npy(sample_dir / "for_labview_protein_/sasa_tensor.npy").astype(numpy.float32)
    sasa = mean_max_scaler(non_zeros, sasa)
    sasa[zeros] = nzv
    sasa = numpy.expand_dims(sasa, axis=3)

    feature_map = {
        "N": N,
        "bfactors": bfactors,
        "buriedness": buriedness,
        "charge": charge,
        "radius": radius,
        "hbdon": hbdon,
        "hbacc": hbacc,
        "sasa": sasa,
    }
    features = numpy.concatenate([feature_map[name] for name in feature_names], axis=3)

    target = load_numeric_npy(sample_dir / "for_labview_pocket_/N_tensor_new_pocket.npy").astype(numpy.float32)
    non_zeros_target = numpy.nonzero(target)
    eta = 1e-11
    target[non_zeros] += eta
    target[non_zeros_target] = 1.0
    target[zeros] = nzv
    target = numpy.expand_dims(target, axis=3)

    return features.astype(numpy.float32), target.astype(numpy.float32)


class PocketDataset(Dataset):
    def __init__(self, filepaths, feature_names=None, augment=False):
        required = [
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

        raw_paths = [p.strip() for p in filepaths if p.strip()]
        valid_paths = []

        bad_data = 0
        for p in raw_paths:
            base = PROJECT_ROOT / p
            required_paths = [base / rel for rel in required]
            if not all(x.is_file() for x in required_paths):
                continue
            try:
                # Validate loadability to avoid runtime crashes in DataLoader.
                for npy_path in required_paths:
                    _ = load_numeric_npy(npy_path)
                valid_paths.append(p)
            except Exception:
                bad_data += 1

        skipped = len(raw_paths) - len(valid_paths)
        print(
            f"Dataset entries: {len(raw_paths)} total, {len(valid_paths)} usable, {skipped} skipped "
            f"(missing/corrupt tensors: {bad_data} corrupt)."
        )

        self.filepaths = valid_paths
        self.feature_names = FEATURE_ORDER if feature_names is None else feature_names
        self.raw_count = len(raw_paths)
        self.usable_count = len(valid_paths)
        self.skipped_count = skipped
        self.corrupt_count = bad_data
        self.augment = augment
        if len(self.filepaths) == 0:
            raise ValueError("No usable dataset entries found. Run tensor generation scripts first.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        features, target = open_files(self.filepaths[idx], feature_names=self.feature_names)
        x = torch.from_numpy(features).permute(3, 0, 1, 2)  # [C, D, H, W]
        y = torch.from_numpy(target).permute(3, 0, 1, 2)    # [1, D, H, W]
        if self.augment:
            x, y = random_3d_augmentation(x, y)
        return x, y


def random_3d_augmentation(x, y):
    # Apply shared 90-degree rotations and flips so geometry stays aligned.
    spatial_dims = [1, 2, 3]

    for dim in spatial_dims:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[dim])
            y = torch.flip(y, dims=[dim])

    rot_planes = [(1, 2), (1, 3), (2, 3)]
    plane = rot_planes[torch.randint(0, len(rot_planes), (1,)).item()]
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        x = torch.rot90(x, k=k, dims=plane)
        y = torch.rot90(y, k=k, dims=plane)

    return x.contiguous(), y.contiguous()


def custom_metrics(y_true, y_pred):
    eta = 1e-12
    mask = y_true > eta
    if mask.sum() == 0:
        return torch.tensor(0.0, device=y_true.device)
    return torch.mean(torch.abs(y_true[mask] - y_pred[mask]))


def segmentation_counts(y_true, y_pred_prob, threshold=0.5):
    eta = 1e-12
    mask = y_true > eta
    if mask.sum() == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    y_true_bin = y_true > 0.5
    y_pred_bin = y_pred_prob >= threshold

    y_true_masked = y_true_bin[mask]
    y_pred_masked = y_pred_bin[mask]

    tp = torch.logical_and(y_pred_masked, y_true_masked).sum().item()
    fp = torch.logical_and(y_pred_masked, torch.logical_not(y_true_masked)).sum().item()
    fn = torch.logical_and(torch.logical_not(y_pred_masked), y_true_masked).sum().item()
    tn = torch.logical_and(torch.logical_not(y_pred_masked), torch.logical_not(y_true_masked)).sum().item()

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def segmentation_metrics_from_counts(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "voxel_accuracy": accuracy,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def pocket_detected_and_distance(sample_path, predicted_tensor, threshold=0.5):
    """
    Determine if a pocket is detected and compute centroid distance to reference pocket.
    Returns (detected: bool, distance: float|None).
    """
    pdb = Path(sample_path.rstrip("/")).name
    xyz_protein, predicted_values, xyz_pocket_target = obtain_coordinates(pdb, predicted_tensor)
    predicted_values = numpy.asarray(predicted_values)

    detected_mask = predicted_values >= threshold
    detected = bool(numpy.any(detected_mask))
    if not detected:
        return False, None

    pred_coords = xyz_protein[detected_mask]
    pred_centroid = numpy.mean(pred_coords, axis=0)
    target_centroid = numpy.mean(xyz_pocket_target, axis=0)
    distance = float(numpy.linalg.norm(pred_centroid - target_centroid))
    return True, distance


class UNetAttention3D(nn.Module):
    def __init__(self, in_channels=8, dropout_rate=0.1):
        super().__init__()

        self.down1_conv = nn.Conv3d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.down1_bn = nn.BatchNorm3d(32)
        self.down1_act = nn.LeakyReLU(inplace=True)
        self.down1_drop = nn.Dropout3d(dropout_rate)

        self.bottom_conv = nn.Conv3d(32, 128, kernel_size=4, stride=2, padding=1)
        self.bottom_bn = nn.BatchNorm3d(128)
        self.bottom_act = nn.LeakyReLU(inplace=True)
        self.bottom_drop = nn.Dropout3d(dropout_rate)
        self.attn = DANet3D(128)

        self.up1_t = nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=1)
        self.up1_bn = nn.BatchNorm3d(32)
        self.up1_act = nn.LeakyReLU(inplace=True)
        self.up1_drop = nn.Dropout3d(dropout_rate)

        self.up2_t = nn.ConvTranspose3d(64, 8, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.up2_bn = nn.BatchNorm3d(8)
        self.up2_act = nn.LeakyReLU(inplace=True)
        self.up2_drop = nn.Dropout3d(dropout_rate)

        # u2 is concatenation of original input (in_channels) and upsample branch (8 channels)
        self.out_conv = nn.Conv3d(in_channels + 8, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        d1 = self.down1_drop(self.down1_act(self.down1_bn(self.down1_conv(x))))
        d2 = self.bottom_drop(self.bottom_act(self.bottom_bn(self.bottom_conv(d1))))
        d2 = self.attn(d2)

        u1 = self.up1_drop(self.up1_act(self.up1_bn(self.up1_t(d2))))
        u1 = torch.cat([d1, u1], dim=1)

        u2 = self.up2_drop(self.up2_act(self.up2_bn(self.up2_t(u1))))
        u2 = torch.cat([x, u2], dim=1)

        return self.out_conv(u2)


def obtain_coordinates(pdb, predicted_tensor):
    pred = predicted_tensor
    if pred.ndim == 5:
        pred = pred[0, 0]
    elif pred.ndim == 4:
        pred = pred[0]

    boundaries = numpy.loadtxt(PROJECT_ROOT / "refined-set" / pdb / "for_labview_protein_/axis_bins.txt")
    x_bins = boundaries[0]
    y_bins = boundaries[1]
    z_bins = boundaries[2]

    xyz_protein = numpy.loadtxt(PROJECT_ROOT / "refined-set" / pdb / "for_labview_protein_/xyz.txt")
    xyz_pocket_target = numpy.loadtxt(PROJECT_ROOT / "refined-set" / pdb / "for_labview_pocket_/xyz_new_pocket.txt")

    predicted_values = []
    for coord in xyz_protein:
        c_x, c_y, c_z = coord

        for x in range(len(x_bins) - 1):
            if (c_x >= x_bins[x]) and (c_x <= x_bins[x + 1]):
                a = x
                break
        for y in range(len(y_bins) - 1):
            if (c_y >= y_bins[y]) and (c_y <= y_bins[y + 1]):
                b = y
                break
        for z in range(len(z_bins) - 1):
            if (c_z >= z_bins[z]) and (c_z <= z_bins[z + 1]):
                c = z
                break

        predicted_values.append(float(pred[a, b, c]))

    return xyz_protein, predicted_values, xyz_pocket_target


def visualize(xyz_protein, predicted_values, xyz_pocket_target, pdb):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        xyz_protein[:, 0],
        xyz_protein[:, 1],
        xyz_protein[:, 2],
        c=predicted_values,
        marker=".",
        alpha=0.8,
        cmap=plt.cm.copper,
        s=numpy.array(predicted_values) * 20 + 0.05,
        vmin=0,
        vmax=1,
    )
    ax.axis("off")
    plt.title("prediction " + pdb)
    fig.colorbar(p, ax=ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz_pocket_target[:, 0], xyz_pocket_target[:, 1], xyz_pocket_target[:, 2], marker=".", alpha=0.8, c="sandybrown", s=20)
    ax.scatter(xyz_protein[:, 0], xyz_protein[:, 1], xyz_protein[:, 2], marker=".", alpha=0.3, c="k", s=1)
    ax.axis("off")
    plt.title("target " + pdb)
    plt.show()
