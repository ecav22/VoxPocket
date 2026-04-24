import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_pdb_coords(pdb_path: Path):
    coords = []
    with pdb_path.open("r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            try:
                x = float(line[30:38]) / 10.0
                y = float(line[38:46]) / 10.0
                z = float(line[46:54]) / 10.0
            except ValueError:
                continue
            coords.append((x, y, z))
    arr = np.asarray(coords, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return arr


def load_xyz(sample_dir: Path):
    xyz_path = sample_dir / "for_labview_protein_" / "xyz.txt"
    if xyz_path.is_file():
        xyz = np.loadtxt(xyz_path, dtype=np.float32)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        return xyz

    sample_id = sample_dir.name
    for candidate in (
        sample_dir / f"{sample_id}_protein_lite.pdb",
        sample_dir / f"{sample_id}_protein_cleaned.pdb",
        sample_dir / f"{sample_id}_protein.pdb",
    ):
        if candidate.is_file():
            return parse_pdb_coords(candidate)
    raise FileNotFoundError(f"Could not find xyz.txt or a protein PDB in {sample_dir}")


def sample_points(xyz: np.ndarray, max_points: int):
    if xyz.shape[0] <= max_points:
        return xyz
    idx = np.linspace(0, xyz.shape[0] - 1, max_points).astype(int)
    return xyz[idx]


def voxel_centers(axis_bins: np.ndarray):
    return 0.5 * (axis_bins[:-1] + axis_bins[1:])


def active_voxels(tensor: np.ndarray, axis_bins: np.ndarray, threshold: float):
    coords = np.argwhere(tensor > threshold)
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    x_centers = voxel_centers(axis_bins[0])
    y_centers = voxel_centers(axis_bins[1])
    z_centers = voxel_centers(axis_bins[2])

    xyz = np.column_stack(
        [
            x_centers[coords[:, 0]],
            y_centers[coords[:, 1]],
            z_centers[coords[:, 2]],
        ]
    ).astype(np.float32)
    values = tensor[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    return xyz, values


def make_protein_plot(xyz: np.ndarray, title: str, output_path: Path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=3, alpha=0.55, c="black")
    ax.set_title(title)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")
    ax.view_init(elev=22, azim=38)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_voxel_plot(
    protein_xyz: np.ndarray,
    voxel_xyz: np.ndarray,
    voxel_values: np.ndarray,
    title: str,
    output_path: Path,
):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(protein_xyz[:, 0], protein_xyz[:, 1], protein_xyz[:, 2], s=1, alpha=0.08, c="gray")
    scatter = ax.scatter(
        voxel_xyz[:, 0],
        voxel_xyz[:, 1],
        voxel_xyz[:, 2],
        c=voxel_values,
        cmap="viridis",
        s=34,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")
    ax.view_init(elev=22, azim=38)
    fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08, label="voxel value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize a protein point cloud and its voxelized tensor.")
    parser.add_argument("--sample-dir", required=True, help="Benchmark sample directory.")
    parser.add_argument(
        "--channel",
        default="N",
        choices=["N", "bfactors", "buriedness", "sasa", "target"],
        help="Tensor channel to visualize.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Show voxels strictly greater than this value. Defaults: 0 for N/target, 0.05 for averaged channels.",
    )
    parser.add_argument(
        "--max-protein-points",
        type=int,
        default=8000,
        help="Maximum number of protein atoms plotted.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/figures/voxelization",
        help="Directory where PNGs will be written.",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.is_absolute():
        sample_dir = PROJECT_ROOT / sample_dir
    sample_dir = sample_dir.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    axis_bins = np.loadtxt(sample_dir / "for_labview_protein_" / "axis_bins.txt", dtype=np.float32)
    xyz = load_xyz(sample_dir)
    xyz_plot = sample_points(xyz, args.max_protein_points)

    tensor_paths = {
        "N": sample_dir / "for_labview_protein_" / "N_tensor.npy",
        "bfactors": sample_dir / "for_labview_protein_" / "bfactors_tensor.npy",
        "buriedness": sample_dir / "for_labview_protein_" / "buriedness_tensor.npy",
        "sasa": sample_dir / "for_labview_protein_" / "sasa_tensor.npy",
        "target": sample_dir / "for_labview_pocket_" / "N_tensor_new_pocket.npy",
    }
    tensor = np.load(tensor_paths[args.channel]).astype(np.float32)

    if args.threshold is None:
        threshold = 0.0 if args.channel in {"N", "target"} else 0.05
    else:
        threshold = args.threshold

    voxel_xyz, voxel_values = active_voxels(tensor, axis_bins, threshold=threshold)
    if voxel_xyz.shape[0] == 0:
        raise ValueError(
            f"No voxels in channel '{args.channel}' exceeded threshold {threshold} for sample {sample_dir.name}"
        )

    base_name = f"{sample_dir.name}_{args.channel}"
    protein_out = output_dir / f"{base_name}_protein.png"
    voxel_out = output_dir / f"{base_name}_voxels.png"

    make_protein_plot(xyz_plot, f"{sample_dir.name}: protein atoms", protein_out)
    make_voxel_plot(
        xyz_plot,
        voxel_xyz,
        voxel_values,
        f"{sample_dir.name}: {args.channel} voxels > {threshold:g}",
        voxel_out,
    )

    print(f"Sample: {sample_dir}")
    print(f"Channel: {args.channel}")
    print(f"Threshold: {threshold}")
    print(f"Protein figure: {protein_out}")
    print(f"Voxel figure: {voxel_out}")
    print(f"Plotted protein atoms: {xyz_plot.shape[0]}")
    print(f"Plotted voxels: {voxel_xyz.shape[0]}")


if __name__ == "__main__":
    main()
