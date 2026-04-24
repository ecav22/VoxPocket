import sys
from pathlib import Path

from pymol import cmd


def load_xyz_points(xyz_path: Path):
    coords = []
    with xyz_path.open("r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                coords.append(tuple(float(x) for x in parts[:3]))
            except ValueError:
                continue
    return coords


def highlight_pocket(protein_obj: str, xyz_path: Path, distance_angstrom: float = 2.0):
    coords_nm = load_xyz_points(xyz_path)
    if not coords_nm:
        raise ValueError(f"No valid pocket coordinates found in {xyz_path}")

    points_obj = "_pocket_points"
    pocket_obj = "pocket_region"
    surface_obj = "pocket_surface"

    cmd.delete(points_obj)
    cmd.delete(pocket_obj)
    cmd.delete(surface_obj)

    for x_nm, y_nm, z_nm in coords_nm:
        cmd.pseudoatom(points_obj, pos=[x_nm * 10.0, y_nm * 10.0, z_nm * 10.0])

    cmd.select(pocket_obj, f"({protein_obj} within {distance_angstrom:.3f} of {points_obj})")
    cmd.create(surface_obj, pocket_obj)

    cmd.show("sticks", pocket_obj)
    cmd.color("tv_orange", pocket_obj)
    cmd.show("surface", surface_obj)
    cmd.set("transparency", 0.45, surface_obj)
    cmd.color("orange", surface_obj)

    cmd.delete(points_obj)


def main():
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: pymol -cq -r src/preprocessing/render_pocket_highlight.py -- "
            "/path/to/protein.pdb /path/to/xyz_new_pocket.txt /path/to/output.png [distance_angstrom]"
        )

    protein_path = Path(sys.argv[1]).resolve()
    xyz_path = Path(sys.argv[2]).resolve()
    output_png = Path(sys.argv[3]).resolve()
    distance_angstrom = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0

    if not protein_path.is_file():
        raise FileNotFoundError(f"Protein PDB not found: {protein_path}")
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Pocket xyz file not found: {xyz_path}")

    protein_obj = "protein"
    cmd.load(str(protein_path), protein_obj)
    cmd.hide("everything", "all")
    cmd.show("cartoon", protein_obj)
    cmd.color("gray70", protein_obj)
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("depth_cue", 0)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("antialias", 2)

    highlight_pocket(protein_obj, xyz_path, distance_angstrom=distance_angstrom)

    cmd.orient(protein_obj)
    cmd.zoom(protein_obj, buffer=4)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(output_png), dpi=300, ray=1)
    print(f"Wrote image: {output_png}")
    cmd.quit()


if __name__ == "__main__":
    main()
