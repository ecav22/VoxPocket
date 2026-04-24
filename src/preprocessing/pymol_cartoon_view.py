import sys
from pathlib import Path

from pymol import cmd


def color_target_from_xyz(target_xyz_path: Path, protein_obj: str, pocket_obj: str = "pocket_atoms"):
    coords = []
    with target_xyz_path.open("r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                coords.append(tuple(float(x) for x in parts[:3]))
            except ValueError:
                continue

    if not coords:
        return

    pseudo_obj = "_target_points"
    cmd.delete(pseudo_obj)
    cmd.delete(pocket_obj)

    for x, y, z in coords:
        # xyz_new_pocket.txt is stored in nm; PyMOL expects Angstrom.
        cmd.pseudoatom(pseudo_obj, pos=[x * 10.0, y * 10.0, z * 10.0])

    cmd.select(pocket_obj, f"({protein_obj} within 2.0 of {pseudo_obj})")
    cmd.show("sticks", pocket_obj)
    cmd.color("tv_orange", pocket_obj)
    cmd.delete(pseudo_obj)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: pymol -r src/preprocessing/pymol_cartoon_view.py -- /path/to/protein.pdb [target_xyz.txt] [output_png]"
        )

    protein_path = Path(sys.argv[1]).resolve()
    target_xyz = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    output_png = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None

    if not protein_path.is_file():
        raise FileNotFoundError(f"Protein PDB not found: {protein_path}")
    if target_xyz is not None and not target_xyz.is_file():
        raise FileNotFoundError(f"Pocket xyz file not found: {target_xyz}")

    protein_obj = "protein"
    cmd.load(str(protein_path), protein_obj)
    cmd.hide("everything", "all")
    cmd.show("cartoon", protein_obj)
    cmd.color("slate", protein_obj)
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("depth_cue", 0)

    if target_xyz is not None:
        color_target_from_xyz(target_xyz, protein_obj)

    cmd.orient(protein_obj)
    cmd.zoom(protein_obj, buffer=4)

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        cmd.png(str(output_png), dpi=300, ray=1)
        print(f"Wrote image: {output_png}")
        cmd.quit()


if __name__ == "__main__":
    main()
