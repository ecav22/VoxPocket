import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Create a P2Rank .ds file from a VoxPocket sample file list.")
    parser.add_argument("--filepaths", required=True, help="VoxPocket sample directory file list, relative to project root")
    parser.add_argument("--output-ds", required=True, help="Output P2Rank dataset file")
    parser.add_argument("--prefer-cleaned", action="store_true", help="Use *_protein_cleaned.pdb when available")
    args = parser.parse_args()

    filepaths = Path(args.filepaths)
    out = Path(args.output_ds)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    missing = []
    for raw in filepaths.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        sample_dir = PROJECT_ROOT / raw
        sample_id = sample_dir.name
        cleaned = sample_dir / f"{sample_id}_protein_cleaned.pdb"
        protein = sample_dir / f"{sample_id}_protein.pdb"
        pdb = cleaned if args.prefer_cleaned and cleaned.is_file() else protein
        if not pdb.is_file():
            missing.append(str(sample_dir))
            continue
        # P2Rank accepts one structure path per line in a .ds file.
        lines.append(str(pdb.resolve()))

    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Input VoxPocket file list: {filepaths}")
    print(f"P2Rank structures written: {len(lines)}")
    print(f"Missing protein PDBs: {len(missing)}")
    print(f"Output .ds: {out}")
    if missing:
        print("First missing entries:")
        for item in missing[:20]:
            print(item)


if __name__ == "__main__":
    main()
