import argparse
import csv
from pathlib import Path


def load_rows(csv_path: Path):
    with csv_path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value is None or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize VoxPocket benchmark evaluation rows into a compact comparison table.')
    parser.add_argument('--results-csv', type=str, required=True)
    parser.add_argument('--run-pattern', action='append', required=True, help='Substring to match in run_name; can be passed multiple times')
    parser.add_argument('--output-file', type=str, default='', help='Optional markdown file to write the summary table to')
    args = parser.parse_args()

    rows = load_rows(Path(args.results_csv))
    matched = []
    for row in rows:
        run_name = row.get('run_name', '')
        if any(pat in run_name for pat in args.run_pattern):
            matched.append(row)

    if not matched:
        raise ValueError('No rows matched the requested run patterns.')

    header = '| Run | Samples | Threshold | Dice/F1 | IoU | Detection | Mean Dist. | Median Dist. |'
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|'
    table = [header, sep]

    for row in matched:
        usable = row.get('usable_samples', '')
        detected = row.get('detected_count', '')
        rate = to_float(row.get('detection_rate', ''))
        detection = f"{detected}/{usable} ({rate*100:.1f}%)" if usable and detected and rate is not None else row.get('detection_rate', '')
        dice = row.get('f1', '') or row.get('dice', '')
        table.append(
            '| {run} | {samples} | {thr} | {dice} | {iou} | {det} | {mean_d} | {med_d} |'.format(
                run=row.get('run_name', ''),
                samples=usable,
                thr=row.get('threshold', ''),
                dice=dice,
                iou=row.get('iou', ''),
                det=detection,
                mean_d=row.get('mean_distance_to_reference', ''),
                med_d=row.get('median_distance_to_reference', ''),
            )
        )

    out = '\n'.join(table) + '\n'
    print(out)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(out)
        print(f'Wrote markdown summary to {output_path}')
