# VoxPocket on P2Rank Benchmarks

This note explains how to evaluate VoxPocket on the benchmark datasets used by the P2Rank paper (`CHEN11`, `JOINED`, `COACH420`, `HOLO4K`).

## Important caveat

VoxPocket cannot be evaluated on these datasets directly from the `.ds` lists alone. The model expects fully preprocessed sample directories containing:

- `for_labview_protein_/...` tensors
- `for_labview_pocket_/N_tensor_new_pocket.npy`
- `xyz_new_pocket.txt`

In other words, the benchmark structures must first be converted into VoxPocket's processed sample format.

A second caveat is that some P2Rank benchmark structures, especially `COACH420`, are chain-specific files such as `1afkA.pdb`. By default the helper script maps these to 4-character PDB ids like `1afk`. This is useful only if your benchmark preprocessing also stores samples by PDB id. If you preprocess chain-specific structures into chain-specific sample folders, use `--id-mode stem` instead.

## 1. Build a VoxPocket file list from a benchmark `.ds` file

Example for `COACH420` after the benchmark has been preprocessed into project folders:

```bash
python3 src/model/prepare_p2rank_benchmark.py \
  --dataset-file /path/to/p2rank-datasets/coach420.ds \
  --processed-prefix refined-set \
  --id-mode pdb4 \
  --check-required-files \
  --output-file config/benchmarks/coach420_voxpocket.txt \
  --missing-file artifacts/torch/coach420_coverage.txt
```

Example for chain-specific preprocessing layout:

```bash
python3 src/model/prepare_p2rank_benchmark.py \
  --dataset-file /path/to/p2rank-datasets/coach420.ds \
  --processed-prefix benchmarks/coach420 \
  --id-mode stem \
  --check-required-files \
  --output-file config/benchmarks/coach420_voxpocket.txt \
  --missing-file artifacts/torch/coach420_coverage.txt
```

The script reports:

- total benchmark entries
- how many have usable VoxPocket tensors
- which samples are missing

## 2. Run VoxPocket evaluation on the benchmark file list

Use the best current checkpoint and threshold:

```bash
python3 src/model/Validate_all.py \
  --filepaths config/benchmarks/coach420_voxpocket.txt \
  --checkpoint artifacts/torch/model_unet_bn_attention_best.pt \
  --threshold 0.3 \
  --run-name coach420_voxpocket_presence_t0.3 \
  --results-csv artifacts/torch/experiment_results.csv \
  --detection-csv artifacts/torch/detection_results.csv
```

Repeat for `HOLO4K`, `CHEN11`, or `JOINED` with different file lists and run names.

## 3. Summarize benchmark runs into a compact table

```bash
python3 src/model/summarize_benchmark_runs.py \
  --results-csv artifacts/torch/experiment_results.csv \
  --run-pattern coach420_voxpocket_presence_t0.3 \
  --run-pattern holo4k_voxpocket_presence_t0.3 \
  --output-file artifacts/torch/p2rank_benchmark_summary.md
```

This produces a markdown table with:

- usable sample count
- threshold
- Dice/F1
- IoU
- detection rate
- mean and median distance

## 4. Comparison to P2Rank

The fairest comparison is:

- run VoxPocket on the same benchmark proteins
- run P2Rank on the same benchmark proteins or use benchmark predictions from the P2Rank dataset repo
- compare on a common metric set

Recommended common metrics:

- detection rate
- top-1 / top-k centroid distance
- Dice / IoU only if both methods produce compatible pocket masks

Because P2Rank is a ranked pocket predictor and VoxPocket is a dense segmentation model, centroid-distance and detection comparisons are generally more direct than Dice/IoU.
