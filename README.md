# Pocket Prediction

Code and data pipeline for protein pocket prediction on PDBbind-derived structures.

## Repository Structure

- `src/preprocessing/`: structure cleanup, feature extraction, tensor generation, pocket target generation
- `src/model/`: PyTorch model, training, and evaluation scripts
- `config/`: file lists and checkpoint tracking files
- `artifacts/torch/`: trained model checkpoints and training history
- `docs/`: project and dataset documentation
- `metadata/`: original index files from PDBbind package
- `legacy/`: archived backups/old experiments/LabVIEW files
- `refined-set/`: processed dataset directories (not recommended for GitHub commit)

## Quick Start

Generate/update file list:

```bash
python3 src/model/prepare_filenames.py
```

Train (toy mode currently enabled in `src/model/Model.py`):

```bash
python3 src/model/Model.py --features N,bfactors,buriedness,charge
```

Evaluate all:

```bash
python3 src/model/Validate_all.py --features N,bfactors,buriedness,charge
```

Evaluate one sample with visualization:

```bash
python3 src/model/Validate.py
```
