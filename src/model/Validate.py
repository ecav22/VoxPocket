import pickle
import numpy
import torch
from pathlib import Path

import utilities

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("MPS detected but disabled: ConvTranspose3d is unsupported on MPS for this model.")
    checkpoint = torch.load(PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention.pt", map_location=device)
    feature_names = checkpoint.get("feature_names", utilities.FEATURE_ORDER)

    model = utilities.UNetAttention3D(
        in_channels=checkpoint.get("in_channels", len(feature_names)),
        dropout_rate=checkpoint.get("dropout_rate", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(PROJECT_ROOT / "artifacts/torch/history_unet_bn_attention.pkl", "rb") as f:
        history_data = pickle.load(f)
    print("Loaded history keys:", list(history_data.keys()))

    with open(PROJECT_ROOT / "config/filepaths.txt", "r") as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    if len(files) == 0:
        raise ValueError("filepaths.txt is empty. Run prepare_filenames.py first.")

    dataset = utilities.PocketDataset(files, feature_names=feature_names)
    sample_path = dataset.filepaths[0]
    pdb = sample_path.rstrip("/").split("/")[-1]
    print("Validating sample:", sample_path)

    obs, target = utilities.open_files(sample_path, feature_names=feature_names)

    obs_t = torch.from_numpy(obs).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    target_t = torch.from_numpy(target).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(obs_t)
        loss = criterion(logits, target_t).item()
        pred = torch.sigmoid(logits)
        metric = utilities.custom_metrics(target_t, pred).item()

    print(f"Validation loss: {loss:.6f}")
    print(f"Validation custom_metrics: {metric:.6f}")

    predicted_tensor = pred.detach().cpu().numpy()
    xyz_protein, predicted_values, xyz_pocket_target = utilities.obtain_coordinates(pdb, predicted_tensor)
    utilities.visualize(xyz_protein, predicted_values, xyz_pocket_target, pdb)


if __name__ == "__main__":
    main()
