from __future__ import division
import argparse
import pickle
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import utilities

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(utilities.FEATURE_ORDER),
        help="Comma-separated feature names. Valid: " + ",".join(utilities.FEATURE_ORDER),
    )
    parser.add_argument("--toy", type=str, default="true", help="Toy mode true/false")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs when toy=false")
    parser.add_argument("--toy-epochs", type=int, default=1, help="Training epochs when toy=true")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-samples", type=int, default=24, help="Max samples when toy=true")
    parser.add_argument("--max-steps", type=int, default=12, help="Max steps per epoch when toy=true")
    parser.add_argument("--run-name", type=str, default="default", help="Label for CSV tracking")
    parser.add_argument(
        "--results-csv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/torch/experiment_results.csv"),
        help="CSV path for experiment logging",
    )
    args = parser.parse_args()
    feature_names = utilities.parse_feature_string(args.features)

    toy_mode = utilities.parse_bool(args.toy)
    toy_max_samples = args.max_samples
    toy_max_steps_per_epoch = args.max_steps

    with open(PROJECT_ROOT / "config/filepaths.txt", "r") as f:
        files = [line.strip() for line in f.readlines() if line.strip()]

    if len(files) == 0:
        raise ValueError("filepaths.txt is empty. Run prepare_filenames.py first.")

    batch_size = args.batch_size
    num_epochs = args.toy_epochs if toy_mode else args.epochs
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # ConvTranspose3d is not supported on MPS; run this architecture on CPU when CUDA is unavailable.
        device = torch.device("cpu")

    if toy_mode:
        files = files[:toy_max_samples]

    print("Using device:", device)
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("MPS detected but disabled: ConvTranspose3d is unsupported on MPS for this model.")
    print("Training samples:", len(files))
    print("Toy mode:", toy_mode)
    print("Features:", feature_names)
    print("Run name:", args.run_name)

    dataset = utilities.PocketDataset(files, feature_names=feature_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if toy_mode:
        steps_per_epoch = min(len(loader), toy_max_steps_per_epoch)
    else:
        steps_per_epoch = len(loader)

    model = utilities.UNetAttention3D(in_channels=len(feature_names), dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    history = {"loss": [], "custom_metrics": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metric = 0.0

        for step, (features, target) in enumerate(loader):
            if step >= steps_per_epoch:
                break

            features = features.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(logits)
            metric = utilities.custom_metrics(target, preds)

            epoch_loss += loss.item()
            epoch_metric += metric.item()

        epoch_loss /= steps_per_epoch
        epoch_metric /= steps_per_epoch
        history["loss"].append(epoch_loss)
        history["custom_metrics"].append(epoch_metric)

        print(f"Epoch {epoch + 1}/{num_epochs} - steps: {steps_per_epoch} - loss: {epoch_loss:.6f} - custom_metrics: {epoch_metric:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_channels": len(feature_names),
        "feature_names": feature_names,
        "dropout_rate": dropout_rate,
        "history": history,
    }
    checkpoint_path = PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention.pt"
    history_path = PROJECT_ROOT / "artifacts/torch/history_unet_bn_attention.pkl"
    torch.save(checkpoint, checkpoint_path)

    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    final_loss = history["loss"][-1] if len(history["loss"]) > 0 else None
    final_metric = history["custom_metrics"][-1] if len(history["custom_metrics"]) > 0 else None
    utilities.append_experiment_row(
        args.results_csv,
        {
            "phase": "train",
            "run_name": args.run_name,
            "features": ",".join(feature_names),
            "device": str(device),
            "toy_mode": toy_mode,
            "epochs": num_epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "raw_samples": dataset.raw_count,
            "usable_samples": dataset.usable_count,
            "skipped_samples": dataset.skipped_count,
            "corrupt_samples": dataset.corrupt_count,
            "mean_loss": final_loss,
            "mean_custom_metrics": final_metric,
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(history_path),
            "notes": "train_complete",
        },
    )

    print("Saved artifacts/torch/model_unet_bn_attention.pt and artifacts/torch/history_unet_bn_attention.pkl")
    print(f"Appended train results to {args.results_csv}")


if __name__ == "__main__":
    main()
