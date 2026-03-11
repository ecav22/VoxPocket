from __future__ import division
import argparse
import pickle
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import utilities

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def dice_loss_from_logits(y_true, logits, eps=1e-8):
    preds = torch.sigmoid(logits)
    mask = y_true > 1e-12
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    y_true_bin = (y_true > 0.5).float()
    mask_f = mask.float()
    preds = preds * mask_f
    y_true_bin = y_true_bin * mask_f

    intersection = (preds * y_true_bin).sum()
    denom = preds.sum() + y_true_bin.sum()
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice


def make_loss_fn(loss_name, pos_weight, dice_weight, device):
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    def loss_fn(y_true, logits):
        if loss_name == "bce":
            return bce(logits, y_true)
        if loss_name == "dice":
            return dice_loss_from_logits(y_true, logits)
        if loss_name == "bce_dice":
            return bce(logits, y_true) + dice_weight * dice_loss_from_logits(y_true, logits)
        raise ValueError(f"Unsupported loss name: {loss_name}")

    return loss_fn


def evaluate_loader(model, loader, loss_fn, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            logits = model(features)
            pred = torch.sigmoid(logits)

            total_loss += loss_fn(target, logits).item()
            total_metric += utilities.custom_metrics(target, pred).item()
            counts = utilities.segmentation_counts(target, pred, threshold=threshold)
            total_counts["tp"] += counts["tp"]
            total_counts["fp"] += counts["fp"]
            total_counts["fn"] += counts["fn"]
            total_counts["tn"] += counts["tn"]

    n = len(loader)
    mean_loss = total_loss / n
    mean_metric = total_metric / n
    dice = utilities.segmentation_metrics_from_counts(total_counts)["dice"]
    return mean_loss, mean_metric, dice


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
    parser.add_argument("--loss", type=str, default="bce_dice", choices=["bce", "dice", "bce_dice"], help="Loss function")
    parser.add_argument("--pos-weight", type=float, default=1.0, help="Positive class weight for BCE")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight when using bce_dice")
    parser.add_argument(
        "--val-filepaths",
        type=str,
        default=str(PROJECT_ROOT / "config/splits/val.txt"),
        help="Path to validation filepaths list; ignored if missing/empty",
    )
    parser.add_argument("--save-best", type=str, default="true", help="Save best-by-val-dice checkpoint true/false")
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
    save_best = utilities.parse_bool(args.save_best)
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
    print("Loss:", args.loss, "pos_weight:", args.pos_weight, "dice_weight:", args.dice_weight)

    dataset = utilities.PocketDataset(files, feature_names=feature_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = None
    val_dataset = None
    val_file = Path(args.val_filepaths)
    if val_file.exists():
        with open(val_file, "r") as f:
            val_files = [line.strip() for line in f.readlines() if line.strip()]
        if len(val_files) > 0:
            val_dataset = utilities.PocketDataset(val_files, feature_names=feature_names)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            print("Validation samples:", len(val_dataset))
        else:
            print("Validation file exists but is empty; skipping validation.")
    else:
        print("Validation file not found; skipping validation.")

    if toy_mode:
        steps_per_epoch = min(len(loader), toy_max_steps_per_epoch)
    else:
        steps_per_epoch = len(loader)

    model = utilities.UNetAttention3D(in_channels=len(feature_names), dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = make_loss_fn(args.loss, args.pos_weight, args.dice_weight, device)

    history = {"loss": [], "custom_metrics": [], "val_loss": [], "val_custom_metrics": [], "val_dice": []}
    best_val_dice = -1.0
    best_checkpoint_path = PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention_best.pt"

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
            loss = loss_fn(target, logits)
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

        msg = f"Epoch {epoch + 1}/{num_epochs} - steps: {steps_per_epoch} - loss: {epoch_loss:.6f} - custom_metrics: {epoch_metric:.6f}"
        if val_loader is not None:
            val_loss, val_metric, val_dice = evaluate_loader(model, val_loader, loss_fn, device, threshold=0.5)
            history["val_loss"].append(val_loss)
            history["val_custom_metrics"].append(val_metric)
            history["val_dice"].append(val_dice)
            msg += f" - val_loss: {val_loss:.6f} - val_custom_metrics: {val_metric:.6f} - val_dice: {val_dice:.6f}"

            if save_best and val_dice > best_val_dice:
                best_val_dice = val_dice
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "in_channels": len(feature_names),
                    "feature_names": feature_names,
                    "dropout_rate": dropout_rate,
                    "history": history,
                    "best_val_dice": best_val_dice,
                    "best_epoch": epoch + 1,
                    "loss_name": args.loss,
                    "pos_weight": args.pos_weight,
                    "dice_weight": args.dice_weight,
                }
                torch.save(best_checkpoint, best_checkpoint_path)
        print(msg)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_channels": len(feature_names),
        "feature_names": feature_names,
        "dropout_rate": dropout_rate,
        "history": history,
        "loss_name": args.loss,
        "pos_weight": args.pos_weight,
        "dice_weight": args.dice_weight,
    }
    checkpoint_path = PROJECT_ROOT / "artifacts/torch/model_unet_bn_attention.pt"
    history_path = PROJECT_ROOT / "artifacts/torch/history_unet_bn_attention.pkl"
    torch.save(checkpoint, checkpoint_path)

    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    final_loss = history["loss"][-1] if len(history["loss"]) > 0 else None
    final_metric = history["custom_metrics"][-1] if len(history["custom_metrics"]) > 0 else None
    final_val_dice = history["val_dice"][-1] if len(history["val_dice"]) > 0 else None
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
            "notes": f"train_complete;loss={args.loss};pos_weight={args.pos_weight};dice_weight={args.dice_weight};final_val_dice={final_val_dice};best_val_dice={best_val_dice}",
        },
    )

    print("Saved artifacts/torch/model_unet_bn_attention.pt and artifacts/torch/history_unet_bn_attention.pkl")
    if save_best and val_loader is not None and best_val_dice >= 0:
        print(f"Saved best checkpoint by val dice: {best_checkpoint_path} (best_val_dice={best_val_dice:.6f})")
    print(f"Appended train results to {args.results_csv}")


if __name__ == "__main__":
    main()
