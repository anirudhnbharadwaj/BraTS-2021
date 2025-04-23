import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nibabel as nib
import logging
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.inferers import sliding_window_inference
from medpy.metric.binary import hd95, assd

sns.set_style("darkgrid")


def compute_metrics(pred, gt):
    """Compute evaluation metrics and confusion matrix for specific tumor regions using MONAI and medpy."""
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match ground truth shape {gt.shape}")
    
    regions = {
        "ET": ([3], [3]),               # Enhancing Tumor: label 3
        "TC": ([1, 3], [1, 3]),         # Tumor Core: labels 1 (NCR/NET) + 3 (enhancing)
        "WT": ([1, 2, 3], [1, 2, 3])    # Whole Tumor: labels 1 (edema) + 2 (non-enhancing) + 3 (enhancing)
    }
    metrics = {}
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    confusion_metric = ConfusionMatrixMetric(metric_name=["recall", "precision", "specificity"], reduction="mean", compute_sample=True)

    for region_name, (pred_labels, gt_labels) in regions.items():
        pred_mask = np.isin(pred, pred_labels).astype(np.uint8)
        gt_mask = np.isin(gt, gt_labels).astype(np.uint8)
        
        # Log the number of non-zero voxels for debugging
        # pred_nonzero = np.count_nonzero(pred_mask)
        # gt_nonzero = np.count_nonzero(gt_mask)
        # logging.debug(f"Region {region_name}: pred_mask non-zero voxels = {pred_nonzero}, gt_mask non-zero voxels = {gt_nonzero}")

        # Convert to torch tensors for MONAI metrics
        pred_tensor = torch.tensor(pred_mask[None, None], dtype=torch.uint8)  # [1, 1, H, W, D]
        gt_tensor = torch.tensor(gt_mask[None, None], dtype=torch.uint8)      # [1, 1, H, W, D]

        # Dice Coefficient (MONAI)
        dice = dice_metric(pred_tensor, gt_tensor).item()
        if not np.any(gt_mask):
            dice = 1.0 if not np.any(pred_mask) else 0.0

        # Confusion Matrix Metrics (MONAI)
        confusion_metric(pred_tensor, gt_tensor)
        confusion_results = confusion_metric.aggregate()
        sensitivity = confusion_results[0].item()  # Recall
        precision = confusion_results[1].item()
        specificity = confusion_results[2].item()

        # Extract confusion matrix components (TP, FP, FN, TN)
        tp = int(np.logical_and(pred_mask, gt_mask).sum())
        fp = int(pred_mask.sum() - tp)
        fn = int(gt_mask.sum() - tp)
        tn = int(np.logical_and(~pred_mask, ~gt_mask).sum())

        # Jaccard (Derived from Dice)
        jaccard = dice / (2 - dice) if dice > 0 else 0.0

        # Hausdorff Distance 95% (medpy)
        hd_95 = hd95(pred_mask, gt_mask) if np.any(pred_mask) and np.any(gt_mask) else 0.0


        metrics[region_name] = {
            "Dice": dice,
            "Jaccard": jaccard,
            "Sensitivity": sensitivity,
            "Precision": precision,
            "Specificity": specificity,
            "HD95": hd_95,
            "ConfusionMatrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        }
        confusion_metric.reset()

    return metrics

def visualize_sample(data_list, save_path, config):
    """Visualize a sample's modalities and ground truth mask."""
    if not data_list:
        logging.error("data_list is empty in visualize_sample.")
        return
    
    sample = data_list[0]
    slice_idx = config["slice_idx"]
    try:
        img = nib.load(sample["image"]).get_fdata()
        lbl = nib.load(sample["seg"]).get_fdata()
    except FileNotFoundError as e:
        logging.error(f"Failed to load file: {e}")
        return
    
    if slice_idx >= img.shape[2]:
        logging.error(f"Slice index {slice_idx} out of bounds for image shape {img.shape}")
        return
    
    slice_data = img[:, :, slice_idx, :]
    label_slice = lbl[:, :, slice_idx]
    label_cmap = sns.color_palette("rocket", n_colors=4)
    
    logging.info(f"Image shape: {img.shape}, GT shape: {lbl.shape}, Slice idx: {slice_idx}")
    
    modalities = ["FLAIR", "T1", "T1CE", "T2"]
    label_names = ["Background", "Non-Enhancing Tumor", "Edema", "Enhancing Tumor"]
    
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    for i, modality in enumerate(modalities):
        sns.heatmap(slice_data[:, :, i], ax=axs[i], cmap="viridis", cbar=True)
        axs[i].set_title(f"{modality} Slice")
        axs[i].axis("off")
    sns.heatmap(label_slice, ax=axs[4], cmap=label_cmap, cbar=False)
    for l, c in zip(range(4), label_cmap):
        axs[4].plot([], [], color=c, label=label_names[l])
    axs[4].legend(loc="upper right")
    axs[4].set_title("Ground Truth Mask")
    axs[4].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Sample visualization saved to {save_path}")

def visualize_prediction(model, loader, model_name, save_path, config, device):
    """Visualize segmentation prediction for all modalities on validation data."""
    if not loader.dataset:
        logging.error("Dataset is empty in visualize_prediction.")
        return
    
    label_cmap = sns.color_palette("rocket", n_colors=4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        # Get the first batch from the loader
        val_batch = next(iter(loader))
        inputs = val_batch["image"].to(device)  # Shape: [batch_size, 4, H, W, D]
        labels = val_batch["seg"].to(device)    # Shape: [batch_size, 1, H, W, D]
        
        # Use the first sample from the batch
        inputs = inputs[0:1]  # Shape: [1, 4, H, W, D]
        labels = labels[0:1]  # Shape: [1, 1, H, W, D]
        
        # Perform inference
        outputs = sliding_window_inference(inputs, config["spatial_size"], 4, model)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]  # Shape: [H, W, D]
        gt = labels.cpu().numpy()[0, 0]                       # Shape: [H, W, D]
        
        # Get slice index from config, default to middle if not specified
        slice_idx = config.get("slice_idx", inputs.shape[4] // 2)
        if slice_idx >= inputs.shape[4]:
            logging.error(f"Slice index {slice_idx} out of bounds for input shape {inputs.shape}")
            return
        
        # Extract all modalities for the chosen slice
        modalities = ["FLAIR", "T1", "T1CE", "T2"]
        modality_data = inputs.cpu().numpy()[0, :, :, :, slice_idx]  # Shape: [4, H, W]
        
        # Create a figure with 2 rows: 4 modalities in row 1, GT and Pred in row 2
        fig = plt.figure(figsize=(20, 10))  # Adjusted figure size for layout
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])  # 2 rows, 4 columns
        
        # Row 1: Modalities (4 subplots)
        for i, modality in enumerate(modalities):
            ax = fig.add_subplot(gs[0, i])
            sns.heatmap(modality_data[i], ax=ax, cmap="viridis", cbar=True, square=True)
            ax.set_title(f"{modality} Image")
            ax.axis("off")
        
        # Row 2: Ground Truth and Prediction (centered in columns 1-2 and 2-3)
        ax_gt = fig.add_subplot(gs[1, 1:2])  # Span columns 1-2 (centered under modalities)
        sns.heatmap(gt[:, :, slice_idx], ax=ax_gt, cmap=label_cmap, cbar=True, square=True)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")
        
        ax_pred = fig.add_subplot(gs[1, 2:3])  # Span columns 2-3 (centered under modalities)
        sns.heatmap(pred[:, :, slice_idx], ax=ax_pred, cmap=label_cmap, cbar=True, square=True)
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")
        
        # Add a suptitle
        plt.suptitle(f"{model_name} Prediction vs Ground Truth (Slice {slice_idx})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit suptitle
        
        # Save the plot
        try:
            plt.savefig(save_path, dpi=300)  # Higher DPI for better quality
            plt.close()
            logging.info(f"Prediction visualization saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save prediction visualization: {e}")
            plt.close()
            
            
def plot_confusion_matrix(metrics_list, model_name, save_path):
    """Plot confusion matrices for each tumor region."""
    if not metrics_list:
        logging.error("metrics_list is empty in plot_confusion_matrix.")
        return
    
    os.makedirs(save_path, exist_ok=True)
    regions = ["ET", "TC", "WT"]
    
    for region in regions:
        try:
            total_tp = sum(m[region]["ConfusionMatrix"]["TP"] for m in metrics_list)
            total_fp = sum(m[region]["ConfusionMatrix"]["FP"] for m in metrics_list)
            total_fn = sum(m[region]["ConfusionMatrix"]["FN"] for m in metrics_list)
            total_tn = sum(m[region]["ConfusionMatrix"]["TN"] for m in metrics_list)
            
            cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
            cm_df = pd.DataFrame(cm, index=["Negative", "Positive"], columns=["Negative", "Positive"])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
            plt.title(f"Confusion Matrix - {model_name} ({region})", fontsize=16)
            plt.ylabel("True Label", fontsize=12)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.savefig(os.path.join(save_path, f"confusion_matrix_{region.lower()}.png"))
            plt.close()
        except Exception as e:
            logging.error(f"Failed to plot confusion matrix for {region}: {e}")
            plt.close()
    
    logging.info(f"Confusion matrix plots saved to {save_path} for regions {', '.join(regions)}")

def plot_metrics(metrics_list, model_name, save_path):
    """Plot all metrics for tumor regions."""
    if not metrics_list:
        logging.error("metrics_list is empty in plot_metrics.")
        return
    
    os.makedirs(save_path, exist_ok=True)
    metric_names = ["Dice", "Jaccard", "Sensitivity", "Precision", "Specificity", "HD95"]
    regions = ["ET", "TC", "WT"]

    for metric_name in metric_names:
        try:
            metric_data = {region: [m[region][metric_name] for m in metrics_list if np.isfinite(m[region][metric_name])] for region in regions}
            if not any(metric_data.values()):
                logging.warning(f"No valid data for {metric_name}, skipping plot")
                continue
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=pd.DataFrame(metric_data), palette="rocket")
            plt.title(f"{model_name} - {metric_name} Scores", fontsize=16)
            plt.ylabel(f"{metric_name}", fontsize=12)
            plt.xlabel("Region", fontsize=12)
            plt.savefig(os.path.join(save_path, f"{metric_name.lower()}.png"))
            plt.close()
        except Exception as e:
            logging.error(f"Failed to plot {metric_name}: {e}")
            plt.close()
    
    logging.info(f"Metrics plots saved to {save_path} for {', '.join(metric_names)}")

def plot_loss_curves(loss_history, model_name, save_path):
    """Plot training loss across folds."""
    if not loss_history or not isinstance(loss_history, list) or not all(isinstance(l, list) for l in loss_history):
        logging.error("loss_history is empty or invalid in plot_loss_curves.")
        return
    
    os.makedirs(save_path, exist_ok=True)
    num_folds = len(loss_history)
    epochs = range(1, len(loss_history[0]) + 1) if loss_history and loss_history[0] else []
    
    try:
        plt.figure(figsize=(12, 6))
        for fold in range(num_folds):
            plt.plot(epochs, loss_history[fold], label=f"Fold {fold + 1}", marker='o')
        plt.title(f"{model_name} - Training Loss Across Folds", fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "loss.png"))
        plt.close()
        logging.info(f"Loss curve plot saved to {save_path}/loss.png")
    except Exception as e:
        logging.error(f"Failed to plot loss curves: {e}")
        plt.close()

def generate_table(metrics_list, save_path):
    """Generate a table summarizing metrics across folds."""
    if not metrics_list:
        logging.error("metrics_list is empty in generate_table.")
        return None
    
    table = []
    regions = ["ET", "TC", "WT"]
    metric_names = ["Dice", "Jaccard", "Sensitivity", "Precision", "Specificity", "HD95"]
    
    for fold, metrics in enumerate(metrics_list):
        row = [fold + 1]
        for region in regions:
            for metric_name in metric_names:
                row.append(metrics[region][metric_name])
        table.append(row)
    
    avg_row = ["Average"]
    for region in regions:
        for metric_name in metric_names:
            values = [m[region][metric_name] for m in metrics_list if np.isfinite(m[region][metric_name])]
            avg_value = np.mean(values) if values else 0.0
            avg_row.append(avg_value)
    
    columns = ["Fold"]
    for region in regions:
        for metric_name in metric_names:
            columns.append(f"{region}_{metric_name}")
    
    df = pd.DataFrame(table + [avg_row], columns=columns)
    try:
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, "metrics_table.csv"), index=False)
        logging.info(f"Metrics table saved to {save_path}/metrics_table.csv")
    except Exception as e:
        logging.error(f"Failed to save metrics table: {e}")
        return None
    return df