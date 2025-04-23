import logging

logging.basicConfig(
    level=logging.INFO,
    filename="info_att.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

import os
import json
import time
import argparse
import subprocess
import psutil
import socket

logging.info("Imported json, time, argparse, subprocess, psutil, socket")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

logging.info("Imported torch, torch.nn, torch.utils.data, torch.optim.lr_scheduler")

import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold

logging.info("Imported numpy, tqdm, pandas, sklearn.model_selection")

import wandb

logging.info("Imported wandb")

from monai.losses import DiceFocalLoss
from monai.utils import set_determinism
from monai.data import Dataset, pad_list_data_collate

logging.info("Imported monai.losses, monai.utils, monai.data")

from data.data import get_transforms, prepare_data, compute_class_distribution, compute_class_weights
from models.UNet.models import StandardUNet, AttentionUNet
from data.utils import compute_metrics, visualize_sample, visualize_prediction, plot_metrics, plot_loss_curves

logging.info("Imported data.data, models.UNet.models, data.utils")
logging.info("Finished importing modules.")

logging.info("\n")
logging.info("-" * 50)
logging.info("\n")

logging.info("Starting main function.")

logging.info("-" * 50)
logging.info("\n")

def check_internet():
    """Check internet connectivity."""
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        logging.info("Connected to google.com via port 80.")
        return True
    except (socket.timeout, OSError):
        logging.info("Failed to connect to a well-known host.")
        return False

def initialize_wandb():
    """Initialize W&B with online/offline mode."""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        logging.info("WANDB_API_KEY not set. Falling back to offline mode.")
        return False
    online = check_internet()
    mode = "online" if online else "offline"
    logging.info(f"WandB initializing in {mode} mode.")
    wandb.login(key=wandb_api_key)
    return online

def get_gpu_usage():
    """Get GPU utilization and memory usage for all GPUs via nvidia-smi."""
    if not torch.cuda.is_available():
        return [{"utilization": 0, "memory_used": 0, "memory_total": 0}]
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,nounits,noheader", shell=True, text=True)
        lines = output.strip().split('\n')
        gpu_data = []
        for i, line in enumerate(lines):
            utilization, mem_used, mem_total = map(float, line.split(','))
            gpu_data.append({
                "gpu_id": i,
                "utilization": utilization,
                "memory_used": mem_used / 1024,
                "memory_total": mem_total / 1024
            })
        return gpu_data
    except Exception as e:
        logging.warning(f"Failed to get GPU usage: {e}")
        return [{"gpu_id": i, "utilization": 0, "memory_used": 0, "memory_total": 0} for i in range(torch.cuda.device_count())]

def train_and_evaluate(model, model_name, train_loader, val_loader, config, device, class_weights, checkpoint_dir, output_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pth")
    
    loss_function = DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        gamma=config["gamma"],
        lambda_dice=config["lambda_dice"],
        lambda_focal=config["lambda_focal"]
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta_min"], config["beta_max"]),
        weight_decay=config["weight_decay"]
    )
    
    warmup_epochs = config["warmup_epochs"]
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["t_max"], eta_min=config["min_lr"])
    
    best_metric = -1
    loss_history = []
    val_dice_history = []  # Store WT Dice for validation steps
    val_jac_history = []  # Store JAC for validation steps
    val_sen_history = []  # Store sensitivity for validation steps
    val_prec_history = []  # Store precision for validation steps
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        loss_history = checkpoint["loss_history"]
        logging.info(f"Loaded checkpoint for {model_name} from epoch {start_epoch - 1}, best WT Dice: {best_metric:.4f}")
    else:
        logging.info(f"No checkpoint found for {model_name}, starting from scratch.")
    
    fold = int(model_name.split("fold")[-1]) if "fold" in model_name else -1
    
    for epoch in tqdm(range(start_epoch, config["max_epochs"]), desc=f"{model_name} Training", colour="green"):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        wandb.log({
            f"Fold {fold}/Training Loss": avg_epoch_loss,
            f"Fold {fold}/Epoch": epoch + 1,
            f"Fold {fold}/Learning Rate": optimizer.param_groups[0]["lr"]
        })
        
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        if (epoch + 1) % config["val_interval"] == 0 or (epoch + 1) == config["max_epochs"]:
            model.eval()
            val_metrics = {region: {metric: [] for metric in ["Dice", "Jaccard", "Sensitivity", "Precision", "Specificity", "HD95"]} for region in ["ET", "TC", "WT"]}
            with torch.no_grad():
                for val_batch in val_loader:
                    inputs = val_batch["image"].to(device)
                    labels = val_batch["seg"].to(device)
                    outputs = model(inputs)
                    pred = torch.argmax(outputs, dim=1).cpu().numpy()
                    gt = labels.cpu().numpy()[:, 0]
                    for i in range(pred.shape[0]):
                        batch_metrics = compute_metrics(pred[i], gt[i])
                        for region in val_metrics:
                            for metric_name in val_metrics[region]:
                                val_metrics[region][metric_name].append(batch_metrics[region][metric_name])
            
            avg_val_metrics = {region: {metric: np.mean(scores) for metric, scores in metrics.items()} for region, metrics in val_metrics.items()}
            val_dice_history.append(avg_val_metrics["WT"]["Dice"])
            val_jac_history.append(avg_val_metrics["WT"]["Jaccard"])
            val_sen_history.append(avg_val_metrics["WT"]["Sensitivity"])
            val_prec_history.append(avg_val_metrics["WT"]["Precision"])
            logging.info(f"{model_name} - Epoch {epoch + 1}, Validation Metrics:")
            for region in ["ET", "TC", "WT"]:
                logging.info(f"  {region}: " + ", ".join(f"{metric}={value:.4f}" for metric, value in avg_val_metrics[region].items()))
            
            wandb.log({
                f"Fold {fold}/Validation WT Dice": avg_val_metrics["WT"]["Dice"],
                f"Fold {fold}/Validation ET Dice": avg_val_metrics["ET"]["Dice"],
                f"Fold {fold}/Validation TC Dice": avg_val_metrics["TC"]["Dice"],
                f"Fold {fold}/Validation WT Jaccard": avg_val_metrics["WT"]["Jaccard"],
                f"Fold {fold}/Validation ET Jaccard": avg_val_metrics["ET"]["Jaccard"],
                f"Fold {fold}/Validation TC Jaccard": avg_val_metrics["TC"]["Jaccard"],
                f"Fold {fold}/Validation WT Sensitivity": avg_val_metrics["WT"]["Sensitivity"],
                f"Fold {fold}/Validation ET Sensitivity": avg_val_metrics["ET"]["Sensitivity"],
                f"Fold {fold}/Validation TC Sensitivity": avg_val_metrics["TC"]["Sensitivity"],
                f"Fold {fold}/Validation WT Precision": avg_val_metrics["WT"]["Precision"],
                f"Fold {fold}/Validation ET Precision": avg_val_metrics["ET"]["Precision"],
                f"Fold {fold}/Validation TC Precision": avg_val_metrics["TC"]["Precision"],
                f"Fold {fold}/Epoch": epoch + 1
            })
            time.sleep(3)  # Small delay after validation metrics
            
            val_dice = avg_val_metrics["WT"]["Dice"]
            if val_dice > best_metric:
                best_metric = val_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric": best_metric,
                    "loss_history": loss_history
                }, checkpoint_path)
                logging.info(f"New best model saved for {model_name} with WT Dice: {best_metric:.4f}")
    
    # Plot fold-specific training loss with validation markers
    loss_df = pd.DataFrame({
        "Epoch": list(range(1, len(loss_history) + 1)),
        "Training Loss": loss_history,
        "Validation WT Dice": [None] * len(loss_history)
    })
    val_epochs = [e for e in range(config["val_interval"] - 1, config["max_epochs"], config["val_interval"])] + [config["max_epochs"] - 1]
    for i, val_dice in zip(val_epochs, val_dice_history):
        loss_df.loc[i, "Validation WT Dice"] = val_dice
    
    wandb.log({
        f"Fold {fold}/Training Progress": wandb.plot.line_series(
            xs=[loss_df["Epoch"], loss_df["Epoch"]],
            ys=[loss_df["Training Loss"], loss_df["Validation WT Dice"]],
            keys=["Training Loss", "Validation WT Dice"],
            title=f"Fold {fold}: Training Loss & Validation WT Dice",
            xname="Epoch"
        )
    })
    time.sleep(3)  # Delay after plotting training progress
    
    return {"metrics": avg_val_metrics, "loss_history": loss_history}, best_metric

def main(config_path, model_choice):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    set_determinism(seed=config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    logging.info("-" * 50)
    logging.info("\n")
    
    logging.info(f"Using device: {device}, Number of GPUs: {num_gpus}")
    
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        logging.info(f"GPU names: {gpu_names}")
        try:
            logging.info(f"nvidia-smi:\n{subprocess.check_output('nvidia-smi', shell=True, text=True)}")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to run nvidia-smi: {e}")
            
    logging.info("-" * 50)
    logging.info("\n")
    
    logging.info("Hyperparameters:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
        
    logging.info("-" * 50)
    logging.info("\n")
    
    has_internet = initialize_wandb()
    wandb_mode = "online" if has_internet else "offline"
    logging.info(f"W&B mode: {wandb_mode}")
    
    logging.info("-" * 50)
    logging.info("\n")
    
    data_dir = os.path.join(config["root_path"], "BraTS2021_Training_Data")
    data_list = prepare_data(data_dir, config)
    
    logging.info("-" * 50)
    logging.info("\n")
    
    start_time = time.time()
    class_frequencies = compute_class_distribution(data_list)
    class_weights = compute_class_weights(class_frequencies)
    logging.info(f"Class distribution computed in {(time.time() - start_time):.2f}s")
    
    logging.info("-" * 50)
    logging.info("\n")
    
    
    all_models = {"StdUNet": StandardUNet, "AttUNet": AttentionUNet}
    models = all_models if model_choice == "both" else {model_choice: all_models[model_choice]}
    if model_choice not in ["both", "StdUNet", "AttUNet"]:
        raise ValueError(f"Invalid model choice: {model_choice}")
    
    best_metrics = {}
    all_results = {}
    
    for name, model_cls in models.items():
        output_dir = f"outputs_{name}"
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO, 
                            filename=os.path.join(output_dir, "training.log"),
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            filemode='w')
        
        wandb.init(project="BraTS2021_Segmentation", 
                   name=f"pri_{name}_{time.strftime('%Y%m%d_%H%M%S')}",
                   config=config,
                   mode=wandb_mode,
                   dir=output_dir)
        wandb.config.update({"model": name, "wandb_mode": wandb_mode, "num_gpus": num_gpus})
        
        logging.info(f"Starting training for {name}, dataset size: {len(data_list)}")
        
        logging.info("-" * 50)
        logging.info("\n")
        
        sample_path = os.path.join(output_dir, "sample_visualization.png")
        visualize_sample(data_list, sample_path, config)
        time.sleep(2)  
        wandb.log({"Sample Visualization": wandb.Image(sample_path)})
        
        logging.info("-" * 50)
        logging.info("\n")
        
        time.sleep(2)  
        
        kfold = KFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])
        results = {"metrics": [], "loss_history": [], "system_metrics": []}
        prediction_images = []
        
        for fold, (train_ids, val_ids) in enumerate(tqdm(kfold.split(data_list), desc=f"{name} CV Folds", total=config["k_folds"], colour="red")):
            logging.info(f"Fold {fold + 1}/{config['k_folds']}")
            train_subset = Subset(Dataset(data=data_list, transform=get_transforms(config, train=True)), train_ids)
            val_subset = Subset(Dataset(data=data_list, transform=get_transforms(config, train=False)), val_ids)
            
            train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], collate_fn=pad_list_data_collate)
            val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=pad_list_data_collate)
            
            model = model_cls().to(device)
            if num_gpus > 1:
                model = nn.DataParallel(model)
                logging.info(f"Model {name} wrapped with DataParallel for {num_gpus} GPUs")
            
            fold_results, fold_best_metric = train_and_evaluate(model, f"{name}_fold{fold}", train_loader, val_loader, config, device, class_weights, checkpoint_dir, output_dir)
            results["metrics"].append(fold_results["metrics"])
            results["loss_history"].append(fold_results["loss_history"])
            best_metrics[f"{name}_fold_{fold}"] = fold_best_metric
            
            # Generate and log prediction image
            pred_path = os.path.join(output_dir, f"prediction_fold{fold}.png")
            visualize_prediction(model, val_loader, f"{name}_fold{fold}", pred_path, config, device)
            while not os.path.exists(pred_path):
                time.sleep(2)
            logging.info(f"Image saved at {pred_path}")
            try:
                wandb.log({f"Fold {fold}/Prediction": wandb.Image(pred_path)})
                logging.info(f"Logged prediction image for Fold {fold}")
            except Exception as e:
                logging.error(f"Failed to log image: {e}")
            prediction_images.append(pred_path)
            time.sleep(3)  
            
            # Log system resources after each fold
            cpu_usage = psutil.cpu_percent(interval=1)
            gpu_usage = get_gpu_usage()
            system_metrics = {
                "CPU Usage (%)": cpu_usage,
                "Memory Used (GB)": psutil.virtual_memory().used / 1024**3,
                "Memory Available (GB)": psutil.virtual_memory().available / 1024**3
            }
            for gpu in gpu_usage:
                system_metrics[f"GPU {gpu['gpu_id']} Utilization (%)"] = gpu["utilization"]
                system_metrics[f"GPU {gpu['gpu_id']} Memory Used (GB)"] = gpu["memory_used"]
            wandb.log({f"Fold {fold}/System Resources": wandb.Table(
                columns=["Metric", "Value"],
                data=[
                    ["CPU Usage (%)", system_metrics["CPU Usage (%)"]],
                    ["Memory Used (GB)", system_metrics["Memory Used (GB)"]],
                    ["Memory Available (GB)", system_metrics["Memory Available (GB)"]],
                    *[ [f"GPU {gpu['gpu_id']} Utilization (%)", gpu["utilization"]] for gpu in gpu_usage ],
                    *[ [f"GPU {gpu['gpu_id']} Memory Used (GB)", gpu["memory_used"]] for gpu in gpu_usage ]
                ]
            )})
            time.sleep(2)  # Delay after logging system resources
            results["system_metrics"].append(system_metrics)
        
        # Aggregate Metrics for Visualization
        logging.info("\n")
        logging.info("-" * 50)
        logging.info("\n")
        plot_metrics(results["metrics"], name, output_dir)
        time.sleep(3)
        plot_loss_curves(results["loss_history"], name, output_dir)
        logging.info("\n")
        logging.info("-" * 50)
        logging.info("\n")
        
        time.sleep(3)
        wandb.log({
            "Aggregated Dice Scores": wandb.Image(os.path.join(output_dir, "dice.png")),
            "Aggregated Jaccard Scores": wandb.Image(os.path.join(output_dir, "jaccard.png")),
            "Aggregated Sensitivity Scores": wandb.Image(os.path.join(output_dir, "sensitivity.png")),
            "Aggregated Specificity Scores": wandb.Image(os.path.join(output_dir, "specificity.png")),
            "Aggregated Precision Scores": wandb.Image(os.path.join(output_dir, "precision.png")),
            "Aggregated HD95 Scores": wandb.Image(os.path.join(output_dir, "hd95.png")),
            "Aggregated Loss Across Folds": wandb.Image(os.path.join(output_dir, "loss.png"))
        })
        time.sleep(3)  
        
        # Summary Reporting with Try-Except
        try:
            # Summary Metrics
            avg_metrics = {}
            for region in ["ET", "TC", "WT"]:
                avg_metrics[region] = {}
                for metric in ["Dice", "Jaccard", "Sensitivity", "Precision", "Specificity", "HD95"]:
                    values = [fold_metrics[region][metric] for fold_metrics in results["metrics"]]
                    avg_metrics[region][metric] = np.mean(values)
            
            model_best_metrics = {k: v for k, v in best_metrics.items() if k.startswith(name)}
            avg_best_metric = np.mean(list(model_best_metrics.values()))
            wandb.summary.update({
                "Average Best WT Dice": avg_best_metric,
                "Fold Best WT Dice": model_best_metrics
            })
            time.sleep(2)
            logging.info(f"Best WT Dice Scores for {name}: {model_best_metrics}")
            logging.info(f"Average Best WT Dice: {avg_best_metric:.4f}")
            
            all_results[name] = {
                "metrics": avg_metrics,
                "best_metrics": model_best_metrics,
                "loss_history": results["loss_history"],
                "system_metrics": results["system_metrics"],
                "prediction_images": prediction_images,
                "sample_image": sample_path
            }
            
            # Summary Report (Fixed: 'Fold' as string consistently)
            summary_data = []
            for name in all_results:
                for fold in range(config["k_folds"]):
                    fold_key = f"{name}_fold{fold}"
                    system_metrics = all_results[name]["system_metrics"][fold]
                    summary_data.append([
                        name, str(fold), best_metrics.get(fold_key, 0),
                        all_results[name]["loss_history"][fold][-1] if all_results[name]["loss_history"][fold] else 0,
                        system_metrics["CPU Usage (%)"], system_metrics["Memory Used (GB)"],
                        system_metrics.get("GPU 0 Utilization (%)", 0), system_metrics.get("GPU 0 Memory Used (GB)", 0)
                    ])
                avg_system_metrics = {k: np.mean([sm[k] for sm in all_results[name]["system_metrics"]]) for k in all_results[name]["system_metrics"][0].keys()}
                summary_data.append([
                    f"{name} (Average)", "All", np.mean(list(all_results[name]["best_metrics"].values())),
                    np.mean([lh[-1] for lh in all_results[name]["loss_history"] if lh]),
                    avg_system_metrics["CPU Usage (%)"], avg_system_metrics["Memory Used (GB)"],
                    avg_system_metrics.get("GPU 0 Utilization (%)", 0), avg_system_metrics.get("GPU 0 Memory Used (GB)", 0)
                ])
            
            wandb.log({"Summary Report": wandb.Table(
                columns=["Model", "Fold", "Best WT Dice", "Final Training Loss", "CPU Usage (%)", "Memory Used (GB)", "GPU 0 Utilization (%)", "GPU 0 Memory Used (GB)"],
                data=summary_data
            )})
            
            logging.info("Summary Report:")
            logging.info(pd.DataFrame(summary_data, columns=["Model", "Fold", "Best WT Dice", "Final Training Loss", "CPU Usage (%)", "Memory Used (GB)", "GPU 0 Utilization (%)", "GPU 0 Memory Used (GB)"]).to_string())
        
        except Exception as e:
            logging.error(f"Error in summary reporting for {name}: {str(e)}")
            # Provide fallback values to continue execution
            avg_metrics = {"ET": {}, "TC": {}, "WT": {}}
            model_best_metrics = {k: v for k, v in best_metrics.items() if k.startswith(name)}
            avg_best_metric = np.mean(list(model_best_metrics.values())) if model_best_metrics else 0.0
            all_results[name] = {
                "metrics": avg_metrics,
                "best_metrics": model_best_metrics,
                "loss_history": results["loss_history"],
                "system_metrics": results["system_metrics"],
                "prediction_images": prediction_images,
                "sample_image": sample_path
            }
            wandb.summary.update({
                "Average Best WT Dice": avg_best_metric,
                "Fold Best WT Dice": model_best_metrics,
                "Summary Error": f"Failed to compute summary: {str(e)}"
            })
        
        time.sleep(2)
        wandb.finish()
        time.sleep(5)
        
        logging.info("-" * 50)
        logging.info("\n")
        
        logging.info(f"Completed {name}! W&B logs in {wandb_mode} mode.")
        
        logging.info("-" * 50)
        logging.info("\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation")
    parser.add_argument("--config", type=str, default="config/hyperparams.json", help="Path to config JSON")
    parser.add_argument("--model", type=str, default="both", choices=["StdUNet", "AttUNet", "both"], help="Model choice")
    args = parser.parse_args()
    main(args.config, args.model)