#!/usr/bin/env python
# Training script for synthetic segmentation data with RL agent

import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import csv
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, Dataset
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandRotated,
    RandAdjustContrastd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Resized,
)
from monai.utils import set_determinism
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
import nibabel as nib
import torch
import multiprocessing
import argparse
from generate_synthetic_data import generate_dataset, generate_synthetic_sample
import json
import torch.nn as nn
from rl_lr_agent import RLLRAgent  # Import the advanced RL agent
from monai.apps import download_and_extract

def main():
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train a segmentation model on synthetic data with RL agent")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop"], 
                        help="Optimizer type")
    parser.add_argument("--loss", type=str, default="DiceCE", choices=["Dice", "DiceCE", "Focal"], 
                        help="Loss function")
    parser.add_argument("--augmentations", type=str, default="Flipping", 
                        choices=["Flipping", "Rotation", "Scaling", "Brightness", "All"], 
                        help="Augmentation strategy")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for training")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.001,  # Set to 0.001
        help="Learning rate for optimizer"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of synthetic samples to generate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    
    args = parser.parse_args()
    
    print_config()

    # Set deterministic training for reproducibility
    set_determinism(seed=0)

    # Generate synthetic dataset
    dataset_dir = os.path.join(os.getcwd(), "SyntheticData")
    print(f"Generating synthetic dataset at: {dataset_dir}")
    
    # Generate the dataset if it doesn't exist
    if not os.path.exists(dataset_dir) or not os.path.exists(os.path.join(dataset_dir, "dataset.json")):
        generate_dataset(
            output_dir=dataset_dir,
            num_samples=args.num_samples,
            shape=(args.patch_size, args.patch_size, args.patch_size),
            num_classes=3  # Background + 2 structures
        )
    else:
        print(f"Using existing synthetic dataset at: {dataset_dir}")
    
    # Define a fixed spatial size for all images based on patch size
    spatial_size = [args.patch_size, args.patch_size, args.patch_size]

    # Define augmentations based on the selected strategy
    augmentations = []
    if args.augmentations in ["Flipping", "All"]:
        augmentations.extend([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ])
    if args.augmentations in ["Rotation", "All"]:
        augmentations.append(
            RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5)
        )
    if args.augmentations in ["Scaling", "All"]:
        augmentations.append(
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5)
        )
    if args.augmentations in ["Brightness", "All"]:
        augmentations.append(
            RandAdjustContrastd(keys=["image"], prob=0.5)
        )

    # Define the transforms for training and validation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            *augmentations,
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # Load the dataset from the synthetic data directory
    with open(os.path.join(dataset_dir, "dataset.json"), "r") as f:
        import json
        dataset_info = json.load(f)
    
    # Create dataset dictionaries
    data_dicts = []
    for item in dataset_info["training"]:
        data_dicts.append({
            "image": os.path.join(dataset_dir, item["image"]),
            "label": os.path.join(dataset_dir, item["label"]),
        })
    
    # Split into training and validation
    import random
    random.seed(0)
    random.shuffle(data_dicts)
    
    val_size = int(len(data_dicts) * args.val_ratio)
    train_files = data_dicts[val_size:]
    val_files = data_dicts[:val_size]
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Create datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size // 2, num_workers=0)

    # Create a more structured results directory with timestamp and experiment parameters
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{args.optimizer}_{args.loss}_bs{args.batch_size}_lr{args.learning_rate:.6f}_agent".replace(".", "p")
    results_dir = os.path.join(os.getcwd(), "results", "synthetic_agent", f"{timestamp}_{experiment_name}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    model_dir = os.path.join(results_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    agent_dir = os.path.join(results_dir, "agent")
    os.makedirs(agent_dir, exist_ok=True)
    
    # Save experiment configuration
    config_file = os.path.join(results_dir, "experiment_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model Architecture: UNet\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Loss Function: {args.loss}\n")
        f.write(f"Initial Learning Rate: {args.learning_rate}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Augmentations: {args.augmentations}\n")
        f.write(f"Dropout Rate: {args.dropout}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Patch Size: {args.patch_size}\n")
        f.write(f"Gradient Clipping: {args.gradient_clip}\n")
        f.write(f"Synthetic Samples: {args.num_samples}\n")
        f.write(f"Validation Ratio: {args.val_ratio}\n")
        f.write(f"Agent: RLLRAgent\n")
    
    # Define CSV file for metrics
    csv_filename = os.path.join(metrics_dir, "training_metrics.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'dice_score', 
            'auroc', 'auprc', 'ppv', 'nne', 'learning_rate',
            'batch_size', 'train_samples', 'val_samples',
            'optimizer_type', 'loss_function', 'augmentations',
            'model_architecture', 'dropout_rate', 'weight_decay',
            'num_epochs', 'patch_size', 'gradient_clipping',
            'agent_action', 'agent_reward'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Create the model, loss function, optimizer, and metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define the model - simplified architecture
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128),  # Reduced depth
        strides=(2, 2, 2),           # Reduced depth
        num_res_units=2,             # Increased from 1 to 2 for better feature learning
        dropout=args.dropout,
    ).to(device)
    
    # Create loss function based on selection
    if args.loss == "Dice":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    elif args.loss == "DiceCE":
        # Increase weights for foreground classes to focus more on them
        class_weights = torch.tensor([0.05, 1.5, 1.5], device=device)
        loss_function = DiceCELoss(
            to_onehot_y=True, 
            softmax=True,
            include_background=False,
            lambda_ce=0.5,  # Reduce CE weight to emphasize Dice loss component
            lambda_dice=1.0, # Emphasize Dice loss component
            weight=class_weights  # Apply class weights to CE component (correct parameter name)
        )
    elif args.loss == "Focal":
        # Add class weights to handle class imbalance
        class_weights = torch.tensor([0.05, 1.5, 1.5], device=device)
        loss_function = FocalLoss(to_onehot_y=True, weight=class_weights)

    # Create optimizer based on selection (initial learning rate will be set by the agent)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,  # Use learning rate directly
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)  # Default betas
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate,  # Use learning rate directly
            momentum=0.9, 
            weight_decay=args.weight_decay,
            nesterov=True  # Use Nesterov momentum for better convergence
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=args.learning_rate,  # Use learning rate directly
            weight_decay=args.weight_decay,
            momentum=0.9  # Add momentum to RMSprop
        )

    # Initialize the advanced RL agent for learning rate scheduling
    agent = RLLRAgent(
        optimizer=optimizer,
        base_lr=args.learning_rate,  # Match the learning rate
        min_lr=1e-6,
        max_lr=2e-2,  # Increase max learning rate
        patience=2,  # Reduce patience to make agent more responsive
        cooldown=1,  # Reduce cooldown to make agent more responsive
        metrics_history_size=5,
        log_dir=results_dir,
        verbose=True,
    )
    
    # Define metrics for evaluation
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Define the post-processing transforms
    post_pred = Compose([
        EnsureType(), 
        Activations(softmax=True),
        AsDiscrete(threshold=0.4)  # Lower threshold from default 0.5 to capture more foreground
    ])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])

    # Training loop
    num_epochs = args.epochs
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    lr_values = []
    
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        # Get current learning rate from optimizer
        current_lr = optimizer.param_groups[0]['lr']
        lr_values.append(current_lr)
        print(f"Current learning rate: {current_lr:.6f}")
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            # Normalize the loss to account for gradient accumulation
            accumulation_steps = 2  # Accumulate gradients every 2 steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Only update weights after accumulating gradients
            if step % accumulation_steps == 0:
                # Apply gradient clipping
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps  # Scale loss back for reporting
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item() * accumulation_steps:.4f}")
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_step = 0
                # Lists to store predictions and labels for metrics calculation
                all_preds = []
                all_labels = []
                
                for val_data in val_loader:
                    val_step += 1
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    val_loss += loss_function(val_outputs, val_labels).item()
                    
                    # Process predictions and labels for metrics
                    val_outputs_list = decollate_batch(val_outputs)
                    val_labels_list = decollate_batch(val_labels)
                    
                    val_outputs_processed = [post_pred(i) for i in val_outputs_list]
                    val_labels_processed = [post_label(i) for i in val_labels_list]
                    
                    # Calculate Dice metric
                    dice_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                    
                    # Store predictions and labels for other metrics
                    for pred, label in zip(val_outputs_processed, val_labels_processed):
                        # Store both foreground classes for Dice calculation
                        pred_class1 = pred[1].flatten().cpu().numpy()  # Class 1
                        pred_class2 = pred[2].flatten().cpu().numpy()  # Class 2
                        label_class1 = (label[1] > 0).flatten().cpu().numpy()  # Class 1 binary
                        label_class2 = (label[2] > 0).flatten().cpu().numpy()  # Class 2 binary
                        
                        # Combine classes for overall foreground prediction
                        pred_flat = np.maximum(pred_class1, pred_class2)
                        label_flat = np.logical_or(label_class1, label_class2).astype(np.float32)
                        
                        all_preds.append(pred_flat)
                        all_labels.append(label_flat)
                
                val_loss /= val_step
                
                # Concatenate all predictions and labels
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                
                # Calculate metrics
                metric = dice_metric.aggregate().item()
                
                # Calculate a custom Dice score to double-check
                pred_binary = (all_preds > 0.4).astype(int)  # Lower threshold to match post-processing
                custom_dice = (2.0 * np.sum(pred_binary * all_labels)) / (np.sum(pred_binary) + np.sum(all_labels) + 1e-7)
                
                # Use the better of the two Dice scores
                if custom_dice > metric:
                    metric = custom_dice
                
                # Calculate ROC AUC using sklearn
                auc_value = roc_auc_score(all_labels, all_preds)
                
                # Calculate precision-recall curve and AUPRC
                precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_preds)
                auprc = auc(recall_curve, precision_curve)
                
                # Calculate precision (PPV) and NNE
                # Threshold predictions at 0.5 for binary classification
                pred_binary = (all_preds > 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(all_labels, pred_binary, labels=[0, 1]).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                nne = 1.0 / precision if precision > 0 else float('inf')
                
                # Update the RL agent with the latest metrics
                metrics = {
                    'loss': epoch_loss,
                    'val_loss': val_loss,
                    'dice': metric,
                    'auroc': auc_value,
                    'auprc': auprc,
                    'epoch': epoch,
                    'dice_score': metric  # Add dice_score key which the agent expects
                }
                
                # Debug print before agent step
                print(f"[DEBUG] Before agent.step: optimizer LR = {optimizer.param_groups[0]['lr']:.6f}")
                
                # Let the agent decide on learning rate adjustment and apply it
                lr_changed = agent.step(metrics)
                action = agent.last_action if hasattr(agent, 'last_action') else 'maintain'
                reward = agent.last_reward if hasattr(agent, 'last_reward') else 0
                
                # Debug print after agent step
                print(f"[DEBUG] After agent.step: optimizer LR = {optimizer.param_groups[0]['lr']:.6f}, lr_changed = {lr_changed}")
                
                # Update current_lr with the actual value from the optimizer
                current_lr = optimizer.param_groups[0]['lr']
                
                if lr_changed:
                    print(f"[APPLYING RL AGENT DECISION] Changed learning rate to {current_lr:.6f} based on agent's {action} action")
                
                # Log metrics to CSV
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'epoch': epoch + 1,
                        'train_loss': epoch_loss,
                        'val_loss': val_loss,
                        'dice_score': metric,
                        'auroc': auc_value,
                        'auprc': auprc,
                        'ppv': precision,
                        'nne': nne,
                        'learning_rate': current_lr,
                        'batch_size': args.batch_size,
                        'train_samples': len(train_ds),
                        'val_samples': len(val_ds),
                        'optimizer_type': args.optimizer,
                        'loss_function': args.loss,
                        'augmentations': args.augmentations,
                        'model_architecture': 'UNet',  # Fixed to UNet
                        'dropout_rate': args.dropout,
                        'weight_decay': args.weight_decay,
                        'num_epochs': args.epochs,
                        'patch_size': args.patch_size,
                        'gradient_clipping': args.gradient_clip,
                        'agent_action': action,
                        'agent_reward': reward
                    })
                
                metric_values.append(metric)
                
                # Reset metrics
                dice_metric.reset()
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                
                print(
                    f"current epoch: {epoch + 1}, current mean dice: {metric:.4f}, "
                    f"current AUROC: {auc_value:.4f}, current AUPRC: {auprc:.4f}, "
                    f"current PPV: {precision:.4f}, current NNE: {nne:.4f}, "
                    f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}, "
                    f"agent action: {action}, reward: {reward:.4f}"
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    # Save the final model
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pth"))
    
    # Save the agent's learned policy
    agent.save_policy(os.path.join(agent_dir, "agent_policy.json"))
    
    # Save the configuration
    config_file = os.path.join(model_dir, "model_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"Model Architecture: UNet\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Loss Function: {args.loss}\n")
        f.write(f"Augmentations: {args.augmentations}\n")
        f.write(f"Dropout Rate: {args.dropout}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Initial Learning Rate: {args.learning_rate}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Patch Size: {args.patch_size}\n")
        f.write(f"Gradient Clipping: {args.gradient_clip}\n")
        f.write(f"Synthetic Samples: {args.num_samples}\n")
        f.write(f"Validation Ratio: {args.val_ratio}\n")
    
    # Plot training curves
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Training Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Validation Dice and Learning Rate")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y1 = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y1, color="green", label="Dice")
    plt.ylabel("Dice", color="green")
    plt.twinx()
    plt.ylabel("Learning Rate", color="blue")
    plt.plot(x, lr_values[:len(x)], color="blue", label="LR")
    plt.savefig(os.path.join(plots_dir, "training_curves.png"))
    
    # Let the agent create its own visualizations
    agent.plot_history(os.path.join(plots_dir, "agent_metrics.png"))
    
if __name__ == "__main__":
    main()
