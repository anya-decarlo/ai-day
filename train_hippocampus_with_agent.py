#!/usr/bin/env python
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import csv
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
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
import torch
import multiprocessing
import argparse
from datetime import datetime

# Import our learning rate agent
from rl_lr_agent import RLLRAgent

def main():
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train a segmentation model on the Hippocampus dataset with RL Agent")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "RMSprop"], 
                        help="Optimizer type")
    parser.add_argument("--loss", type=str, default="Dice", choices=["Dice", "DiceCE", "Focal"], 
                        help="Loss function")
    parser.add_argument("--augmentations", type=str, default="Flipping", 
                        choices=["Flipping", "Rotation", "Scaling", "Brightness", "All"], 
                        help="Augmentation strategy")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for training")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    # Agent-specific parameters
    parser.add_argument("--use_agent", action="store_true", help="Whether to use the RL agent")
    parser.add_argument("--agent_min_lr", type=float, default=1e-6, help="Minimum learning rate for agent")
    parser.add_argument("--agent_max_lr", type=float, default=1e-2, help="Maximum learning rate for agent")
    parser.add_argument("--agent_patience", type=int, default=3, help="Patience for agent before adjusting LR")
    parser.add_argument("--agent_cooldown", type=int, default=5, help="Cooldown period after agent adjusts LR")
    parser.add_argument("--agent_learning_rate", type=float, default=0.1, help="Learning rate for agent")
    parser.add_argument("--agent_discount_factor", type=float, default=0.9, help="Discount factor for agent")
    parser.add_argument("--agent_exploration_rate", type=float, default=0.2, help="Exploration rate for agent")
    parser.add_argument("--agent_min_exploration_rate", type=float, default=0.05, help="Minimum exploration rate for agent")
    parser.add_argument("--agent_exploration_decay", type=float, default=0.95, help="Exploration decay for agent")
    
    args = parser.parse_args()
    
    print_config()

    # Set deterministic training for reproducibility
    set_determinism(seed=0)

    # Define the dataset directory
    dataset_dir = os.path.join(os.getcwd(), "Task04_Hippocampus")
    print(f"Using dataset directory: {dataset_dir}")

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
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resize all images to a fixed size to ensure consistent dimensions
            Resized(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode=("trilinear", "nearest"),
            ),
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
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resize all images to a fixed size to ensure consistent dimensions
            Resized(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode=("trilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    
    # Create the dataset
    try:
        train_ds = DecathlonDataset(
            root_dir=dataset_dir,
            task="Task04_Hippocampus",
            transform=train_transforms,
            section="training",
            download=False,
            cache_rate=0.0,
        )
        val_ds = DecathlonDataset(
            root_dir=dataset_dir,
            task="Task04_Hippocampus",
            transform=val_transforms,
            section="validation",
            download=False,
            cache_rate=0.0,
        )
        print(f"Training dataset has {len(train_ds)} samples")
        print(f"Validation dataset has {len(val_ds)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Try loading from the dataset directory directly
        print("Attempting to load dataset from directory structure...")
        
        # Load the dataset manually
        import json
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as f:
            dataset_info = json.load(f)
        
        # Extract training and validation data
        train_files = []
        val_files = []
        
        # Use 80% for training, 20% for validation
        all_data = dataset_info["training"]
        split_idx = int(0.8 * len(all_data))
        
        for i, data in enumerate(all_data):
            img_path = os.path.join(dataset_dir, data["image"].replace("./", ""))
            label_path = os.path.join(dataset_dir, data["label"].replace("./", ""))
            
            if i < split_idx:
                train_files.append({"image": img_path, "label": label_path})
            else:
                val_files.append({"image": img_path, "label": label_path})
        
        from monai.data import CacheDataset
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)
        
        print(f"Manually loaded training dataset with {len(train_ds)} samples")
        print(f"Manually loaded validation dataset with {len(val_ds)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create UNet model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=args.dropout,
    ).to(device)
    
    # Select loss function
    if args.loss == "Dice":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    elif args.loss == "DiceCE":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    elif args.loss == "Focal":
        loss_function = FocalLoss(to_onehot_y=True)
    
    # Select optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize the RL learning rate agent if enabled
    if args.use_agent:
        lr_agent = RLLRAgent(
            optimizer=optimizer,
            base_lr=args.learning_rate,
            min_lr=args.agent_min_lr,
            max_lr=args.agent_max_lr,
            patience=args.agent_patience,
            cooldown=args.agent_cooldown,
            metrics_history_size=10,
            log_dir=os.path.join(os.getcwd(), "results"),
            verbose=True,
            learning_rate=args.agent_learning_rate,
            discount_factor=args.agent_discount_factor,
            exploration_rate=args.agent_exploration_rate,
            min_exploration_rate=args.agent_min_exploration_rate,
            exploration_decay=args.agent_exploration_decay
        )
        print("RL Agent initialized and enabled")
    else:
        lr_agent = None
        print("RL Agent disabled")
    
    # Define metrics for evaluation
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Define post-processing transforms
    post_pred = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
    
    # Setup directories for results
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Define CSV file for logging metrics
    csv_filename = os.path.join(results_dir, "training_metrics_with_agent.csv")
    
    # Define fieldnames for CSV
    fieldnames = [
        'epoch', 'train_loss', 'val_loss', 'dice_score', 'auroc', 'auprc', 
        'precision', 'nne', 'learning_rate', 'optimizer', 'loss_function', 
        'augmentations', 'dropout', 'weight_decay', 'agent_action'
    ]
    
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Training parameters
    num_epochs = args.epochs
    val_interval = 1  # Validate every epoch
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    agent_actions = []  # Track agent actions
    
    # Training loop
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            
            # Apply gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

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
                        # Flatten predictions and labels for ROC AUC calculation
                        # For multi-class, we'll focus on class 1 (foreground)
                        pred_flat = pred[1].flatten().cpu().numpy()  # Class 1
                        label_flat = (label[1] > 0).flatten().cpu().numpy()  # Class 1 binary
                        
                        all_preds.append(pred_flat)
                        all_labels.append(label_flat)
                
                val_loss /= val_step
                
                # Concatenate all predictions and labels
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                
                # Calculate metrics
                metric = dice_metric.aggregate().item()
                
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
                
                # Reset dice metric for next validation round
                dice_metric.reset()
                
                # Determine if this is the best model so far
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    # Save best model
                    torch.save(model.state_dict(), os.path.join(results_dir, "best_model_with_agent.pth"))
                    print("Saved new best model")
                
                print(
                    f"Current epoch: {epoch + 1}, Current mean dice: {metric:.4f}"
                    f"\nBest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                    f"\nAUROC: {auc_value:.4f}, AUPRC: {auprc:.4f}"
                    f"\nPrecision: {precision:.4f}, NNE: {nne:.4f}"
                )
                
                # Store metrics for plotting
                metric_values.append(metric)
                
                # Use the RL agent to adjust learning rate if enabled
                agent_action = "None"
                if args.use_agent and lr_agent is not None:
                    # Provide metrics to the agent
                    metrics_data = {
                        'epoch': epoch + 1,
                        'train_loss': epoch_loss,
                        'val_loss': val_loss,
                        'dice_score': metric,
                        'auroc': auc_value,
                        'auprc': auprc
                    }
                    
                    # Let the agent decide if it should adjust the learning rate
                    action_taken = lr_agent.step(metrics_data)
                    
                    if action_taken:
                        new_lr = optimizer.param_groups[0]['lr']
                        agent_action = f"Adjusted LR from {current_lr:.6f} to {new_lr:.6f}"
                        print(f"RL Agent: {agent_action}")
                    else:
                        agent_action = "No adjustment"
                        print("RL Agent: No adjustment needed")
                
                agent_actions.append(agent_action)
                
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
                        'precision': precision,
                        'nne': nne,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'optimizer': args.optimizer,
                        'loss_function': args.loss,
                        'augmentations': args.augmentations,
                        'dropout': args.dropout,
                        'weight_decay': args.weight_decay,
                        'agent_action': agent_action
                    })
    
    # Plot training metrics
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Training Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, label="Training Loss")
    plt.subplot(1, 2, 2)
    plt.title("Validation Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, label="Validation Dice")
    
    # Create visualization directories
    vis_dir = os.path.join(results_dir, "visualizations")
    metrics_plots_dir = os.path.join(vis_dir, "training_metrics")
    agent_plots_dir = os.path.join(vis_dir, "agent_plots")
    os.makedirs(metrics_plots_dir, exist_ok=True)
    os.makedirs(agent_plots_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save training metrics plot
    metrics_plot_path = os.path.join(metrics_plots_dir, f"run_{timestamp}_training_curves.png")
    plt.savefig(metrics_plot_path)
    print(f"Saved training metrics plot to {metrics_plot_path}")
    
    # Also save to the original location for backward compatibility
    plt.savefig(os.path.join(results_dir, "training_metrics_with_agent.png"))
    
    # Plot the RL agent's learning rate history if agent was used
    if args.use_agent and lr_agent is not None:
        agent_plot_path = os.path.join(agent_plots_dir, f"run_{timestamp}_rl_agent_history.png")
        lr_agent.plot_history(agent_plot_path)
        print(f"Saved agent history plot to {agent_plot_path}")
    
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    # Return the best metric value
    return best_metric

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
