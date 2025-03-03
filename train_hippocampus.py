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

def main():
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train a segmentation model on the Hippocampus dataset")
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
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
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

    # Load the dataset directly from the Task04_Hippocampus directory
    train_ds = DecathlonDataset(
        root_dir=os.path.dirname(dataset_dir),
        task="Task04_Hippocampus",
        transform=train_transforms,
        section="training",
        download=False,
        cache_rate=0.0,
    )

    val_ds = DecathlonDataset(
        root_dir=os.path.dirname(dataset_dir),
        task="Task04_Hippocampus",
        transform=val_transforms,
        section="validation",
        download=False,
        cache_rate=0.0,
        val_frac=0.2,
    )

    # Create data loaders with standard batch sizes
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size // 2, num_workers=0)

    # Create a directory to save the results
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create a CSV file to log metrics
    csv_filename = os.path.join(results_dir, "training_metrics.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'dice_score', 
            'auroc', 'auprc', 'ppv', 'nne', 'learning_rate',
            'batch_size', 'train_samples', 'val_samples',
            'optimizer_type', 'loss_function', 'augmentations',
            'model_architecture', 'dropout_rate', 'weight_decay',
            'num_epochs', 'patch_size', 'gradient_clipping'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Create the model, loss function, optimizer, and metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create UNet model with dropout parameter
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=args.dropout,
    ).to(device)

    # Create loss function based on selection
    if args.loss == "Dice":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    elif args.loss == "DiceCE":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    elif args.loss == "Focal":
        loss_function = FocalLoss(to_onehot_y=True)

    # Create optimizer based on selection
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Define the post-processing transforms
    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])

    # Training loop
    num_epochs = args.epochs
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

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
                        'learning_rate': optimizer.param_groups[0]['lr'],
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
                        'gradient_clipping': args.gradient_clip
                    })
                
                metric_values.append(metric)
                
                # Reset metrics
                dice_metric.reset()
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(results_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                
                print(
                    f"current epoch: {epoch + 1}, current mean dice: {metric:.4f}, "
                    f"current AUROC: {auc_value:.4f}, current AUPRC: {auprc:.4f}, "
                    f"current PPV: {precision:.4f}, current NNE: {nne:.4f}, "
                    f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    # Save the final model
    torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pth"))
    
    # Save the configuration
    config_file = os.path.join(results_dir, "model_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"Model Architecture: UNet\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Loss Function: {args.loss}\n")
        f.write(f"Augmentations: {args.augmentations}\n")
        f.write(f"Dropout Rate: {args.dropout}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Patch Size: {args.patch_size}\n")
        f.write(f"Gradient Clipping: {args.gradient_clip}\n")
        f.write(f"Number of Epochs: {args.epochs}\n")
        f.write(f"Best Dice Score: {best_metric:.4f} at epoch {best_metric_epoch}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
