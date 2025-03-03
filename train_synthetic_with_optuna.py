#!/usr/bin/env python
# Training script for synthetic segmentation data with Optuna hyperparameter optimization

import os
import json
import time
import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score

import torch
import torch.nn as nn
from monai.apps import download_and_extract
from monai.config import print_config
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandAdjustContrastd,
    ScaleIntensityd,
    ToTensord,
    RandRotated,
    RandScaleIntensityd,
    RandSpatialCropd,
    Resized,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.handlers.utils import from_engine
import nibabel as nib
import argparse
from generate_synthetic_data import generate_dataset, generate_synthetic_sample

# Global variables to store results
BEST_DICE = 0.0
BEST_PARAMS = {}
RESULTS_DIR = ""

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization
    """
    global BEST_DICE, BEST_PARAMS, RESULTS_DIR
    
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    normalization_type = trial.suggest_categorical("normalization_type", ["batch", "instance"])
    include_background = trial.suggest_categorical("include_background", [True, False])
    
    # Loss function parameters
    lambda_ce = trial.suggest_float("lambda_ce", 0.1, 2.0) if args.loss == "DiceCE" else None
    lambda_dice = trial.suggest_float("lambda_dice", 0.1, 2.0) if args.loss == "DiceCE" else None
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 5.0) if args.loss == "Focal" else None
    
    # Class weights - balance between foreground classes
    class_weight_1 = trial.suggest_float("class_weight_1", 0.5, 5.0)
    class_weight_2 = trial.suggest_float("class_weight_2", 0.5, 5.0)
    class_weights = [1.0, class_weight_1, class_weight_2]
    
    # Threshold for segmentation
    threshold = trial.suggest_float("threshold", 0.1, 0.9)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    data_dir = os.path.join(os.getcwd(), "SyntheticData")
    
    # Load the dataset from the synthetic data directory
    with open(os.path.join(data_dir, "dataset.json"), "r") as f:
        dataset_info = json.load(f)
    
    # Create dataset dictionaries
    data_dicts = []
    for item in dataset_info["training"]:
        data_dicts.append({
            "image": os.path.join(data_dir, item["image"]),
            "label": os.path.join(data_dir, item["label"]),
        })
    
    # Split into training and validation
    val_idx = int(len(data_dicts) * 0.2)
    train_files = data_dicts[val_idx:]
    val_files = data_dicts[:val_idx]
    
    # Define a fixed spatial size for all images
    spatial_size = [args.patch_size, args.patch_size, args.patch_size]
    
    # Define transforms for training and validation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
            RandSpatialCropd(keys=["image", "label"], roi_size=spatial_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5),
            RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    
    # Create datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(4, multiprocessing.cpu_count()), 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=min(4, multiprocessing.cpu_count()), 
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    norm_type = Norm.BATCH if normalization_type == "batch" else Norm.INSTANCE
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # Background + 2 structures
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
        dropout=dropout_rate,
        norm=norm_type,
    ).to(device)
    
    # Define loss function
    if args.loss == "Dice":
        loss_function = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=include_background,
        )
    elif args.loss == "DiceCE":
        class_weights_tensor = torch.tensor(class_weights, device=device)
        loss_function = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=include_background,
            lambda_ce=lambda_ce,
            lambda_dice=lambda_dice,
            weight=class_weights_tensor
        )
    elif args.loss == "Focal":
        class_weights_tensor = torch.tensor(class_weights, device=device)
        loss_function = FocalLoss(
            to_onehot_y=True, 
            weight=class_weights_tensor, 
            gamma=focal_gamma
        )
    
    # Create optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate,
            momentum=momentum, 
            weight_decay=weight_decay,
            nesterov=True
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        )
    
    # Define metrics for evaluation
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    
    # Define post-processing transforms
    post_pred = Compose([
        EnsureType(), 
        Activations(softmax=True), 
        AsDiscrete(threshold=threshold)
    ])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])  # 3 classes
    
    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    
    # Training loop
    for epoch in range(args.epochs_per_trial):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            
            # Apply gradient clipping if enabled
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_step = 0
            
            for val_data in val_loader:
                val_step += 1
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                # Sliding window inference for validation
                val_outputs = sliding_window_inference(
                    val_inputs, spatial_size, 4, model, overlap=0.5
                )
                
                # Calculate validation loss
                val_loss_batch = loss_function(val_outputs, val_labels)
                val_loss += val_loss_batch.item()
                
                # Process outputs for metric calculation
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                
                # Compute metric
                dice_metric(y_pred=val_outputs, y=val_labels)
            
            # Aggregate validation metrics
            val_loss /= val_step
            val_loss_values.append(val_loss)
            
            # Calculate mean dice score
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            
            # Check for new best metric
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                
                # Save best model for this trial
                trial_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}")
                os.makedirs(trial_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(trial_dir, "best_model.pth"))
                
                # Update global best if this is the best overall
                if metric > BEST_DICE:
                    BEST_DICE = metric
                    BEST_PARAMS = {
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "momentum": momentum,
                        "dropout_rate": dropout_rate,
                        "batch_size": batch_size,
                        "normalization_type": normalization_type,
                        "include_background": include_background,
                        "lambda_ce": lambda_ce,
                        "lambda_dice": lambda_dice,
                        "focal_gamma": focal_gamma,
                        "class_weights": class_weights,
                        "threshold": threshold
                    }
                    
                    # Save best model overall
                    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
    
    # Report the best metric as the objective value
    return best_metric

def main():
    global RESULTS_DIR
    
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train a segmentation model on synthetic data with Optuna")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["Adam", "SGD", "RMSprop"], 
                        help="Optimizer type")
    parser.add_argument("--loss", type=str, default="Dice", choices=["Dice", "DiceCE", "Focal"], 
                        help="Loss function")
    parser.add_argument("--epochs_per_trial", type=int, default=30, 
                        help="Number of training epochs per trial")
    parser.add_argument("--n_trials", type=int, default=20, 
                        help="Number of Optuna trials")
    parser.add_argument("--patch_size", type=int, default=64, 
                        help="Patch size for training")
    parser.add_argument("--gradient_clip", type=float, default=0.0, 
                        help="Gradient clipping value")
    
    global args
    args = parser.parse_args()
    
    print_config()
    
    # Set deterministic training for reproducibility
    set_determinism(seed=0)
    
    # Create results directory
    RESULTS_DIR = os.path.join(os.getcwd(), "results_optuna")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save hyperparameters
    with open(os.path.join(RESULTS_DIR, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Create and run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    # Print study statistics
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save study results
    with open(os.path.join(RESULTS_DIR, "best_params.json"), "w") as f:
        json.dump(BEST_PARAMS, f, indent=4)
    
    # Create visualizations
    fig1 = plot_optimization_history(study)
    fig1.write_image(os.path.join(RESULTS_DIR, "optimization_history.png"))
    
    fig2 = plot_param_importances(study)
    fig2.write_image(os.path.join(RESULTS_DIR, "param_importances.png"))
    
    # Print final results
    print(f"Best Dice Score: {BEST_DICE:.4f}")
    print("Best Parameters:")
    for key, value in BEST_PARAMS.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
