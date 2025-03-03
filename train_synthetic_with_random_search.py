#!/usr/bin/env python
# Training script for synthetic segmentation data with random search hyperparameter optimization

import os
import json
import time
import csv
import random
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
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

def sample_random_hyperparameters(param_space):
    """
    Sample random hyperparameters from the parameter space
    """
    hyperparams = {}
    
    # Sample continuous parameters
    for param, (min_val, max_val) in param_space["continuous"].items():
        if "rate" in param or "decay" in param:  # Use log scale for rates
            hyperparams[param] = 10 ** random.uniform(np.log10(min_val), np.log10(max_val))
        else:
            hyperparams[param] = random.uniform(min_val, max_val)
    
    # Sample integer parameters
    for param, (min_val, max_val) in param_space["integer"].items():
        hyperparams[param] = random.randint(min_val, max_val)
    
    # Sample categorical parameters
    for param, options in param_space["categorical"].items():
        hyperparams[param] = random.choice(options)
    
    # Sample boolean parameters
    for param, prob in param_space["boolean"].items():
        hyperparams[param] = random.random() < prob
    
    # Sample special parameters
    if "class_weights" in param_space["special"]:
        # Generate random class weights with a bias towards emphasizing foreground classes
        weights = [1.0]  # Background weight
        for _ in range(param_space["special"]["class_weights"] - 1):
            weights.append(random.uniform(0.5, 3.0))
        hyperparams["class_weights"] = weights
    
    return hyperparams

def define_parameter_space(loss_type):
    """
    Define the parameter space for random search
    """
    param_space = {
        "continuous": {
            "learning_rate": (1e-4, 1e-1),
            "weight_decay": (1e-6, 1e-3),
            "momentum": (0.8, 0.99),
            "dropout_rate": (0.0, 0.5),
            "threshold": (0.3, 0.7),
        },
        "integer": {
            "batch_size": (2, 16),
        },
        "categorical": {
            "normalization_type": ["batch", "instance"],
        },
        "boolean": {
            "include_background": 0.5,  # 50% chance of True
        },
        "special": {
            "class_weights": 3,  # Number of classes
        }
    }
    
    # Add loss-specific parameters
    if loss_type == "DiceCE":
        param_space["continuous"]["lambda_ce"] = (0.1, 2.0)
        param_space["continuous"]["lambda_dice"] = (0.1, 2.0)
    elif loss_type == "Focal":
        param_space["continuous"]["focal_gamma"] = (0.5, 5.0)
    
    return param_space

def train_model(hyperparams, train_loader, val_loader, device, spatial_size, results_dir):
    """
    Train a model with the given hyperparameters and return the best validation metric
    """
    # Extract hyperparameters
    learning_rate = hyperparams["learning_rate"]
    weight_decay = hyperparams["weight_decay"]
    momentum = hyperparams["momentum"]
    dropout_rate = hyperparams["dropout_rate"]
    batch_size = hyperparams["batch_size"]
    normalization_type = hyperparams["normalization_type"]
    include_background = hyperparams["include_background"]
    lambda_ce = hyperparams.get("lambda_ce", None)
    lambda_dice = hyperparams.get("lambda_dice", None)
    focal_gamma = hyperparams.get("focal_gamma", None)
    class_weights = hyperparams["class_weights"]
    threshold = hyperparams["threshold"]
    
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
    for epoch in range(args.epochs_per_config):
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
                
                # Create a unique directory for this configuration
                config_hash = hash(frozenset(hyperparams.items())) % 10000
                config_dir = os.path.join(results_dir, f"config_{config_hash}")
                os.makedirs(config_dir, exist_ok=True)
                
                # Save best model for this configuration
                torch.save(model.state_dict(), os.path.join(config_dir, "best_model.pth"))
                
                # Save hyperparameters
                with open(os.path.join(config_dir, "hyperparams.json"), "w") as f:
                    json.dump(hyperparams, f, indent=4)
    
    # Return the best metric and the hyperparameters
    return best_metric, hyperparams

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a segmentation model on synthetic data with random search")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["Adam", "SGD", "RMSprop"], 
                        help="Optimizer type")
    parser.add_argument("--loss", type=str, default="Dice", choices=["Dice", "DiceCE", "Focal"], 
                        help="Loss function")
    parser.add_argument("--epochs_per_config", type=int, default=30, 
                        help="Number of training epochs per configuration")
    parser.add_argument("--patch_size", type=int, default=64, 
                        help="Patch size for training")
    parser.add_argument("--gradient_clip", type=float, default=0.0, 
                        help="Gradient clipping value")
    parser.add_argument("--num_trials", type=int, default=20, 
                        help="Number of random configurations to try")
    
    global args
    args = parser.parse_args()
    
    print_config()
    
    # Set deterministic training for reproducibility
    set_determinism(seed=0)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory
    results_dir = os.path.join(os.getcwd(), "results_random_search")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(os.path.join(results_dir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Load dataset
    data_dir = os.path.join(os.getcwd(), "SyntheticData")
    print(f"Using existing synthetic dataset at: {data_dir}")
    
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
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
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
    
    # Define parameter space for random search
    param_space = define_parameter_space(args.loss)
    
    # Create CSV file for logging results
    csv_filename = os.path.join(results_dir, "random_search_results.csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["trial_id", "dice_score"] + list(param_space["continuous"].keys()) + \
                list(param_space["integer"].keys()) + list(param_space["categorical"].keys()) + \
                list(param_space["boolean"].keys()) + ["class_weights"]
        writer.writerow(header)
    
    # Track best configuration
    best_metric = -1
    best_config = None
    
    # Run random search
    for trial_id in range(args.num_trials):
        print(f"\nTrial {trial_id + 1}/{args.num_trials}")
        
        # Sample random hyperparameters
        config = sample_random_hyperparameters(param_space)
        
        print("Configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        # Create data loaders with the current batch size
        batch_size = config["batch_size"]
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
        
        # Train and evaluate model with current configuration
        metric, hyperparams = train_model(config, train_loader, val_loader, device, spatial_size, results_dir)
        
        # Log results
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            row = [trial_id, metric]
            
            # Add all hyperparameters to the row
            for param_type in ["continuous", "integer", "categorical", "boolean"]:
                for param in param_space[param_type]:
                    row.append(config[param])
            
            # Add class weights
            row.append(str(config["class_weights"]))
            
            writer.writerow(row)
        
        print(f"Trial {trial_id + 1} achieved Dice score: {metric:.4f}")
        
        # Update best configuration
        if metric > best_metric:
            best_metric = metric
            best_config = config.copy()
            
            # Save best model
            config_hash = hash(frozenset(config.items())) % 10000
            config_dir = os.path.join(results_dir, f"config_{config_hash}")
            
            # Copy best model to results directory
            best_model_path = os.path.join(config_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                import shutil
                shutil.copy(best_model_path, os.path.join(results_dir, "best_model.pth"))
            
            # Save best configuration
            with open(os.path.join(results_dir, "best_config.json"), "w") as f:
                json.dump(best_config, f, indent=4)
    
    # Print final results
    print("\nRandom Search Complete!")
    print(f"Best Dice Score: {best_metric:.4f}")
    print("Best Configuration:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
