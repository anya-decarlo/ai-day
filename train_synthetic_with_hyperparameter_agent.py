#!/usr/bin/env python
# Training script for synthetic segmentation data with comprehensive RL hyperparameter agent

import os
import json
import time
import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score

import torch
import torch.nn as nn
from monai.apps import download_and_extract
from monai.config import print_config
from rl_hyperparameter_agent import RLHyperparameterAgent
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

def main():
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train a segmentation model on synthetic data with RL hyperparameter agent")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["Adam", "SGD", "RMSprop"], 
                        help="Optimizer type")
    parser.add_argument("--loss", type=str, default="Dice", choices=["Dice", "DiceCE", "Focal"], 
                        help="Loss function")
    parser.add_argument("--augmentations", type=str, default="None", 
                        choices=["None", "Flipping", "Rotation", "Scaling", "Brightness", "All"], 
                        help="Augmentation strategy")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for training")
    parser.add_argument("--gradient_clip", type=float, default=0.0, help="Gradient clipping value")
    parser.add_argument("--early_stopping", type=int, default=50, 
                        help="Stop training if no improvement for this many epochs (0 to disable)")
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.01,  # Start with a higher learning rate that likely needs adjustment
        help="Initial learning rate for optimizer"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of synthetic samples to generate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    
    # RL hyperparameter agent specific arguments
    parser.add_argument("--agent_learning_rate", type=float, default=0.3, 
                        help="Learning rate for the RL agent (not the model)")
    parser.add_argument("--agent_exploration_rate", type=float, default=0.5, 
                        help="Initial exploration rate for the RL agent")
    parser.add_argument("--agent_cooldown", type=int, default=0, 
                        help="Cooldown period after parameter changes")
    
    args = parser.parse_args()
    
    print_config()

    # Set deterministic training for reproducibility
    set_determinism(seed=0)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    results_dir = os.path.join(os.getcwd(), "results_hyperparameter_agent")
    os.makedirs(results_dir, exist_ok=True)

    # Save hyperparameters
    with open(os.path.join(results_dir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Use existing synthetic dataset
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

    # Define a fixed spatial size for all images based on patch size
    spatial_size = [args.patch_size, args.patch_size, args.patch_size]

    # Define augmentations based on the selected strategy
    augmentations = []
    if args.augmentations in ["Flipping", "All"]:
        augmentations.append(RandFlipd(keys=["image", "label"], prob=0.5))
    if args.augmentations in ["Rotation", "All"]:
        augmentations.append(RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5))
    if args.augmentations in ["Scaling", "All"]:
        augmentations.append(RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5))
        augmentations.append(RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5))
    if args.augmentations in ["Brightness", "All"]:
        augmentations.append(RandAdjustContrastd(keys=["image"], prob=0.5))

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
            *augmentations,  # Add selected augmentations
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
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=min(4, multiprocessing.cpu_count()), 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=min(4, multiprocessing.cpu_count()), 
        pin_memory=torch.cuda.is_available()
    )

    # Create model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # Background + 2 structures
        channels=(16, 32, 64, 128),  # Simpler architecture with fewer channels
        strides=(2, 2, 2),  # Fewer downsampling steps
        num_res_units=1,  # Fewer residual units
        dropout=args.dropout,
        norm="instance",  # Start with instance norm which performs better on this data
    ).to(device)

    # Print model summary
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Define loss function with default parameters (will be adjusted by the agent)
    if args.loss == "Dice":
        # Start with basic parameters
        loss_function = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,  # Start with including background (may not be optimal)
        )
    elif args.loss == "DiceCE":
        # Start with unbalanced parameters
        class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)  # Equal weights to start
        loss_function = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,  # Start with including background
            lambda_ce=0.5,  # Start with CE weighted less
            lambda_dice=1.5,  # Start with Dice weighted more
            weight=class_weights
        )
    elif args.loss == "Focal":
        # Start with basic parameters
        class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
        loss_function = FocalLoss(to_onehot_y=True, weight=class_weights, gamma=2.0)  # Default gamma

    # Create optimizer with initial learning rate
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate,
            momentum=0.9, 
            weight_decay=args.weight_decay,
            nesterov=True
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=0.9
        )

    # Initialize the RL hyperparameter agent with wider ranges for exploration
    agent = RLHyperparameterAgent(
        optimizer=optimizer,
        loss_function=loss_function,
        model=model,  # Pass the model to allow architecture changes
        base_lr=args.learning_rate,
        min_lr=1e-8,  # Allow for extremely small learning rates
        max_lr=0.5,    # Allow for very large learning rates
        initial_lambda_ce=0.5 if args.loss == "DiceCE" else None,  # Start unbalanced
        initial_lambda_dice=1.5 if args.loss == "DiceCE" else None,  # Start unbalanced
        initial_class_weights=[1.0, 1.0, 1.0],  # Equal class weights
        initial_threshold=0.5,  # Standard threshold
        initial_include_background=True,  # Start by including background
        initial_normalization_type='instance_norm',  # Start with instance norm
        patience=1,  # Shorter patience to allow more frequent adjustments
        cooldown=args.agent_cooldown,
        metrics_history_size=3,
        log_dir=results_dir,
        verbose=True,
        learning_rate=args.agent_learning_rate,
        exploration_rate=0.7,  # Very high exploration rate to try many options
        min_exploration_rate=0.15,  # Keep significant exploration even late in training
        exploration_decay=0.85,  # Slower decay to explore more
        # Add wider ranges for hyperparameter adjustments
        lambda_ce_range=(0.1, 5.0),
        lambda_dice_range=(0.1, 5.0),
        class_weight_range=(0.1, 10.0),
        threshold_range=(0.1, 0.9)
    )

    # Define metrics for evaluation
    dice_metric = DiceMetric(include_background=True, reduction="mean")  # Include background in metric

    # Define post-processing transforms
    post_pred = Compose([
        EnsureType(), 
        Activations(softmax=True), 
        AsDiscrete(threshold=agent.threshold)  # Use agent's threshold
    ])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])  # 3 classes

    # Create CSV file for logging metrics
    csv_filename = os.path.join(results_dir, "metrics.csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "dice_score", "lr", "lambda_ce", "lambda_dice", "class_weights", "threshold", "include_background", "normalization_type"])

    # Training and validation loops
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []

    # Early stopping variables
    no_improvement_count = 0
    early_stopping_patience = args.early_stopping

    total_start = time.time()

    # Training loop
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Get current hyperparameters from agent
        current_hyperparams = agent.get_current_hyperparameters()
        print(f"Current hyperparameters:")
        print(f"  Learning rate: {current_hyperparams['learning_rate']:.6f}")
        if current_hyperparams['lambda_ce'] is not None:
            print(f"  Lambda CE: {current_hyperparams['lambda_ce']:.2f}")
        if current_hyperparams['lambda_dice'] is not None:
            print(f"  Lambda Dice: {current_hyperparams['lambda_dice']:.2f}")
        print(f"  Class weights: {current_hyperparams['class_weights']}")
        print(f"  Threshold: {current_hyperparams['threshold']:.2f}")
        print(f"  Include background: {current_hyperparams['include_background']}")
        print(f"  Normalization type: {current_hyperparams['normalization_type']}")

        # Update post-processing threshold from agent
        post_pred = Compose([
            EnsureType(), 
            Activations(softmax=True), 
            AsDiscrete(threshold=current_hyperparams['threshold'])
        ])

        # Update loss function parameters if needed
        if hasattr(loss_function, 'include_background'):
            loss_function.include_background = current_hyperparams['include_background']

        if hasattr(loss_function, 'lambda_ce') and current_hyperparams['lambda_ce'] is not None:
            loss_function.lambda_ce = current_hyperparams['lambda_ce']

        if hasattr(loss_function, 'lambda_dice') and current_hyperparams['lambda_dice'] is not None:
            loss_function.lambda_dice = current_hyperparams['lambda_dice']

        # Update class weights if applicable
        if hasattr(loss_function, 'weight') and current_hyperparams['class_weights'] is not None:
            loss_function.weight = torch.tensor(current_hyperparams['class_weights'], device=device)

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
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

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

            # Log metrics
            with open(csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, 
                    epoch_loss, 
                    val_loss, 
                    metric, 
                    current_hyperparams['learning_rate'],
                    current_hyperparams['lambda_ce'] if current_hyperparams['lambda_ce'] is not None else "N/A",
                    current_hyperparams['lambda_dice'] if current_hyperparams['lambda_dice'] is not None else "N/A",
                    current_hyperparams['class_weights'],
                    current_hyperparams['threshold'],
                    current_hyperparams['include_background'],
                    current_hyperparams['normalization_type']
                ])

            # Update agent with new metrics
            agent_metrics = {
                'dice_score': metric,
                'val_loss': val_loss
            }

            # Let the agent decide on hyperparameter adjustments
            agent_action = agent.step(agent_metrics)

            if agent_action['hyperparameters_changed']:
                print(f"Agent adjusted {agent_action['parameter']} with action {agent_action['action']}")
                print(f"  From: {agent_action['old_value']}")
                print(f"  To: {agent_action['new_value']}")

            # Check for new best metric
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                no_improvement_count = 0  # Reset counter when we find a better model

                # Save best model
                torch.save(
                    model.state_dict(), 
                    os.path.join(results_dir, "best_model.pth")
                )
                print(f"Saved new best model with dice score: {best_metric:.4f}")
            else:
                no_improvement_count += 1  # Increment counter when no improvement

                # Check for early stopping
                if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                    break

            print(
                f"Current epoch: {epoch + 1}, current mean dice: {metric:.4f}, "
                f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

    # Training completed
    total_time = time.time() - total_start
    print(f"Training completed in {total_time:.4f} seconds")
    print(f"Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    # Plot learning curves
    agent.plot_learning_curves(save_path=os.path.join(results_dir, "learning_curves.png"))

    # Plot metrics
    plt.figure("Train/Val", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    x = list(range(1, len(epoch_loss_values) + 1))
    plt.plot(x, epoch_loss_values, label="Train")
    plt.plot(x, val_loss_values, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Mean Dice")
    plt.plot(x, metric_values)
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.savefig(os.path.join(results_dir, "metrics.png"))

    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))

    # Final evaluation on validation set
    model.eval()
    with torch.no_grad():
        # Create confusion matrix
        all_preds = []
        all_labels = []

        for val_data in val_loader:
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

            # Sliding window inference
            val_outputs = sliding_window_inference(
                val_inputs, spatial_size, 4, model, overlap=0.5
            )

            # Get predictions
            val_outputs = torch.softmax(val_outputs, dim=1)
            val_preds = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            # Flatten for confusion matrix
            all_preds.extend(val_preds.flatten())
            all_labels.extend(val_labels.flatten())

        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Save confusion matrix
        np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = ['Background', 'Structure 1', 'Structure 2']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))

    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()
