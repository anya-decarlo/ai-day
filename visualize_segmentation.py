#!/usr/bin/env python
# Visualization script for hippocampus segmentation results

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.apps import DecathlonDataset
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    Resized,
    EnsureTyped,
)
from monai.utils import set_determinism

# Set deterministic training for reproducibility
set_determinism(seed=0)

def main():
    # Define paths
    dataset_dir = "./Task04_Hippocampus"
    results_dir = "./results"
    vis_dir = os.path.join(results_dir, "visualizations", "segmentations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Define spatial size based on the model training
    spatial_size = (64, 64, 64)
    
    # Define transforms for validation
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
    
    # Load dataset using DecathlonDataset
    try:
        val_ds = DecathlonDataset(
            root_dir=dataset_dir,
            task="Task04_Hippocampus",
            transform=val_transforms,
            section="validation",
            download=False,
            cache_rate=0.0,
        )
        print(f"Validation dataset has {len(val_ds)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please make sure the dataset is available at {dataset_dir}")
        
        # Try loading from the dataset directory directly
        print("Attempting to load dataset from directory structure...")
        
        # Load the dataset manually
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as f:
            dataset_info = json.load(f)
        
        # Extract validation data
        val_files = []
        for item in dataset_info["training"]:
            if int(item["image"].split("_")[-1].split(".")[0]) % 5 == 0:  # Simple validation split
                val_files.append(
                    {
                        "image": os.path.join(dataset_dir, item["image"]),
                        "label": os.path.join(dataset_dir, item["label"]),
                    }
                )
        
        print(f"Found {len(val_files)} validation cases")
        
        # Create dataset
        val_ds = CacheDataset(
            data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4
        )
    
    # Create dataloader
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model with the same architecture as the training model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # Changed to match the trained model
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,  # Match the dropout rate from training
    ).to(device)
    
    # Load best model
    model_path = os.path.join(results_dir, "best_metric_model.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define post-processing transforms
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
    
    # Visualize results for a few samples
    num_samples = min(3, len(val_ds))  # Visualize just 3 samples
    
    print(f"Generating visualizations for {num_samples} samples...")
    
    for i in range(num_samples):
        val_data = val_ds[i]
        val_inputs = val_data["image"].unsqueeze(0).to(device)
        val_labels = val_data["label"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            val_outputs = model(val_inputs)
            raw_outputs = val_outputs.clone()
            val_outputs = post_pred(val_outputs)
        
        # Convert to numpy for visualization
        image = val_inputs.squeeze().cpu().numpy()
        label = val_labels.squeeze().cpu().numpy()
        prediction = val_outputs.squeeze().cpu().numpy()
        raw_probs = torch.softmax(raw_outputs, dim=1).squeeze().cpu().numpy()
        
        # Print shapes for debugging
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Raw probabilities shape: {raw_probs.shape}")
        
        # Take the first channel of prediction (or use argmax to get the class index)
        # Since we're using AsDiscrete(argmax=True), we need to get the class with highest probability
        if len(prediction.shape) == 4:
            prediction_vis = np.argmax(prediction, axis=0)
            print(f"Visualization prediction shape: {prediction_vis.shape}")
        else:
            prediction_vis = prediction
            
        if raw_probs.shape[0] == 3:  # If we have 3 classes (background + 2 hippocampus structures)
            prob_class1 = raw_probs[1]  # First hippocampus structure
            prob_class2 = raw_probs[2]  # Second hippocampus structure
            
            combined_probs = np.zeros((*prob_class1.shape, 3))  # RGB image
            combined_probs[..., 0] = prob_class1  # Red channel for class 1
            combined_probs[..., 1] = prob_class2  # Green channel for class 2
        
        # Find middle slices for each dimension
        slice_x = image.shape[0] // 2
        slice_y = image.shape[1] // 2
        slice_z = image.shape[2] // 2
        
        # Create figure with 3 rows (dimensions) and 3 columns (image, label, prediction)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Row 1: Sagittal view (YZ plane)
        axes[0, 0].imshow(image[slice_x, :, :], cmap="gray")
        axes[0, 0].set_title("Input Image (Sagittal)")
        axes[0, 1].imshow(label[slice_x, :, :], cmap="hot")
        axes[0, 1].set_title("Ground Truth (Sagittal)")
        axes[0, 2].imshow(prediction_vis[slice_x, :, :], cmap="hot")
        axes[0, 2].set_title("Prediction (Sagittal)")
        
        # Row 2: Coronal view (XZ plane)
        axes[1, 0].imshow(image[:, slice_y, :], cmap="gray")
        axes[1, 0].set_title("Input Image (Coronal)")
        axes[1, 1].imshow(label[:, slice_y, :], cmap="hot")
        axes[1, 1].set_title("Ground Truth (Coronal)")
        axes[1, 2].imshow(prediction_vis[:, slice_y, :], cmap="hot")
        axes[1, 2].set_title("Prediction (Coronal)")
        
        # Row 3: Axial view (XY plane)
        axes[2, 0].imshow(image[:, :, slice_z], cmap="gray")
        axes[2, 0].set_title("Input Image (Axial)")
        axes[2, 1].imshow(label[:, :, slice_z], cmap="hot")
        axes[2, 1].set_title("Ground Truth (Axial)")
        axes[2, 2].imshow(prediction_vis[:, :, slice_z], cmap="hot")
        axes[2, 2].set_title("Prediction (Axial)")
        
        # Remove axis ticks
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(vis_dir, f"segmentation_sample_{i+1}.png")
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"Saved visualization to {save_path}")
        
        if raw_probs.shape[0] == 3:  # If we have 3 classes (background + 2 hippocampus structures)
            # Create figure with 3 rows (dimensions) and 2 columns (class 1, class 2)
            fig, axes = plt.subplots(3, 2, figsize=(12, 15))
            
            # Row 1: Sagittal view (YZ plane)
            axes[0, 0].imshow(combined_probs[slice_x, :, :], cmap="hot")
            axes[0, 0].set_title("Class 1 & 2 Probabilities (Sagittal)")
            axes[0, 1].imshow(prob_class1[slice_x, :, :], cmap="hot")
            axes[0, 1].set_title("Class 1 Probability (Sagittal)")
            
            # Row 2: Coronal view (XZ plane)
            axes[1, 0].imshow(combined_probs[:, slice_y, :], cmap="hot")
            axes[1, 0].set_title("Class 1 & 2 Probabilities (Coronal)")
            axes[1, 1].imshow(prob_class1[:, slice_y, :], cmap="hot")
            axes[1, 1].set_title("Class 1 Probability (Coronal)")
            
            # Row 3: Axial view (XY plane)
            axes[2, 0].imshow(combined_probs[:, :, slice_z], cmap="hot")
            axes[2, 0].set_title("Class 1 & 2 Probabilities (Axial)")
            axes[2, 1].imshow(prob_class1[:, :, slice_z], cmap="hot")
            axes[2, 1].set_title("Class 1 Probability (Axial)")
            
            # Remove axis ticks
            for ax in axes.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(vis_dir, f"segmentation_probabilities_{i+1}.png")
            plt.savefig(save_path)
            plt.close(fig)
            
            print(f"Saved probability visualization to {save_path}")
    
    # Create overlay visualizations (image + segmentation)
    print("Generating overlay visualizations...")
    
    for i in range(num_samples):
        val_data = val_ds[i]
        val_inputs = val_data["image"].unsqueeze(0).to(device)
        val_labels = val_data["label"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            val_outputs = model(val_inputs)
            raw_outputs = val_outputs.clone()
            val_outputs = post_pred(val_outputs)
        
        # Convert to numpy for visualization
        image = val_inputs.squeeze().cpu().numpy()
        label = val_labels.squeeze().cpu().numpy()
        prediction = val_outputs.squeeze().cpu().numpy()
        raw_probs = torch.softmax(raw_outputs, dim=1).squeeze().cpu().numpy()
        
        # Print shapes for debugging
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Raw probabilities shape: {raw_probs.shape}")
        
        # Take the first channel of prediction (or use argmax to get the class index)
        # Since we're using AsDiscrete(argmax=True), we need to get the class with highest probability
        if len(prediction.shape) == 4:
            prediction_vis = np.argmax(prediction, axis=0)
            print(f"Visualization prediction shape: {prediction_vis.shape}")
        else:
            prediction_vis = prediction
            
        if raw_probs.shape[0] == 3:  # If we have 3 classes (background + 2 hippocampus structures)
            prob_class1 = raw_probs[1]  # First hippocampus structure
            prob_class2 = raw_probs[2]  # Second hippocampus structure
            
            combined_probs = np.zeros((*prob_class1.shape, 3))  # RGB image
            combined_probs[..., 0] = prob_class1  # Red channel for class 1
            combined_probs[..., 1] = prob_class2  # Green channel for class 2
        
        # Find middle slices for each dimension
        slice_x = image.shape[0] // 2
        slice_y = image.shape[1] // 2
        slice_z = image.shape[2] // 2
        
        # Create figure with 3 rows (dimensions) and 2 columns (ground truth overlay, prediction overlay)
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        
        # Row 1: Sagittal view (YZ plane)
        axes[0, 0].imshow(image[slice_x, :, :], cmap="gray")
        axes[0, 0].imshow(label[slice_x, :, :], cmap="hot", alpha=0.5)
        axes[0, 0].set_title("Ground Truth Overlay (Sagittal)")
        axes[0, 1].imshow(image[slice_x, :, :], cmap="gray")
        axes[0, 1].imshow(prediction_vis[slice_x, :, :], cmap="hot", alpha=0.5)
        axes[0, 1].set_title("Prediction Overlay (Sagittal)")
        
        # Row 2: Coronal view (XZ plane)
        axes[1, 0].imshow(image[:, slice_y, :], cmap="gray")
        axes[1, 0].imshow(label[:, slice_y, :], cmap="hot", alpha=0.5)
        axes[1, 0].set_title("Ground Truth Overlay (Coronal)")
        axes[1, 1].imshow(image[:, slice_y, :], cmap="gray")
        axes[1, 1].imshow(prediction_vis[:, slice_y, :], cmap="hot", alpha=0.5)
        axes[1, 1].set_title("Prediction Overlay (Coronal)")
        
        # Row 3: Axial view (XY plane)
        axes[2, 0].imshow(image[:, :, slice_z], cmap="gray")
        axes[2, 0].imshow(label[:, :, slice_z], cmap="hot", alpha=0.5)
        axes[2, 0].set_title("Ground Truth Overlay (Axial)")
        axes[2, 1].imshow(image[:, :, slice_z], cmap="gray")
        axes[2, 1].imshow(prediction_vis[:, :, slice_z], cmap="hot", alpha=0.5)
        axes[2, 1].set_title("Prediction Overlay (Axial)")
        
        # Remove axis ticks
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(vis_dir, f"segmentation_overlay_{i+1}.png")
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"Saved overlay visualization to {save_path}")
        
        if raw_probs.shape[0] == 3:  # If we have 3 classes (background + 2 hippocampus structures)
            # Create figure with 3 rows (dimensions) and 2 columns (class 1 overlay, class 2 overlay)
            fig, axes = plt.subplots(3, 2, figsize=(12, 15))
            
            # Row 1: Sagittal view (YZ plane)
            axes[0, 0].imshow(image[slice_x, :, :], cmap="gray")
            axes[0, 0].imshow(prob_class1[slice_x, :, :], cmap="hot", alpha=0.5)
            axes[0, 0].set_title("Class 1 Overlay (Sagittal)")
            axes[0, 1].imshow(image[slice_x, :, :], cmap="gray")
            axes[0, 1].imshow(prob_class2[slice_x, :, :], cmap="hot", alpha=0.5)
            axes[0, 1].set_title("Class 2 Overlay (Sagittal)")
            
            # Row 2: Coronal view (XZ plane)
            axes[1, 0].imshow(image[:, slice_y, :], cmap="gray")
            axes[1, 0].imshow(prob_class1[:, slice_y, :], cmap="hot", alpha=0.5)
            axes[1, 0].set_title("Class 1 Overlay (Coronal)")
            axes[1, 1].imshow(image[:, slice_y, :], cmap="gray")
            axes[1, 1].imshow(prob_class2[:, slice_y, :], cmap="hot", alpha=0.5)
            axes[1, 1].set_title("Class 2 Overlay (Coronal)")
            
            # Row 3: Axial view (XY plane)
            axes[2, 0].imshow(image[:, :, slice_z], cmap="gray")
            axes[2, 0].imshow(prob_class1[:, :, slice_z], cmap="hot", alpha=0.5)
            axes[2, 0].set_title("Class 1 Overlay (Axial)")
            axes[2, 1].imshow(image[:, :, slice_z], cmap="gray")
            axes[2, 1].imshow(prob_class2[:, :, slice_z], cmap="hot", alpha=0.5)
            axes[2, 1].set_title("Class 2 Overlay (Axial)")
            
            # Remove axis ticks
            for ax in axes.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(vis_dir, f"segmentation_class_overlay_{i+1}.png")
            plt.savefig(save_path)
            plt.close(fig)
            
            print(f"Saved class overlay visualization to {save_path}")
    
    print(f"All visualizations saved to {vis_dir}")

if __name__ == "__main__":
    main()
