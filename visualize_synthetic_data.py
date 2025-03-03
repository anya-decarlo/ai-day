#!/usr/bin/env python
# Script to generate and visualize synthetic 3D data

import os
import numpy as np
import matplotlib.pyplot as plt
from generate_synthetic_data import generate_synthetic_sample

def visualize_sample(image, segmentation, sample_idx=0):
    """Visualize a synthetic 3D image and its segmentation mask."""
    # Create output directory
    output_dir = "./synthetic_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find middle slices for each dimension
    slice_x = image.shape[0] // 2
    slice_y = image.shape[1] // 2
    slice_z = image.shape[2] // 2
    
    # Create figure with 3 rows (dimensions) and 3 columns (image, segmentation, overlay)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Sagittal view (YZ plane)
    axes[0, 0].imshow(image[slice_x, :, :], cmap="gray")
    axes[0, 0].set_title("Input Image (Sagittal)")
    
    # Use a colormap that distinguishes classes but is more medical-like
    seg_cmap = plt.cm.get_cmap('tab10', 3)  # Use tab10 with 3 distinct colors
    axes[0, 1].imshow(segmentation[slice_x, :, :], cmap=seg_cmap)
    axes[0, 1].set_title("Segmentation (Sagittal)")
    
    # Overlay
    axes[0, 2].imshow(image[slice_x, :, :], cmap="gray")
    mask = segmentation[slice_x, :, :] > 0
    colored_seg = np.zeros((*segmentation[slice_x, :, :].shape, 4))
    for i in range(1, 3):
        mask_i = segmentation[slice_x, :, :] == i
        if i == 1:  # First class - red with transparency
            colored_seg[mask_i] = [1, 0, 0, 0.5]
        elif i == 2:  # Second class - blue with transparency
            colored_seg[mask_i] = [0, 0, 1, 0.5]
    axes[0, 2].imshow(colored_seg)
    axes[0, 2].set_title("Overlay (Sagittal)")
    
    # Row 2: Coronal view (XZ plane)
    axes[1, 0].imshow(image[:, slice_y, :], cmap="gray")
    axes[1, 0].set_title("Input Image (Coronal)")
    
    axes[1, 1].imshow(segmentation[:, slice_y, :], cmap=seg_cmap)
    axes[1, 1].set_title("Segmentation (Coronal)")
    
    # Overlay
    axes[1, 2].imshow(image[:, slice_y, :], cmap="gray")
    mask = segmentation[:, slice_y, :] > 0
    colored_seg = np.zeros((*segmentation[:, slice_y, :].shape, 4))
    for i in range(1, 3):
        mask_i = segmentation[:, slice_y, :] == i
        if i == 1:  # First class - red with transparency
            colored_seg[mask_i] = [1, 0, 0, 0.5]
        elif i == 2:  # Second class - blue with transparency
            colored_seg[mask_i] = [0, 0, 1, 0.5]
    axes[1, 2].imshow(colored_seg)
    axes[1, 2].set_title("Overlay (Coronal)")
    
    # Row 3: Axial view (XY plane)
    axes[2, 0].imshow(image[:, :, slice_z], cmap="gray")
    axes[2, 0].set_title("Input Image (Axial)")
    
    axes[2, 1].imshow(segmentation[:, :, slice_z], cmap=seg_cmap)
    axes[2, 1].set_title("Segmentation (Axial)")
    
    # Overlay
    axes[2, 2].imshow(image[:, :, slice_z], cmap="gray")
    mask = segmentation[:, :, slice_z] > 0
    colored_seg = np.zeros((*segmentation[:, :, slice_z].shape, 4))
    for i in range(1, 3):
        mask_i = segmentation[:, :, slice_z] == i
        if i == 1:  # First class - red with transparency
            colored_seg[mask_i] = [1, 0, 0, 0.5]
        elif i == 2:  # Second class - blue with transparency
            colored_seg[mask_i] = [0, 0, 1, 0.5]
    axes[2, 2].imshow(colored_seg)
    axes[2, 2].set_title("Overlay (Axial)")
    
    # Remove axis ticks
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, f"synthetic_sample_{sample_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"Saved visualization to {save_path}")
    
    # Create a figure to show 3D structure through multiple slices
    num_slices = 5
    fig, axes = plt.subplots(3, num_slices, figsize=(15, 9))
    
    # Calculate slice positions
    z_slices = np.linspace(0, image.shape[2]-1, num_slices).astype(int)
    
    # Plot each slice
    for i, z in enumerate(z_slices):
        # Show image in grayscale
        axes[0, i].imshow(image[:, :, z], cmap="gray")
        
        # Create colored overlay for segmentation
        mask = segmentation[:, :, z] > 0
        colored_seg = np.zeros((*segmentation[:, :, z].shape, 4))
        for c in range(1, 3):
            mask_c = segmentation[:, :, z] == c
            if c == 1:  # First class - red with transparency
                colored_seg[mask_c] = [1, 0, 0, 0.5]
            elif c == 2:  # Second class - blue with transparency
                colored_seg[mask_c] = [0, 0, 1, 0.5]
        
        axes[0, i].imshow(colored_seg)
        axes[0, i].set_title(f"Axial Slice {z}")
        
        # Coronal slices
        y_slices = np.linspace(0, image.shape[1]-1, num_slices).astype(int)
        axes[1, i].imshow(image[:, y_slices[i], :], cmap="gray")
        
        # Create colored overlay for segmentation
        mask = segmentation[:, y_slices[i], :] > 0
        colored_seg = np.zeros((*segmentation[:, y_slices[i], :].shape, 4))
        for c in range(1, 3):
            mask_c = segmentation[:, y_slices[i], :] == c
            if c == 1:  # First class - red with transparency
                colored_seg[mask_c] = [1, 0, 0, 0.5]
            elif c == 2:  # Second class - blue with transparency
                colored_seg[mask_c] = [0, 0, 1, 0.5]
        
        axes[1, i].imshow(colored_seg)
        axes[1, i].set_title(f"Coronal Slice {y_slices[i]}")
        
        # Sagittal slices
        x_slices = np.linspace(0, image.shape[0]-1, num_slices).astype(int)
        axes[2, i].imshow(image[x_slices[i], :, :], cmap="gray")
        
        # Create colored overlay for segmentation
        mask = segmentation[x_slices[i], :, :] > 0
        colored_seg = np.zeros((*segmentation[x_slices[i], :, :].shape, 4))
        for c in range(1, 3):
            mask_c = segmentation[x_slices[i], :, :] == c
            if c == 1:  # First class - red with transparency
                colored_seg[mask_c] = [1, 0, 0, 0.5]
            elif c == 2:  # Second class - blue with transparency
                colored_seg[mask_c] = [0, 0, 1, 0.5]
        
        axes[2, i].imshow(colored_seg)
        axes[2, i].set_title(f"Sagittal Slice {x_slices[i]}")
    
    # Remove axis ticks
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save multi-slice figure
    save_path = os.path.join(output_dir, f"synthetic_slices_{sample_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"Saved multi-slice visualization to {save_path}")

def main():
    # Generate and visualize 3 samples
    for i in range(3):
        print(f"Generating sample {i+1}...")
        image, segmentation = generate_synthetic_sample(shape=(64, 64, 64), num_classes=3)
        
        # Print some statistics
        print(f"Image shape: {image.shape}")
        print(f"Image intensity range: [{image.min():.2f}, {image.max():.2f}]")
        print(f"Segmentation shape: {segmentation.shape}")
        print(f"Segmentation classes: {np.unique(segmentation)}")
        
        # Visualize the sample
        visualize_sample(image, segmentation, i+1)

if __name__ == "__main__":
    main()
