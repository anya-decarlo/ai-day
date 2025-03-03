#!/usr/bin/env python
# Script to generate synthetic 3D medical imaging data with ground truth segmentations
# Compatible with MONAI framework

import os
import numpy as np
import nibabel as nib
import json
from tqdm import tqdm
import shutil
import argparse

def create_sphere(shape, center, radius, value=1):
    """Create a 3D sphere in a volume."""
    grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    grid = grid.reshape(3, -1).T
    dist = np.sqrt(np.sum((grid - center)**2, axis=1))
    mask = dist <= radius
    volume = np.zeros(shape[0] * shape[1] * shape[2])
    volume[mask] = value
    return volume.reshape(shape)

def create_ellipsoid(shape, center, radii, value=1):
    """Create a 3D ellipsoid in a volume."""
    grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    grid = grid.reshape(3, -1).T
    dist = np.sqrt(np.sum(((grid - center) / radii)**2, axis=1))
    mask = dist <= 1.0
    volume = np.zeros(shape[0] * shape[1] * shape[2])
    volume[mask] = value
    return volume.reshape(shape)

def create_cylinder(shape, center, radius, height, axis=2, value=1):
    """Create a 3D cylinder in a volume."""
    grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    grid = grid.reshape(3, -1).T
    
    # Calculate distance in the plane perpendicular to the specified axis
    axes = [0, 1, 2]
    axes.remove(axis)
    dist = np.sqrt(np.sum((grid[:, axes] - np.array([center[i] for i in axes]))**2, axis=1))
    
    # Check if within radius and height bounds
    mask = (dist <= radius) & (np.abs(grid[:, axis] - center[axis]) <= height/2)
    
    volume = np.zeros(shape[0] * shape[1] * shape[2])
    volume[mask] = value
    return volume.reshape(shape)

def add_noise(volume, noise_level=0.1):
    """Add Gaussian noise to a volume."""
    return volume + np.random.normal(0, noise_level, volume.shape)

def add_bias_field(volume, shape):
    """Add a smooth bias field to simulate intensity inhomogeneity."""
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    x = x / float(shape[0] - 1) - 0.5
    y = y / float(shape[1] - 1) - 0.5
    z = z / float(shape[2] - 1) - 0.5
    
    # Create a polynomial bias field
    bias = 1.0 + 0.3 * (x**2 + y**2 + z**2)
    
    return volume * bias

def generate_synthetic_sample(shape=(64, 64, 64), num_classes=3):
    """Generate a synthetic 3D image with corresponding segmentation mask."""
    # Initialize volumes
    image = np.zeros(shape)
    segmentation = np.zeros(shape)
    
    # Create background tissue with more complex intensity variation
    background = 0.2 + 0.15 * np.random.rand(*shape)  # More variation in background
    image = background
    
    # Add structures for each class (excluding background which is class 0)
    for class_idx in range(1, num_classes):
        # Randomly place structures with more variation
        center = np.array([
            shape[0]//2 + np.random.randint(-shape[0]//4, shape[0]//4),
            shape[1]//2 + np.random.randint(-shape[1]//4, shape[1]//4),
            shape[2]//2 + np.random.randint(-shape[2]//4, shape[2]//4)
        ])
        
        # Make structures more similar to each other
        if class_idx == 1:
            # Class 1: Use a sphere with variable radius
            radius = np.random.randint(5, 10)  # More variable radius
            structure_mask = create_sphere(shape, center, radius, value=1)
            
            # Less contrast from background
            intensity = 0.5 + np.random.rand() * 0.2  # More variable intensity
            image[structure_mask > 0] = intensity
            
        elif class_idx == 2:
            # Class 2: Use an ellipsoid with variable radii
            radii = np.array([
                np.random.randint(4, 9),
                np.random.randint(4, 9),
                np.random.randint(4, 9)
            ])
            structure_mask = create_ellipsoid(shape, center, radii, value=1)
            
            # Less contrast from background
            intensity = 0.3 + np.random.rand() * 0.2  # More variable intensity
            image[structure_mask > 0] = intensity
        
        # Add structure to segmentation with class index
        segmentation[structure_mask > 0] = class_idx
    
    # Add more noise
    noise_level = 0.1  # Increased noise
    image = add_noise(image, noise_level=noise_level)
    
    # Add stronger bias field
    image = add_bias_field(image, shape)
    
    # Normalize image to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image, segmentation

def save_nifti(data, filename):
    """Save a numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data, np.eye(4))
    nib.save(nifti_img, filename)

def generate_dataset(output_dir, num_samples=100, shape=(64, 64, 64), num_classes=3):
    """
    Generate a complete synthetic dataset with images and segmentation masks.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to generate
        shape: Shape of each volume (x, y, z)
        num_classes: Number of classes (including background)
        
    Returns:
        None (saves files to disk)
    """
    import os
    import json
    import nibabel as nib
    import numpy as np
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # Create dataset metadata
    dataset_info = {
        "name": "SyntheticSegmentationDataset",
        "description": "Synthetic dataset for medical image segmentation",
        "num_classes": num_classes,
        "shape": shape,
        "training": []
    }
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")
        
        # Generate synthetic sample
        image, segmentation = generate_synthetic_sample(shape=shape, num_classes=num_classes)
        
        # Save as NIfTI files
        image_path = os.path.join(output_dir, "images", f"image_{i:03d}.nii.gz")
        label_path = os.path.join(output_dir, "labels", f"label_{i:03d}.nii.gz")
        
        # Create NIfTI images with identity affine
        affine = np.eye(4)
        image_nii = nib.Nifti1Image(image, affine)
        label_nii = nib.Nifti1Image(segmentation, affine)
        
        # Save to disk
        nib.save(image_nii, image_path)
        nib.save(label_nii, label_path)
        
        # Add to dataset metadata
        dataset_info["training"].append({
            "image": os.path.relpath(image_path, output_dir),
            "label": os.path.relpath(label_path, output_dir)
        })
    
    # Save dataset metadata
    with open(os.path.join(output_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Generated {num_samples} synthetic samples in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic 3D medical imaging data')
    parser.add_argument('--output_dir', type=str, default='./SyntheticData', 
                        help='Output directory for the dataset')
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples to generate')
    parser.add_argument('--shape', type=int, nargs=3, default=[64, 64, 64], 
                        help='Shape of the 3D volumes (x, y, z)')
    parser.add_argument('--num_classes', type=int, default=3, 
                        help='Number of classes including background')
    
    args = parser.parse_args()
    
    # Generate the dataset
    generate_dataset(
        args.output_dir, 
        num_samples=args.num_samples, 
        shape=tuple(args.shape), 
        num_classes=args.num_classes
    )

if __name__ == "__main__":
    main()
