#!/usr/bin/env python
"""
Plot training metrics from the CSV file generated during training.
This script creates various visualizations of the metrics tracked during model training.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_basic_metrics(df, output_dir):
    """Plot basic training and validation metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(df['Epoch'], df['Training_Loss'], 'b-', label='Training Loss')
    # Only include validation loss where it exists (not empty)
    val_loss_data = df[df['Validation_Loss'] != '']
    if not val_loss_data.empty:
        plt.plot(val_loss_data['Epoch'], val_loss_data['Validation_Loss'].astype(float), 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Dice Score
    plt.subplot(2, 2, 2)
    dice_data = df[df['Dice_Score'] != '']
    if not dice_data.empty:
        plt.plot(dice_data['Epoch'], dice_data['Dice_Score'].astype(float), 'g-')
        plt.title('Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(df['Epoch'], df['Learning_Rate'].astype(float), 'm-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Data Distribution
    plt.subplot(2, 2, 4)
    plt.plot(df['Epoch'], df['Data_Mean'].astype(float), 'c-', label='Mean')
    plt.plot(df['Epoch'], df['Data_Variance'].astype(float), 'y-', label='Variance')
    plt.title('Data Distribution Statistics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'basic_metrics.png'))
    plt.close()

def plot_advanced_metrics(df, output_dir):
    """Plot advanced metrics like AUROC, AUPRC, PPV, and NNE."""
    # Filter out rows where these metrics are empty
    advanced_df = df[(df['AUROC'] != '') & (df['AUPRC'] != '') & (df['PPV'] != '') & (df['NNE'] != '')]
    
    if advanced_df.empty:
        print("No advanced metrics data available for plotting")
        return
    
    # Convert to float
    for col in ['AUROC', 'AUPRC', 'PPV', 'NNE']:
        advanced_df[col] = advanced_df[col].astype(float)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: AUROC
    plt.subplot(2, 2, 1)
    plt.plot(advanced_df['Epoch'], advanced_df['AUROC'], 'b-')
    plt.title('Area Under ROC Curve (AUROC)')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: AUPRC
    plt.subplot(2, 2, 2)
    plt.plot(advanced_df['Epoch'], advanced_df['AUPRC'], 'r-')
    plt.title('Area Under Precision-Recall Curve (AUPRC)')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: PPV (Precision)
    plt.subplot(2, 2, 3)
    plt.plot(advanced_df['Epoch'], advanced_df['PPV'], 'g-')
    plt.title('Positive Predictive Value (Precision)')
    plt.xlabel('Epoch')
    plt.ylabel('PPV')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: NNE
    plt.subplot(2, 2, 4)
    plt.plot(advanced_df['Epoch'], advanced_df['NNE'], 'm-')
    plt.title('Number Needed to Evaluate (NNE)')
    plt.xlabel('Epoch')
    plt.ylabel('NNE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'advanced_metrics.png'))
    plt.close()

def plot_combined_metrics(df, output_dir):
    """Create a combined plot showing the relationship between different metrics."""
    # Filter out rows where these metrics are empty
    combined_df = df[(df['Dice_Score'] != '') & (df['AUROC'] != '') & (df['PPV'] != '')]
    
    if combined_df.empty:
        print("No combined metrics data available for plotting")
        return
    
    # Convert to float
    for col in ['Dice_Score', 'AUROC', 'AUPRC', 'PPV']:
        combined_df[col] = combined_df[col].astype(float)
    
    plt.figure(figsize=(10, 6))
    
    # Plot multiple metrics on the same graph
    plt.plot(combined_df['Epoch'], combined_df['Dice_Score'], 'b-', label='Dice Score')
    plt.plot(combined_df['Epoch'], combined_df['AUROC'], 'r-', label='AUROC')
    plt.plot(combined_df['Epoch'], combined_df['AUPRC'], 'g-', label='AUPRC')
    plt.plot(combined_df['Epoch'], combined_df['PPV'], 'm-', label='PPV')
    
    plt.title('Combined Performance Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from CSV file')
    parser.add_argument('--csv_file', type=str, default='Task04_Hippocampus/training_metrics.csv',
                        help='Path to the CSV file containing training metrics')
    parser.add_argument('--output_dir', type=str, default='Task04_Hippocampus',
                        help='Directory to save the plots')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the CSV file
    try:
        df = pd.read_csv(args.csv_file)
        print(f"Successfully loaded metrics from {args.csv_file}")
        print(f"Found {len(df)} epochs of data")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Generate plots
    plot_basic_metrics(df, output_dir)
    plot_advanced_metrics(df, output_dir)
    plot_combined_metrics(df, output_dir)
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
