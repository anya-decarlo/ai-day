#!/usr/bin/env python
# Script to compare results between baseline hyperparameters and RL agent

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def load_metrics(file_path):
    """Load metrics from CSV file"""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return None
    return pd.read_csv(file_path)

def plot_comparison(baseline_metrics, agent_metrics, metric_name, title, ylabel, save_path):
    """Plot comparison between baseline and agent metrics"""
    plt.figure(figsize=(12, 6))
    
    if baseline_metrics is not None and metric_name in baseline_metrics.columns:
        plt.plot(baseline_metrics['epoch'], baseline_metrics[metric_name], 'b-', label='Baseline Hyperparameters')
    
    if agent_metrics is not None and metric_name in agent_metrics.columns:
        plt.plot(agent_metrics['epoch'], agent_metrics[metric_name], 'r-', label='RL Agent')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Create comparison directory
    os.makedirs('comparison_results', exist_ok=True)
    
    # Load metrics
    baseline_metrics = load_metrics('results_baseline/metrics.csv')
    agent_metrics = load_metrics('results_agent/metrics.csv')
    
    if baseline_metrics is None and agent_metrics is None:
        print("No metrics files found. Please run the training first.")
        return
    
    # Plot comparisons
    plot_comparison(
        baseline_metrics, agent_metrics, 
        'dice_score', 'Dice Score Comparison', 
        'Dice Score', 'comparison_results/dice_comparison.png'
    )
    
    plot_comparison(
        baseline_metrics, agent_metrics, 
        'val_loss', 'Validation Loss Comparison', 
        'Loss', 'comparison_results/val_loss_comparison.png'
    )
    
    plot_comparison(
        baseline_metrics, agent_metrics, 
        'train_loss', 'Training Loss Comparison', 
        'Loss', 'comparison_results/train_loss_comparison.png'
    )
    
    # Create summary report
    with open('comparison_results/summary.txt', 'w') as f:
        f.write("Comparison Summary: Baseline Hyperparameters vs. RL Agent\n")
        f.write("=" * 60 + "\n\n")
        
        # Write baseline hyperparameters summary
        f.write("Baseline Hyperparameters:\n")
        f.write("-" * 30 + "\n")
        f.write("Learning Rate: 0.3\n")
        f.write("Optimizer: SGD\n")
        f.write("Loss Function: Dice\n")
        f.write("Include Background: True (MONAI default)\n")
        f.write("Normalization: instance_norm (explicitly set to match agent starting point)\n")
        f.write("Class Weights: [1.0, 1.0, 1.0] (equal weights)\n")
        f.write("Augmentations: Flipping\n\n")
        
        if baseline_metrics is not None:
            best_baseline_dice = baseline_metrics['dice_score'].max()
            best_baseline_epoch = baseline_metrics.loc[baseline_metrics['dice_score'].idxmax(), 'epoch']
            f.write(f"Best Dice Score: {best_baseline_dice:.4f} (Epoch {best_baseline_epoch})\n")
            
            final_baseline_dice = baseline_metrics.iloc[-1]['dice_score']
            f.write(f"Final Dice Score: {final_baseline_dice:.4f}\n\n")
        
        # Write agent hyperparameters summary
        f.write("RL Agent Hyperparameters:\n")
        f.write("-" * 30 + "\n")
        f.write("Initial Learning Rate: 0.3\n")
        f.write("Initial Optimizer: SGD\n")
        f.write("Initial Loss Function: Dice\n")
        f.write("Initial Include Background: True (MONAI default)\n")
        f.write("Initial Normalization: instance_norm (explicitly set)\n")
        f.write("Initial Class Weights: [1.0, 1.0, 1.0] (equal weights)\n")
        f.write("Initial Augmentations: Flipping\n\n")
        
        if agent_metrics is not None:
            best_agent_dice = agent_metrics['dice_score'].max()
            best_agent_epoch = agent_metrics.loc[agent_metrics['dice_score'].idxmax(), 'epoch']
            f.write(f"Best Dice Score: {best_agent_dice:.4f} (Epoch {best_agent_epoch})\n")
            
            final_agent_dice = agent_metrics.iloc[-1]['dice_score']
            f.write(f"Final Dice Score: {final_agent_dice:.4f}\n\n")
            
            # Get final hyperparameters if available
            final_lr = agent_metrics.iloc[-1]['lr'] if 'lr' in agent_metrics.columns else "N/A"
            final_include_bg = agent_metrics.iloc[-1]['include_background'] if 'include_background' in agent_metrics.columns else "N/A"
            final_norm_type = agent_metrics.iloc[-1]['normalization_type'] if 'normalization_type' in agent_metrics.columns else "N/A"
            
            f.write(f"Final Hyperparameters Discovered by Agent:\n")
            f.write(f"  Learning Rate: {final_lr}\n")
            f.write(f"  Include Background: {final_include_bg}\n")
            f.write(f"  Normalization Type: {final_norm_type}\n\n")
        
        # Write comparison
        if baseline_metrics is not None and agent_metrics is not None:
            f.write("Comparison:\n")
            f.write("-" * 30 + "\n")
            dice_diff = best_agent_dice - best_baseline_dice
            f.write(f"Dice Score Difference: {dice_diff:.4f} ({dice_diff*100:.2f}%)\n")
            
            if dice_diff > 0:
                f.write("The RL Agent achieved better performance by discovering optimal hyperparameters!\n")
            elif dice_diff < 0:
                f.write("The Baseline Hyperparameters achieved better performance!\n")
            else:
                f.write("Both approaches achieved similar performance.\n")
    
    print("Comparison complete! Results saved to comparison_results/ directory.")

if __name__ == "__main__":
    main()
