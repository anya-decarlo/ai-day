#!/usr/bin/env python
# Learning Rate Agent for MONAI Training
# This agent dynamically adjusts the learning rate during training

import numpy as np
import pandas as pd
import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from datetime import datetime

class LRAgent:
    """
    Reinforcement Learning Agent for dynamic learning rate adjustment
    
    This agent monitors training metrics and adjusts the learning rate
    to optimize model performance.
    """
    
    def __init__(
        self,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        patience: int = 3,
        cooldown: int = 5,
        lr_factor: float = 0.5,
        lr_increase_factor: float = 1.2,
        metrics_history_size: int = 10,
        log_dir: str = "results",
        verbose: bool = True
    ):
        """
        Initialize the Learning Rate Agent
        
        Args:
            base_lr: Initial learning rate
            min_lr: Minimum allowed learning rate
            max_lr: Maximum allowed learning rate
            patience: Number of epochs with no improvement before reducing LR
            cooldown: Number of epochs to wait after an LR change before making another
            lr_factor: Factor to reduce learning rate by (must be < 1.0)
            lr_increase_factor: Factor to increase learning rate by (must be > 1.0)
            metrics_history_size: Number of epochs to consider for trend analysis
            log_dir: Directory to save agent logs
            verbose: Whether to print agent actions
        """
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.cooldown = cooldown
        self.lr_factor = lr_factor
        self.lr_increase_factor = lr_increase_factor
        self.metrics_history_size = metrics_history_size
        self.log_dir = log_dir
        self.verbose = verbose
        
        # State tracking
        self.epochs_without_improvement = 0
        self.cooldown_counter = 0
        self.best_metric = -float('inf')
        self.history = []
        self.action_history = []
        self.current_epoch = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.join(log_dir, "agent_logs"), exist_ok=True)
        
        # Initialize agent log file
        self.log_file = os.path.join(log_dir, "agent_logs", "lr_agent_log.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("epoch,action,reason,old_lr,new_lr,dice_score,val_loss,trend\n")
    
    def observe(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Process new metrics and decide on an action
        
        Args:
            metrics: Dictionary of metrics from the current epoch
            
        Returns:
            action_info: Dictionary with action details
        """
        self.current_epoch += 1
        
        # Extract key metrics
        dice_score = metrics.get('dice_score', 0)
        val_loss = metrics.get('val_loss', float('inf'))
        
        # Store metrics in history
        self.history.append(metrics)
        if len(self.history) > self.metrics_history_size:
            self.history = self.history[-self.metrics_history_size:]
        
        # Default action is to maintain current learning rate
        action_info = {
            'action': 'maintain',
            'reason': 'default',
            'old_lr': self.current_lr,
            'new_lr': self.current_lr,
            'metrics': metrics
        }
        
        # Skip decision if in cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            action_info['reason'] = f'in_cooldown_{self.cooldown_counter}'
            self._log_action(action_info)
            return action_info
        
        # Analyze metrics trend
        trend = self._analyze_trend()
        action_info['trend'] = trend
        
        # Check if we have a new best metric
        if dice_score > self.best_metric:
            self.best_metric = dice_score
            self.epochs_without_improvement = 0
            action_info['reason'] = 'new_best_metric'
        else:
            self.epochs_without_improvement += 1
            action_info['reason'] = f'no_improvement_{self.epochs_without_improvement}'
            
            # If we've waited long enough with no improvement, reduce learning rate
            if self.epochs_without_improvement >= self.patience:
                if trend == 'plateau':
                    new_lr = max(self.current_lr * self.lr_factor, self.min_lr)
                    if new_lr != self.current_lr:
                        action_info['action'] = 'decrease'
                        action_info['new_lr'] = new_lr
                        action_info['reason'] = f'plateau_after_{self.epochs_without_improvement}_epochs'
                        self.current_lr = new_lr
                        self.cooldown_counter = self.cooldown
                        self.epochs_without_improvement = 0
                elif trend == 'oscillating':
                    new_lr = max(self.current_lr * self.lr_factor * 0.8, self.min_lr)
                    if new_lr != self.current_lr:
                        action_info['action'] = 'decrease'
                        action_info['new_lr'] = new_lr
                        action_info['reason'] = f'oscillating_after_{self.epochs_without_improvement}_epochs'
                        self.current_lr = new_lr
                        self.cooldown_counter = self.cooldown
                        self.epochs_without_improvement = 0
        
        # If we're improving consistently, consider increasing learning rate
        if trend == 'improving' and self.current_epoch > 10:
            # Only increase if we've been consistently improving
            new_lr = min(self.current_lr * self.lr_increase_factor, self.max_lr)
            if new_lr != self.current_lr:
                action_info['action'] = 'increase'
                action_info['new_lr'] = new_lr
                action_info['reason'] = 'consistent_improvement'
                self.current_lr = new_lr
                self.cooldown_counter = self.cooldown
        
        # Log the action
        self._log_action(action_info)
        self.action_history.append(action_info)
        
        return action_info
    
    def _analyze_trend(self) -> str:
        """
        Analyze the trend in validation metrics
        
        Returns:
            trend: String describing the trend ('improving', 'worsening', 'plateau', 'oscillating')
        """
        if len(self.history) < 3:
            return 'insufficient_data'
        
        # Extract dice scores and losses from history
        dice_scores = [h.get('dice_score', 0) for h in self.history[-5:]]
        val_losses = [h.get('val_loss', float('inf')) for h in self.history[-5:]]
        
        # Calculate trends
        dice_diff = [dice_scores[i] - dice_scores[i-1] for i in range(1, len(dice_scores))]
        loss_diff = [val_losses[i-1] - val_losses[i] for i in range(1, len(val_losses))]  # Reversed for loss (lower is better)
        
        # Check for consistent improvement
        if all(d > 0.001 for d in dice_diff) and all(l > 0 for l in loss_diff):
            return 'improving'
        
        # Check for consistent worsening
        if all(d < -0.001 for d in dice_diff) and all(l < 0 for l in loss_diff):
            return 'worsening'
        
        # Check for plateau (very small changes)
        if all(abs(d) < 0.001 for d in dice_diff) and all(abs(l) < 0.001 for l in loss_diff):
            return 'plateau'
        
        # Check for oscillation
        signs_dice = [1 if d > 0 else -1 for d in dice_diff]
        if any(signs_dice[i] != signs_dice[i-1] for i in range(1, len(signs_dice))):
            return 'oscillating'
        
        return 'mixed'
    
    def _log_action(self, action_info: Dict[str, Any]) -> None:
        """Log the agent's action to a file"""
        with open(self.log_file, 'a') as f:
            f.write(f"{self.current_epoch},{action_info['action']},{action_info['reason']},"
                    f"{action_info['old_lr']},{action_info['new_lr']},"
                    f"{action_info['metrics'].get('dice_score', 0)},"
                    f"{action_info['metrics'].get('val_loss', 0)},"
                    f"{action_info.get('trend', 'unknown')}\n")
        
        if self.verbose and action_info['action'] != 'maintain':
            print(f"[LR AGENT] Epoch {self.current_epoch}: {action_info['action']} learning rate "
                  f"from {action_info['old_lr']:.6f} to {action_info['new_lr']:.6f} "
                  f"due to {action_info['reason']}")
    
    def get_learning_rate(self) -> float:
        """Get the current learning rate recommended by the agent"""
        return self.current_lr
    
    def plot_history(self, save_path: Optional[str] = None) -> None:
        """Plot the history of learning rate changes and metrics"""
        if len(self.action_history) < 2:
            print("Not enough history to plot")
            return
        
        epochs = [info.get('metrics', {}).get('epoch', i+1) for i, info in enumerate(self.action_history)]
        learning_rates = [info['old_lr'] for info in self.action_history]
        dice_scores = [info.get('metrics', {}).get('dice_score', 0) for info in self.action_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot learning rate
        ax1.plot(epochs, learning_rates, 'b-', marker='o')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('Learning Rate Adjustments by Agent')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # Plot Dice score
        ax2.plot(epochs, dice_scores, 'g-', marker='x')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Dice Score Progression')
        ax2.grid(True)
        
        # Mark points where LR was changed
        changes = [(i, info) for i, info in enumerate(self.action_history) if info['action'] != 'maintain']
        for i, info in changes:
            ax1.axvline(x=epochs[i], color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=epochs[i], color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.log_dir, "agent_logs", "lr_history.png"))
            print(f"Plot saved to {os.path.join(self.log_dir, 'agent_logs', 'lr_history.png')}")
        
        plt.close()


# Example usage (for testing)
if __name__ == "__main__":
    # Create a test agent
    agent = LRAgent(base_lr=1e-4, verbose=True)
    
    # Simulate some training epochs
    for i in range(20):
        # Simulate metrics
        if i < 5:
            dice = 0.5 + i * 0.02  # Improving
        elif i < 10:
            dice = 0.6  # Plateau
        elif i < 15:
            dice = 0.6 + (i-10) * 0.03  # Improving again
        else:
            dice = 0.75 - (i-15) * 0.01  # Worsening
            
        val_loss = 1.0 - dice/2
        
        metrics = {
            'epoch': i+1,
            'dice_score': dice,
            'val_loss': val_loss
        }
        
        # Get agent's action
        action = agent.observe(metrics)
        print(f"Epoch {i+1}: Dice={dice:.4f}, Val Loss={val_loss:.4f}, LR={agent.get_learning_rate():.6f}")
    
    # Plot the history
    agent.plot_history()
