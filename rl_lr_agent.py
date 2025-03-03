#!/usr/bin/env python
# Reinforcement Learning Agent for Learning Rate Adjustment in MONAI Training

import numpy as np
import pandas as pd
import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from datetime import datetime

class RLLRAgent:
    """
    Reinforcement Learning Agent for dynamic learning rate adjustment
    
    This agent monitors training metrics and learns to adjust the learning rate
    to optimize model performance using reinforcement learning principles.
    """
    
    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        patience: int = 3,
        cooldown: int = 5,
        metrics_history_size: int = 10,
        log_dir: str = "results",
        verbose: bool = True,
        # RL-specific parameters
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
        min_exploration_rate: float = 0.05,
        exploration_decay: float = 0.95
    ):
        """
        Initialize the Reinforcement Learning LR Agent
        
        Args:
            optimizer: PyTorch optimizer to adjust (if None, only recommendations are made)
            base_lr: Initial learning rate
            min_lr: Minimum allowed learning rate
            max_lr: Maximum allowed learning rate
            patience: Number of epochs with no improvement before considering action
            cooldown: Number of epochs to wait after an LR change before making another
            metrics_history_size: Number of epochs to consider for trend analysis
            log_dir: Directory to save agent logs
            verbose: Whether to print agent actions
            learning_rate: Learning rate for the RL algorithm (not the model)
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate for epsilon-greedy policy
            min_exploration_rate: Minimum exploration rate
            exploration_decay: Rate at which exploration rate decays
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        
        # If optimizer is provided, use its learning rate as the current_lr
        if optimizer is not None:
            self.current_lr = optimizer.param_groups[0]['lr']
            if verbose:
                print(f"[RL AGENT] Initializing with optimizer's learning rate: {self.current_lr:.6f}")
        else:
            self.current_lr = base_lr
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.cooldown = cooldown
        self.metrics_history_size = metrics_history_size
        self.log_dir = log_dir
        self.verbose = verbose
        
        # RL-specific parameters
        self.learning_rate = learning_rate  # Alpha (learning rate for the agent)
        self.discount_factor = discount_factor  # Gamma (discount factor)
        self.exploration_rate = exploration_rate  # Epsilon (exploration rate)
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        
        # State tracking
        self.epochs_without_improvement = 0
        self.cooldown_counter = 0
        self.best_metric = -float('inf')
        self.history = []
        self.action_history = []
        self.current_epoch = 0
        self.last_action = None
        self.last_state = None
        self.last_reward = 0
        
        # Define possible actions: 
        # 0: maintain LR, 1: decrease LR, 2: increase LR
        self.actions = ['maintain', 'decrease', 'increase']
        
        # Q-table: maps (state, action) pairs to expected rewards
        # States are discretized trend types: 'improving', 'worsening', 'plateau', 'oscillating', 'mixed'
        self.states = ['improving', 'worsening', 'plateau', 'oscillating', 'mixed', 'insufficient_data']
        
        # Initialize Q-table with small random values to break ties
        self.q_table = {
            state: {action: np.random.uniform(0, 0.1) for action in self.actions}
            for state in self.states
        }
        
        # Initialize reward history for each action in each state
        self.reward_history = {
            state: {action: [] for action in self.actions}
            for state in self.states
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.join(log_dir, "agent_logs"), exist_ok=True)
        
        # Initialize agent log file
        self.log_file = os.path.join(log_dir, "agent_logs", "rl_lr_agent_log.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("epoch,state,action,reason,old_lr,new_lr,reward,q_value,exploration_rate,dice_score,val_loss\n")
                
        # Save initial Q-table
        self._save_q_table()

    def step(self, metrics: Dict[str, float]) -> bool:
        """
        Process new metrics and decide on an action using RL
        
        Args:
            metrics: Dictionary of metrics from the current epoch
            
        Returns:
            bool: True if learning rate was changed, False otherwise
        """
        # Debug print at start of step
        if self.verbose:
            print(f"[RL AGENT DEBUG] Starting step with metrics: {metrics}")
            if self.optimizer is not None:
                print(f"[RL AGENT DEBUG] Current optimizer LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Observe the environment and get the current state
        action_info = self.observe(metrics)
        
        # If we're in cooldown, don't change the learning rate
        if self.cooldown_counter > 0:
            if self.verbose:
                print(f"[RL AGENT DEBUG] In cooldown ({self.cooldown_counter} epochs left), not changing LR")
                # Add more detailed log message about what action would have been taken
                print(f"[RL AGENT INFO] If not in cooldown, would have {action_info['action']}d learning rate to {action_info['new_lr']:.6f}")
            return False
            
        # Get the selected action
        action = action_info['action']
        
        # Debug print action info
        if self.verbose:
            print(f"[RL AGENT DEBUG] Selected action: {action}, new_lr: {action_info['new_lr']:.6f}")
        
        # Apply the action if it's not 'maintain'
        if action != 'maintain' and self.optimizer is not None:
            # Update the learning rate in the optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = action_info['new_lr']
            
            # Make sure our internal state matches the optimizer's learning rate
            self.current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log the actual learning rate from the optimizer for verification
            actual_lr = self.optimizer.param_groups[0]['lr']
            if self.verbose:
                print(f"[RL AGENT DEBUG] Optimizer LR after update: {actual_lr:.6f}")
            
            return True
            
        return False
        
    def observe(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Process new metrics, update Q-values based on rewards, and select next action
        
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
        
        # Analyze the current state (trend)
        current_state = self._analyze_trend()
        
        # Calculate reward for the previous action if there was one
        if self.last_action is not None and self.last_state is not None:
            reward = self._calculate_reward(metrics)
            
            # Update Q-value for the previous state-action pair
            old_q_value = self.q_table[self.last_state][self.last_action]
            
            # Q-learning update rule
            # Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
            best_next_q = max(self.q_table[current_state].values())
            new_q_value = old_q_value + self.learning_rate * (
                reward + self.discount_factor * best_next_q - old_q_value
            )
            
            self.q_table[self.last_state][self.last_action] = new_q_value
            
            # Store reward for history
            self.reward_history[self.last_state][self.last_action].append(reward)
            
            # Update last reward for logging
            self.last_reward = reward
        
        # Default action is to maintain current learning rate
        action_info = {
            'action': 'maintain',
            'reason': 'default',
            'old_lr': self.current_lr,
            'new_lr': self.current_lr,
            'metrics': metrics,
            'state': current_state
        }
        
        # Check if we have a new best metric
        if dice_score > self.best_metric:
            self.best_metric = dice_score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Track cooldown but still calculate what action would be taken
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            action_info['reason'] = f'in_cooldown_{self.cooldown_counter}'
            
            # Still select an action even in cooldown (for logging purposes)
            selected_action = self._select_action(current_state)
            action_info['action'] = selected_action
            
            # Calculate what the new learning rate would be (but don't apply it yet)
            if selected_action == 'decrease':
                new_lr = max(self.current_lr * 0.5, self.min_lr)
                action_info['new_lr'] = new_lr
                action_info['reason'] = f'rl_policy_decrease'
                
                if self.verbose and new_lr != self.current_lr:
                    print(f"[RL LR AGENT] Epoch {self.current_epoch}: decrease learning rate from {self.current_lr:.6f} to {new_lr:.6f} due to {action_info['reason']} (state: {current_state}, Q-value: {self.q_table[current_state][selected_action]:.4f})")
                    
            elif selected_action == 'increase':
                new_lr = min(self.current_lr * 1.2, self.max_lr)
                action_info['new_lr'] = new_lr
                action_info['reason'] = f'rl_policy_increase'
                
                if self.verbose and new_lr != self.current_lr:
                    print(f"[RL LR AGENT] Epoch {self.current_epoch}: increase learning rate from {self.current_lr:.6f} to {new_lr:.6f} due to {action_info['reason']} (state: {current_state}, Q-value: {self.q_table[current_state][selected_action]:.4f})")
            
            self._log_action(action_info)
            return action_info
        
        # Select action based on current state using RL policy
        selected_action = self._select_action(current_state)
        action_info['action'] = selected_action
        
        # Apply the selected action
        if selected_action == 'decrease':
            new_lr = max(self.current_lr * 0.5, self.min_lr)
            action_info['new_lr'] = new_lr
            action_info['reason'] = f'rl_policy_decrease'
            
            # Only change if it's actually different
            if new_lr != self.current_lr:
                if self.verbose:
                    print(f"[RL LR AGENT] Epoch {self.current_epoch}: decrease learning rate from {self.current_lr:.6f} to {new_lr:.6f} due to {action_info['reason']} (state: {current_state}, Q-value: {self.q_table[current_state][selected_action]:.4f})")
                self.current_lr = new_lr
                self.cooldown_counter = self.cooldown
                
        elif selected_action == 'increase':
            new_lr = min(self.current_lr * 1.2, self.max_lr)
            action_info['new_lr'] = new_lr
            action_info['reason'] = f'rl_policy_increase'
            
            # Only change if it's actually different
            if new_lr != self.current_lr:
                if self.verbose:
                    print(f"[RL LR AGENT] Epoch {self.current_epoch}: increase learning rate from {self.current_lr:.6f} to {new_lr:.6f} due to {action_info['reason']} (state: {current_state}, Q-value: {self.q_table[current_state][selected_action]:.4f})")
                self.current_lr = new_lr
                self.cooldown_counter = self.cooldown
        
        # Store current state and action for next iteration
        self.last_state = current_state
        self.last_action = selected_action
        
        # Log the action
        self._log_action(action_info)
        self.action_history.append(action_info)
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
        # Save updated Q-table periodically
        if self.current_epoch % 5 == 0:
            self._save_q_table()
            
        return action_info
        
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calculate reward for the previous action
        
        The reward is based on improvement in dice score and reduction in validation loss
        
        Args:
            metrics: Current metrics
            
        Returns:
            reward: Numerical reward value
        """
        if len(self.history) < 2:
            return 0.0
            
        # Get current and previous metrics
        current_dice = metrics.get('dice_score', 0)
        current_loss = metrics.get('val_loss', float('inf'))
        
        prev_metrics = self.history[-2]
        prev_dice = prev_metrics.get('dice_score', 0)
        prev_loss = prev_metrics.get('val_loss', float('inf'))
        
        # Calculate changes
        dice_change = current_dice - prev_dice
        loss_change = prev_loss - current_loss  # Reversed so positive is good
        
        # Weight the changes (prioritize loss reduction over dice improvement)
        # Increased weight for loss reduction and decreased weight for dice improvement
        reward = (dice_change * 5.0) + (loss_change * 10.0)
        
        # Add a small positive reward for any action that doesn't make things worse
        # This encourages exploration even when metrics are stagnant
        if loss_change >= -0.01:  # If loss didn't increase significantly
            reward += 0.5
            
        # Bonus for new best dice score
        if current_dice > self.best_metric and self.current_epoch > 1:
            reward += 2.0
            
        # Penalty for very large learning rate changes that might destabilize training
        if self.last_action == 'increase' and self.current_lr > self.base_lr * 5:
            reward -= 1.0
            
        # Penalty for very small learning rates that might slow down training
        if self.last_action == 'decrease' and self.current_lr < self.base_lr * 0.1:
            reward -= 1.0
            
        return reward
        
    def _select_action(self, state: str) -> str:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state (trend type)
            
        Returns:
            action: Selected action
        """
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.actions)
            
        # Exploitation: best action based on Q-values
        # If multiple actions have the same Q-value, choose randomly among them
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [action for action, q_value in q_values.items() 
                       if q_value == max_q]
                       
        return np.random.choice(best_actions)

    def _analyze_trend(self) -> str:
        """
        Analyze the trend in validation metrics
        
        Returns:
            trend: String describing the trend ('improving', 'worsening', 'plateau', 'oscillating', 'mixed')
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
        # Get Q-value for the selected action in the current state
        state = action_info['state']
        action = action_info['action']
        q_value = self.q_table[state][action]
        
        with open(self.log_file, 'a') as f:
            f.write(f"{self.current_epoch},{state},{action},{action_info['reason']},"
                    f"{action_info['old_lr']},{action_info['new_lr']},"
                    f"{self.last_reward},{q_value},{self.exploration_rate},"
                    f"{action_info['metrics'].get('dice_score', 0)},"
                    f"{action_info['metrics'].get('val_loss', 0)}\n")
        
        if self.verbose and action_info['action'] != 'maintain':
            print(f"[RL LR AGENT] Epoch {self.current_epoch}: {action_info['action']} learning rate "
                  f"from {action_info['old_lr']:.6f} to {action_info['new_lr']:.6f} "
                  f"due to {action_info['reason']} (state: {state}, Q-value: {q_value:.4f})")
    
    def _save_q_table(self) -> None:
        """Save the Q-table to a JSON file"""
        q_table_path = os.path.join(self.log_dir, "agent_logs", "rl_q_table.json")
        
        # Convert Q-table to serializable format
        q_table_data = {
            "q_table": self.q_table,
            "epoch": self.current_epoch,
            "exploration_rate": self.exploration_rate,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(q_table_path, 'w') as f:
            json.dump(q_table_data, f, indent=2)
    
    def get_learning_rate(self) -> float:
        """Get the current learning rate recommended by the agent"""
        return self.current_lr
    
    def plot_history(self, save_path: Optional[str] = None) -> None:
        """Plot the history of learning rate changes, metrics, and Q-values"""
        if len(self.action_history) < 2:
            print("Not enough history to plot")
            return
        
        epochs = [info.get('metrics', {}).get('epoch', i+1) for i, info in enumerate(self.action_history)]
        learning_rates = [info['old_lr'] for info in self.action_history]
        dice_scores = [info.get('metrics', {}).get('dice_score', 0) for info in self.action_history]
        states = [info.get('state', 'unknown') for info in self.action_history]
        actions = [info['action'] for info in self.action_history]
        
        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot learning rate
        ax1.plot(epochs, learning_rates, 'b-', marker='o')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('Learning Rate Adjustments by RL Agent')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # Plot Dice score
        ax2.plot(epochs, dice_scores, 'g-', marker='x')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Dice Score Progression')
        ax2.grid(True)
        
        # Plot Q-values for each state-action pair over time
        # This is more complex as we need to extract Q-values for each epoch
        if len(self.states) > 0 and len(self.actions) > 0:
            # For simplicity, just plot Q-values for the actions that were actually taken
            q_values = []
            for i, (state, action) in enumerate(zip(states, actions)):
                if i < len(self.action_history):
                    # For earlier epochs, we might not have Q-values recorded
                    # Use the current Q-table as an approximation
                    q_values.append(self.q_table[state][action])
                
            ax3.plot(epochs[:len(q_values)], q_values, 'r-', marker='d')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Q-Value')
            ax3.set_title('Q-Values for Selected Actions')
            ax3.grid(True)
        
        # Mark points where LR was changed
        changes = [(i, info) for i, info in enumerate(self.action_history) if info['action'] != 'maintain']
        for i, info in changes:
            ax1.axvline(x=epochs[i], color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=epochs[i], color='r', linestyle='--', alpha=0.5)
            if len(self.states) > 0 and len(self.actions) > 0 and i < len(q_values):
                ax3.axvline(x=epochs[i], color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
            
        # Close the figure to free memory
        plt.close(fig)
        
        # Also plot the Q-table as a heatmap
        self._plot_q_table(save_path.replace('.png', '_q_table.png') if save_path else None)
    
    def _plot_q_table(self, save_path: Optional[str] = None) -> None:
        """Plot the Q-table as a heatmap"""
        # Convert Q-table to a numpy array for plotting
        q_array = np.zeros((len(self.states), len(self.actions)))
        for i, state in enumerate(self.states):
            for j, action in enumerate(self.actions):
                q_array[i, j] = self.q_table[state][action]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(q_array, cmap='viridis')
        plt.colorbar(label='Q-Value')
        plt.xticks(np.arange(len(self.actions)), self.actions, rotation=45)
        plt.yticks(np.arange(len(self.states)), self.states)
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.title('Q-Table Heatmap')
        
        # Add text annotations in the heatmap
        for i in range(len(self.states)):
            for j in range(len(self.actions)):
                plt.text(j, i, f'{q_array[i, j]:.2f}',
                        ha="center", va="center", color="w" if q_array[i, j] < 0.5 * q_array.max() else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved Q-table heatmap to {save_path}")
        else:
            plt.show()
            
        plt.close()


# Example usage (for testing)
if __name__ == "__main__":
    # Create a test agent
    agent = RLLRAgent(
        base_lr=1e-4,
        min_lr=1e-6,
        max_lr=1e-2,
        verbose=True
    )
    
    # Simulate some training epochs
    for epoch in range(1, 21):
        # Generate some fake metrics
        metrics = {
            'epoch': epoch,
            'dice_score': 0.7 + 0.01 * epoch + np.random.normal(0, 0.01),  # Gradually improving with noise
            'val_loss': 0.3 - 0.005 * epoch + np.random.normal(0, 0.01)    # Gradually decreasing with noise
        }
        
        # Let the agent observe and decide
        action_info = agent.observe(metrics)
        print(f"Epoch {epoch}: {action_info['action']} (state: {action_info['state']})")
    
    # Plot the history
    agent.plot_history()
