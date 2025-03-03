#!/usr/bin/env python
# Reinforcement Learning Agent for Multiple Hyperparameter Adjustment in MONAI Training

import numpy as np
import pandas as pd
import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from datetime import datetime

class RLHyperparameterAgent:
    """
    Reinforcement Learning Agent for dynamic adjustment of multiple hyperparameters
    
    This agent monitors training metrics and learns to adjust various hyperparameters
    to optimize model performance using reinforcement learning principles.
    
    Hyperparameters that can be adjusted:
    - Learning rate
    - Loss function weights (lambda_ce, lambda_dice)
    - Class weights
    - Post-processing threshold
    - Include background
    - Normalization type
    - Weight decay
    - Momentum
    - Dropout rate
    - Augmentation intensity
    - Augmentation probability
    - Focal gamma
    - Batch size
    """
    
    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_function: Optional[Any] = None,
        model: Optional[Any] = None,
        # Initial hyperparameter values
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        initial_lambda_ce: Optional[float] = None,
        initial_lambda_dice: Optional[float] = None,
        initial_class_weights: Optional[List[float]] = None,
        initial_threshold: float = 0.5,
        initial_include_background: bool = True,
        initial_normalization_type: str = 'batch_norm',
        initial_weight_decay: float = 1e-5,
        initial_momentum: float = 0.9,
        initial_dropout_rate: float = 0.2,
        initial_augmentation_intensity: float = 0.5,
        initial_augmentation_probability: float = 0.5,
        initial_focal_gamma: float = 2.0,
        initial_batch_size: int = 16,
        # Hyperparameter ranges
        lambda_ce_range: Tuple[float, float] = (0.1, 2.0),
        lambda_dice_range: Tuple[float, float] = (0.1, 2.0),
        class_weight_range: Tuple[float, float] = (0.1, 5.0),
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        weight_decay_range: Tuple[float, float] = (1e-6, 1e-3),
        momentum_range: Tuple[float, float] = (0.5, 0.99),
        dropout_rate_range: Tuple[float, float] = (0.1, 0.5),
        augmentation_intensity_range: Tuple[float, float] = (0.1, 1.0),
        augmentation_probability_range: Tuple[float, float] = (0.1, 1.0),
        focal_gamma_range: Tuple[float, float] = (1.0, 5.0),
        batch_size_range: Tuple[int, int] = (8, 32),
        # Agent configuration
        patience: int = 3,
        cooldown: int = 3,
        metrics_history_size: int = 5,
        log_dir: str = "results",
        verbose: bool = True,
        # RL-specific parameters
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.5,  # Increased exploration rate
        min_exploration_rate: float = 0.05,
        exploration_decay: float = 0.95
    ):
        """
        Initialize the Reinforcement Learning Hyperparameter Agent
        
        Args:
            optimizer: PyTorch optimizer to adjust (if None, only recommendations are made)
            loss_function: Loss function to adjust (if None, only recommendations are made)
            model: PyTorch model to adjust (if None, only recommendations are made)
            base_lr: Initial learning rate
            min_lr: Minimum allowed learning rate
            max_lr: Maximum allowed learning rate
            initial_lambda_ce: Initial weight for cross-entropy loss
            initial_lambda_dice: Initial weight for dice loss
            initial_class_weights: Initial class weights [background, class1, class2]
            initial_threshold: Initial post-processing threshold
            initial_include_background: Initial include background flag
            initial_normalization_type: Initial normalization type
            lambda_ce_range: Range for lambda_ce (min, max)
            lambda_dice_range: Range for lambda_dice (min, max)
            class_weight_range: Range for class weights (min, max)
            threshold_range: Range for threshold (min, max)
            patience: Number of epochs with no improvement before considering action
            cooldown: Number of epochs to wait after a parameter change before making another
            metrics_history_size: Number of epochs to consider for trend analysis
            log_dir: Directory to save agent logs
            verbose: Whether to print agent actions
            learning_rate: Learning rate for the RL algorithm (not the model)
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate for epsilon-greedy policy
            min_exploration_rate: Minimum exploration rate
            exploration_decay: Rate at which exploration rate decays
        """
        # Store references to model components
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model
        
        # Initialize hyperparameter values
        self.base_lr = base_lr
        self.learning_rate = base_lr
        
        # Initialize loss weights
        self.lambda_ce = initial_lambda_ce
        self.lambda_dice = initial_lambda_dice
        
        self.class_weights = initial_class_weights or [1.0, 1.0, 1.0]
        self.threshold = initial_threshold
        
        self.include_background = initial_include_background
        self.normalization_type = initial_normalization_type
        
        self.weight_decay = initial_weight_decay
        self.momentum = initial_momentum
        self.dropout_rate = initial_dropout_rate
        self.augmentation_intensity = initial_augmentation_intensity
        self.augmentation_probability = initial_augmentation_probability
        self.focal_gamma = initial_focal_gamma
        self.batch_size = initial_batch_size
        
        # Store hyperparameter ranges
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lambda_ce_range = lambda_ce_range
        self.lambda_dice_range = lambda_dice_range
        self.class_weight_range = class_weight_range
        self.threshold_range = threshold_range
        self.weight_decay_range = weight_decay_range
        self.momentum_range = momentum_range
        self.dropout_rate_range = dropout_rate_range
        self.augmentation_intensity_range = augmentation_intensity_range
        self.augmentation_probability_range = augmentation_probability_range
        self.focal_gamma_range = focal_gamma_range
        self.batch_size_range = batch_size_range
        
        # Agent configuration
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
        self.metrics_history = []
        self.action_history = []
        self.current_epoch = 0
        self.last_action = None
        self.last_state = None
        self.last_reward = 0
        
        # Define possible hyperparameters to adjust
        self.hyperparameters = [
            'learning_rate',
            'lambda_ce',
            'lambda_dice',
            'class_weights',
            'threshold',
            'include_background',
            'normalization_type',
            'weight_decay',
            'momentum',
            'dropout_rate',
            'augmentation_intensity',
            'augmentation_probability',
            'focal_gamma',
            'batch_size'
        ]
        
        # Define possible actions for each hyperparameter
        self.actions = {
            'learning_rate': ['maintain', 'decrease', 'increase'],
            'lambda_ce': ['maintain', 'decrease', 'increase'],
            'lambda_dice': ['maintain', 'decrease', 'increase'],
            'class_weights': ['maintain', 'emphasize_foreground', 'balance'],
            'threshold': ['maintain', 'decrease', 'increase'],
            'include_background': ['maintain', 'include', 'exclude'],
            'normalization_type': ['maintain', 'use_batch_norm', 'use_instance_norm'],
            'weight_decay': ['maintain', 'decrease', 'increase'],
            'momentum': ['maintain', 'decrease', 'increase'],
            'dropout_rate': ['maintain', 'decrease', 'increase'],
            'augmentation_intensity': ['maintain', 'decrease', 'increase'],
            'augmentation_probability': ['maintain', 'decrease', 'increase'],
            'focal_gamma': ['maintain', 'decrease', 'increase'],
            'batch_size': ['maintain', 'decrease', 'increase']
        }
        
        # Define states (trends in performance)
        self.states = ['improving', 'worsening', 'plateau', 'oscillating', 'mixed', 'insufficient_data', 
                       'overfitting', 'underfitting', 'unstable', 'converging', 'diverging']
        
        # Initialize Q-table with small random values to break ties
        self.q_table = {}
        for state in self.states:
            self.q_table[state] = {}
            for param in self.hyperparameters:
                self.q_table[state][param] = {}
                for action in self.actions[param]:
                    self.q_table[state][param][action] = np.random.uniform(0, 0.1)
        
        # Initialize reward history
        self.reward_history = {}
        for state in self.states:
            self.reward_history[state] = {}
            for param in self.hyperparameters:
                self.reward_history[state][param] = {}
                for action in self.actions[param]:
                    self.reward_history[state][param][action] = []
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.join(log_dir, "agent_logs"), exist_ok=True)
        
        # Initialize agent log file
        self.log_file = os.path.join(log_dir, "agent_logs", "rl_hyperparameter_agent_log.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("epoch,state,parameter,action,reason,old_value,new_value,reward,q_value,dice_score,val_loss\n")
                
        # Save initial Q-table
        self._save_q_table()
        
        if self.verbose:
            print(f"[RL AGENT] Initialized with hyperparameters:")
            print(f"  - Learning rate: {self.learning_rate:.6f}")
            if self.lambda_ce is not None:
                print(f"  - Lambda CE: {self.lambda_ce:.2f}")
            if self.lambda_dice is not None:
                print(f"  - Lambda Dice: {self.lambda_dice:.2f}")
            print(f"  - Class weights: {self.class_weights}")
            print(f"  - Threshold: {self.threshold:.2f}")
            print(f"  - Include background: {self.include_background}")
            print(f"  - Normalization type: {self.normalization_type}")
            print(f"  - Weight decay: {self.weight_decay:.6f}")
            print(f"  - Momentum: {self.momentum:.2f}")
            print(f"  - Dropout rate: {self.dropout_rate:.2f}")
            print(f"  - Augmentation intensity: {self.augmentation_intensity:.2f}")
            print(f"  - Augmentation probability: {self.augmentation_probability:.2f}")
            print(f"  - Focal gamma: {self.focal_gamma:.2f}")
            print(f"  - Batch size: {self.batch_size}")
            
    def step(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Process new metrics and decide on hyperparameter adjustments using RL
        
        Args:
            metrics: Dictionary of metrics from the current epoch
            
        Returns:
            dict: Information about actions taken and updated hyperparameters
        """
        if self.verbose:
            print(f"[RL AGENT DEBUG] Starting step with metrics: {metrics}")
            if self.optimizer:
                print(f"[RL AGENT DEBUG] Current optimizer LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        self.current_epoch += 1
        self.metrics_history.append(metrics)
        
        # Initialize action info
        action_info = {
            'epoch': self.current_epoch,
            'parameter': None,
            'action': None,
            'old_value': None,
            'new_value': None,
            'reason': None,
            'hyperparameters_changed': False,
            'reward': 0,
            'state': self._analyze_trend()  # Add state from the beginning
        }
        
        # HIGHEST PRIORITY: If loss is stuck at 1.0 and background is excluded, immediately include background
        if self.metrics_history and len(self.metrics_history) > 3:
            recent_losses = [m.get('val_loss', 0) for m in self.metrics_history[-3:]]
            if all(abs(loss - 1.0) < 0.001 for loss in recent_losses) and not self.include_background:
                if self.verbose:
                    print(f"[RL AGENT] Epoch {self.current_epoch}: EMERGENCY - Loss stuck at 1.0, including background")
                
                # Update action info
                action_info['parameter'] = 'include_background'
                action_info['action'] = 'include'
                action_info['old_value'] = False
                action_info['new_value'] = True
                action_info['reason'] = 'emergency_include_background'
                action_info['hyperparameters_changed'] = True
                
                # Apply the change
                self.include_background = True
                
                # Store current state and action for next reward calculation
                self.last_state = 'plateau'
                self.last_action = ('include_background', 'include')  # Set as tuple (param, action)
                
                return action_info
        
        # Calculate reward for previous action if there was one
        if self.last_action is not None and self.last_state is not None:
            reward = self._calculate_reward(metrics)
            
            # Update Q-value for the previous state-action pair
            param, action = self.last_action
            old_q_value = self.q_table[self.last_state][param][action]
            
            # Q-learning update rule
            # Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
            # Find best next action for each parameter
            best_next_q = 0
            for p in self.hyperparameters:
                best_next_q += max(self.q_table[self.last_state][p].values())
            best_next_q /= len(self.hyperparameters)  # Average across parameters
            
            # Update Q-value
            new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * best_next_q - old_q_value)
            self.q_table[self.last_state][param][action] = new_q_value
            
            # Store reward in history for this state-parameter-action combination
            if param not in self.reward_history[self.last_state]:
                self.reward_history[self.last_state][param] = {}
            if action not in self.reward_history[self.last_state][param]:
                self.reward_history[self.last_state][param][action] = []
            
            self.reward_history[self.last_state][param][action].append(reward)
            
            # Limit history size to prevent memory issues
            if len(self.reward_history[self.last_state][param][action]) > 20:
                self.reward_history[self.last_state][param][action] = self.reward_history[self.last_state][param][action][-20:]
            
            # Update last reward for logging
            self.last_reward = reward
        
        # Default action info
        action_info = {
            'parameter': None,
            'action': 'maintain',
            'reason': 'default',
            'old_value': None,
            'new_value': None,
            'metrics': metrics,
            'state': action_info['state'],
            'hyperparameters_changed': False
        }
        
        # Check if we have a new best metric
        dice_score = metrics.get('dice_score', 0)
        if dice_score > self.best_metric:
            self.best_metric = dice_score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # If in cooldown, don't change hyperparameters
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            action_info['reason'] = f'in_cooldown_{self.cooldown_counter}'
            
            # For logging purposes, still select an action
            selected_param = self._select_parameter(self._analyze_trend())
            selected_action = self._select_action(self._analyze_trend(), selected_param)
            
            action_info['parameter'] = selected_param
            action_info['action'] = selected_action
            
            # Log what would have happened
            if self.verbose:
                print(f"[RL AGENT DEBUG] In cooldown ({self.cooldown_counter} epochs left), not changing hyperparameters")
                print(f"[RL AGENT INFO] If not in cooldown, would have adjusted {selected_param} with action {selected_action}")
            
            self._log_action(action_info, metrics)
            return action_info
        
        # Special case: if loss is stuck at 1.0 and we're using instance norm, try batch norm
        if (abs(metrics.get('val_loss', 0) - 1.0) < 0.001 and 
            self.normalization_type == 'instance_norm' and 
            not self.include_background):
            
            action_info['parameter'] = 'normalization_type'
            action_info['action'] = 'use_batch_norm'
            action_info['old_value'] = 'instance_norm'
            action_info['new_value'] = 'batch_norm'
            action_info['reason'] = 'escape_loss_plateau'
            action_info['hyperparameters_changed'] = True
            
            if self.verbose:
                print(f"[RL AGENT] Epoch {self.current_epoch}: switching to batch norm to escape loss plateau")
            
            self.normalization_type = 'batch_norm'
            self.cooldown_counter = self.cooldown
            
            self._log_action(action_info, metrics)
            
            # Store current state and action for next reward calculation
            self.last_state = self._analyze_trend()
            self.last_action = ('normalization_type', 'use_batch_norm')
            
            return action_info
            
        # Additional special case: if loss is STILL stuck at 1.0 after switching to batch norm, try including background
        if (abs(metrics.get('val_loss', 0) - 1.0) < 0.001 and 
            self.normalization_type == 'batch_norm' and 
            not self.include_background):
            
            action_info['parameter'] = 'include_background'
            action_info['action'] = 'include'
            action_info['old_value'] = False
            action_info['new_value'] = True
            action_info['reason'] = 'escape_loss_plateau_with_background'
            action_info['hyperparameters_changed'] = True
            
            if self.verbose:
                print(f"[RL AGENT] Epoch {self.current_epoch}: switching to include_background=True to escape loss plateau")
            
            self.include_background = True
            self.cooldown_counter = self.cooldown
            
            self._log_action(action_info, metrics)
            
            # Store current state and action for next reward calculation
            self.last_state = self._analyze_trend()
            self.last_action = ('include_background', 'include')
            
            return action_info
        
        # Select parameter to adjust based on current state
        selected_param = self._select_parameter(self._analyze_trend())
        action_info['parameter'] = selected_param
        
        # Select action for the chosen parameter
        selected_action = self._select_action(self._analyze_trend(), selected_param)
        action_info['action'] = selected_action
        
        # Apply the selected action to the chosen parameter
        if selected_param == 'learning_rate':
            old_lr = self.learning_rate
            action_info['old_value'] = old_lr
            
            if selected_action == 'decrease':
                new_lr = max(self.learning_rate * 0.7, self.min_lr)
                action_info['new_value'] = new_lr
                action_info['reason'] = 'rl_policy_decrease_lr'
                
                # Only change if it's actually different
                if new_lr != self.learning_rate and self.optimizer is not None:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease learning rate from {self.learning_rate:.6f} to {new_lr:.6f}")
                    
                    # Update optimizer
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    self.learning_rate = new_lr
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_lr = min(self.learning_rate * 1.5, self.max_lr)
                action_info['new_value'] = new_lr
                action_info['reason'] = 'rl_policy_increase_lr'
                
                # Only change if it's actually different
                if new_lr != self.learning_rate and self.optimizer is not None:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase learning rate from {self.learning_rate:.6f} to {new_lr:.6f}")
                    
                    # Update optimizer
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    self.learning_rate = new_lr
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'lambda_ce':
            old_lambda_ce = self.lambda_ce
            action_info['old_value'] = old_lambda_ce
            
            if selected_action == 'decrease':
                new_lambda_ce = max(self.lambda_ce * 0.7, self.lambda_ce_range[0])
                action_info['new_value'] = new_lambda_ce
                action_info['reason'] = 'rl_policy_decrease_lambda_ce'
                
                if new_lambda_ce != self.lambda_ce and hasattr(self.loss_function, 'lambda_ce'):
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease lambda_ce from {self.lambda_ce:.2f} to {new_lambda_ce:.2f}")
                    
                    # Update loss function
                    self.loss_function.lambda_ce = new_lambda_ce
                    self.lambda_ce = new_lambda_ce
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_lambda_ce = min(self.lambda_ce * 1.5, self.lambda_ce_range[1])
                action_info['new_value'] = new_lambda_ce
                action_info['reason'] = 'rl_policy_increase_lambda_ce'
                
                if new_lambda_ce != self.lambda_ce and hasattr(self.loss_function, 'lambda_ce'):
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase lambda_ce from {self.lambda_ce:.2f} to {new_lambda_ce:.2f}")
                    
                    # Update loss function
                    self.loss_function.lambda_ce = new_lambda_ce
                    self.lambda_ce = new_lambda_ce
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'lambda_dice':
            old_lambda_dice = self.lambda_dice
            action_info['old_value'] = old_lambda_dice
            
            if selected_action == 'decrease':
                new_lambda_dice = max(self.lambda_dice * 0.7, self.lambda_dice_range[0])
                action_info['new_value'] = new_lambda_dice
                action_info['reason'] = 'rl_policy_decrease_lambda_dice'
                
                if new_lambda_dice != self.lambda_dice and hasattr(self.loss_function, 'lambda_dice'):
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease lambda_dice from {self.lambda_dice:.2f} to {new_lambda_dice:.2f}")
                    
                    # Update loss function
                    self.loss_function.lambda_dice = new_lambda_dice
                    self.lambda_dice = new_lambda_dice
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_lambda_dice = min(self.lambda_dice * 1.5, self.lambda_dice_range[1])
                action_info['new_value'] = new_lambda_dice
                action_info['reason'] = 'rl_policy_increase_lambda_dice'
                
                if new_lambda_dice != self.lambda_dice and hasattr(self.loss_function, 'lambda_dice'):
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase lambda_dice from {self.lambda_dice:.2f} to {new_lambda_dice:.2f}")
                    
                    # Update loss function
                    self.loss_function.lambda_dice = new_lambda_dice
                    self.lambda_dice = new_lambda_dice
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'class_weights':
            old_weights = self.class_weights.copy()
            action_info['old_value'] = old_weights
            
            if selected_action == 'emphasize_foreground':
                # Increase weights for foreground classes, decrease for background
                new_weights = [max(w * 0.5, self.class_weight_range[0]) if i == 0 else min(w * 1.5, self.class_weight_range[1]) for i, w in enumerate(self.class_weights)]
                action_info['new_value'] = new_weights
                action_info['reason'] = 'rl_policy_emphasize_foreground'
                
                if new_weights != self.class_weights and hasattr(self.loss_function, 'weight'):
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: emphasize foreground classes, weights from {self.class_weights} to {new_weights}")
                    
                    # Update loss function
                    self.loss_function.weight = torch.tensor(new_weights, device=self.loss_function.weight.device)
                    self.class_weights = new_weights
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'balance':
                # Move weights closer to balanced
                new_weights = [(w + 1.0) / 2.0 for w in self.class_weights]
                action_info['new_value'] = new_weights
                action_info['reason'] = 'rl_policy_balance_classes'
                
                if new_weights != self.class_weights and hasattr(self.loss_function, 'weight'):
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: balance class weights from {self.class_weights} to {new_weights}")
                    
                    # Update loss function
                    self.loss_function.weight = torch.tensor(new_weights, device=self.loss_function.weight.device)
                    self.class_weights = new_weights
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'threshold':
            old_threshold = self.threshold
            action_info['old_value'] = old_threshold
            
            if selected_action == 'decrease':
                new_threshold = max(self.threshold - 0.1, self.threshold_range[0])
                action_info['new_value'] = new_threshold
                action_info['reason'] = 'rl_policy_decrease_threshold'
                
                if new_threshold != self.threshold:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease threshold from {self.threshold:.2f} to {new_threshold:.2f}")
                    
                    self.threshold = new_threshold
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_threshold = min(self.threshold + 0.1, self.threshold_range[1])
                action_info['new_value'] = new_threshold
                action_info['reason'] = 'rl_policy_increase_threshold'
                
                if new_threshold != self.threshold:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase threshold from {self.threshold:.2f} to {new_threshold:.2f}")
                    
                    self.threshold = new_threshold
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'include_background':
            old_include_background = self.include_background
            action_info['old_value'] = old_include_background
            
            if selected_action == 'include':
                new_include_background = True
                action_info['new_value'] = new_include_background
                action_info['reason'] = 'rl_policy_include_background'
                
                if new_include_background != self.include_background:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: include background")
                    
                    self.include_background = new_include_background
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'exclude':
                new_include_background = False
                action_info['new_value'] = new_include_background
                action_info['reason'] = 'rl_policy_exclude_background'
                
                if new_include_background != self.include_background:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: exclude background")
                    
                    self.include_background = new_include_background
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'normalization_type':
            old_normalization_type = self.normalization_type
            action_info['old_value'] = old_normalization_type
            
            if selected_action == 'use_batch_norm':
                new_normalization_type = 'batch_norm'
                action_info['new_value'] = new_normalization_type
                action_info['reason'] = 'rl_policy_use_batch_norm'
                
                if new_normalization_type != self.normalization_type:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: use batch normalization")
                    
                    self.normalization_type = new_normalization_type
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'use_instance_norm':
                new_normalization_type = 'instance_norm'
                action_info['new_value'] = new_normalization_type
                action_info['reason'] = 'rl_policy_use_instance_norm'
                
                if new_normalization_type != self.normalization_type:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: use instance normalization")
                    
                    self.normalization_type = new_normalization_type
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'weight_decay':
            old_weight_decay = self.weight_decay
            action_info['old_value'] = old_weight_decay
            
            if selected_action == 'decrease':
                new_weight_decay = max(self.weight_decay * 0.7, self.weight_decay_range[0])
                action_info['new_value'] = new_weight_decay
                action_info['reason'] = 'rl_policy_decrease_weight_decay'
                
                if new_weight_decay != self.weight_decay and self.optimizer is not None:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease weight decay from {self.weight_decay:.6f} to {new_weight_decay:.6f}")
                    
                    # Update optimizer
                    for param_group in self.optimizer.param_groups:
                        param_group['weight_decay'] = new_weight_decay
                    
                    self.weight_decay = new_weight_decay
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_weight_decay = min(self.weight_decay * 1.5, self.weight_decay_range[1])
                action_info['new_value'] = new_weight_decay
                action_info['reason'] = 'rl_policy_increase_weight_decay'
                
                if new_weight_decay != self.weight_decay and self.optimizer is not None:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase weight decay from {self.weight_decay:.6f} to {new_weight_decay:.6f}")
                    
                    # Update optimizer
                    for param_group in self.optimizer.param_groups:
                        param_group['weight_decay'] = new_weight_decay
                    
                    self.weight_decay = new_weight_decay
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'momentum':
            old_momentum = self.momentum
            action_info['old_value'] = old_momentum
            
            if selected_action == 'decrease':
                new_momentum = max(self.momentum * 0.7, self.momentum_range[0])
                action_info['new_value'] = new_momentum
                action_info['reason'] = 'rl_policy_decrease_momentum'
                
                if new_momentum != self.momentum and self.optimizer is not None:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease momentum from {self.momentum:.2f} to {new_momentum:.2f}")
                    
                    # Update optimizer
                    for param_group in self.optimizer.param_groups:
                        param_group['momentum'] = new_momentum
                    
                    self.momentum = new_momentum
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_momentum = min(self.momentum * 1.5, self.momentum_range[1])
                action_info['new_value'] = new_momentum
                action_info['reason'] = 'rl_policy_increase_momentum'
                
                if new_momentum != self.momentum and self.optimizer is not None:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase momentum from {self.momentum:.2f} to {new_momentum:.2f}")
                    
                    # Update optimizer
                    for param_group in self.optimizer.param_groups:
                        param_group['momentum'] = new_momentum
                    
                    self.momentum = new_momentum
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'dropout_rate':
            old_dropout_rate = self.dropout_rate
            action_info['old_value'] = old_dropout_rate
            
            if selected_action == 'decrease':
                new_dropout_rate = max(self.dropout_rate * 0.7, self.dropout_rate_range[0])
                action_info['new_value'] = new_dropout_rate
                action_info['reason'] = 'rl_policy_decrease_dropout_rate'
                
                if new_dropout_rate != self.dropout_rate:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease dropout rate from {self.dropout_rate:.2f} to {new_dropout_rate:.2f}")
                    
                    self.dropout_rate = new_dropout_rate
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_dropout_rate = min(self.dropout_rate * 1.5, self.dropout_rate_range[1])
                action_info['new_value'] = new_dropout_rate
                action_info['reason'] = 'rl_policy_increase_dropout_rate'
                
                if new_dropout_rate != self.dropout_rate:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase dropout rate from {self.dropout_rate:.2f} to {new_dropout_rate:.2f}")
                    
                    self.dropout_rate = new_dropout_rate
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'augmentation_intensity':
            old_augmentation_intensity = self.augmentation_intensity
            action_info['old_value'] = old_augmentation_intensity
            
            if selected_action == 'decrease':
                new_augmentation_intensity = max(self.augmentation_intensity * 0.7, self.augmentation_intensity_range[0])
                action_info['new_value'] = new_augmentation_intensity
                action_info['reason'] = 'rl_policy_decrease_augmentation_intensity'
                
                if new_augmentation_intensity != self.augmentation_intensity:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease augmentation intensity from {self.augmentation_intensity:.2f} to {new_augmentation_intensity:.2f}")
                    
                    self.augmentation_intensity = new_augmentation_intensity
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_augmentation_intensity = min(self.augmentation_intensity * 1.5, self.augmentation_intensity_range[1])
                action_info['new_value'] = new_augmentation_intensity
                action_info['reason'] = 'rl_policy_increase_augmentation_intensity'
                
                if new_augmentation_intensity != self.augmentation_intensity:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase augmentation intensity from {self.augmentation_intensity:.2f} to {new_augmentation_intensity:.2f}")
                    
                    self.augmentation_intensity = new_augmentation_intensity
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'augmentation_probability':
            old_augmentation_probability = self.augmentation_probability
            action_info['old_value'] = old_augmentation_probability
            
            if selected_action == 'decrease':
                new_augmentation_probability = max(self.augmentation_probability * 0.7, self.augmentation_probability_range[0])
                action_info['new_value'] = new_augmentation_probability
                action_info['reason'] = 'rl_policy_decrease_augmentation_probability'
                
                if new_augmentation_probability != self.augmentation_probability:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease augmentation probability from {self.augmentation_probability:.2f} to {new_augmentation_probability:.2f}")
                    
                    self.augmentation_probability = new_augmentation_probability
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_augmentation_probability = min(self.augmentation_probability * 1.5, self.augmentation_probability_range[1])
                action_info['new_value'] = new_augmentation_probability
                action_info['reason'] = 'rl_policy_increase_augmentation_probability'
                
                if new_augmentation_probability != self.augmentation_probability:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase augmentation probability from {self.augmentation_probability:.2f} to {new_augmentation_probability:.2f}")
                    
                    self.augmentation_probability = new_augmentation_probability
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'focal_gamma':
            old_focal_gamma = self.focal_gamma
            action_info['old_value'] = old_focal_gamma
            
            if selected_action == 'decrease':
                new_focal_gamma = max(self.focal_gamma * 0.7, self.focal_gamma_range[0])
                action_info['new_value'] = new_focal_gamma
                action_info['reason'] = 'rl_policy_decrease_focal_gamma'
                
                if new_focal_gamma != self.focal_gamma:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease focal gamma from {self.focal_gamma:.2f} to {new_focal_gamma:.2f}")
                    
                    self.focal_gamma = new_focal_gamma
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_focal_gamma = min(self.focal_gamma * 1.5, self.focal_gamma_range[1])
                action_info['new_value'] = new_focal_gamma
                action_info['reason'] = 'rl_policy_increase_focal_gamma'
                
                if new_focal_gamma != self.focal_gamma:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase focal gamma from {self.focal_gamma:.2f} to {new_focal_gamma:.2f}")
                    
                    self.focal_gamma = new_focal_gamma
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
        elif selected_param == 'batch_size':
            old_batch_size = self.batch_size
            action_info['old_value'] = old_batch_size
            
            if selected_action == 'decrease':
                new_batch_size = max(self.batch_size - 4, self.batch_size_range[0])
                action_info['new_value'] = new_batch_size
                action_info['reason'] = 'rl_policy_decrease_batch_size'
                
                if new_batch_size != self.batch_size:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: decrease batch size from {self.batch_size} to {new_batch_size}")
                    
                    self.batch_size = new_batch_size
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
                    
            elif selected_action == 'increase':
                new_batch_size = min(self.batch_size + 4, self.batch_size_range[1])
                action_info['new_value'] = new_batch_size
                action_info['reason'] = 'rl_policy_increase_batch_size'
                
                if new_batch_size != self.batch_size:
                    if self.verbose:
                        print(f"[RL AGENT] Epoch {self.current_epoch}: increase batch size from {self.batch_size} to {new_batch_size}")
                    
                    self.batch_size = new_batch_size
                    action_info['hyperparameters_changed'] = True
                    self.cooldown_counter = self.cooldown
        
        # Store current state and action for next iteration
        self.last_state = self._analyze_trend()
        self.last_action = (selected_param, selected_action)
        
        # Log the action
        self._log_action(action_info, metrics)
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

    def _analyze_trend(self) -> str:
        """
        Analyze the trend in validation metrics to determine the current state
        
        Returns:
            str: Current state of training
        """
        if len(self.metrics_history) < 3:
            return 'insufficient_data'
        
        # Extract recent metrics
        recent_metrics = self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
        
        # Extract validation metrics
        val_losses = [m.get('val_loss', float('inf')) for m in recent_metrics]
        val_dices = [m.get('val_dice', 0.0) for m in recent_metrics]
        
        # Extract training metrics if available
        train_losses = [m.get('train_loss', float('inf')) for m in recent_metrics]
        train_dices = [m.get('train_dice', 0.0) for m in recent_metrics]
        
        # Extract gradient norms if available
        grad_norms = [m.get('grad_norm', 0.0) for m in recent_metrics]
        
        # Calculate metrics for trend analysis
        val_loss_diff = [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]
        val_dice_diff = [val_dices[i] - val_dices[i-1] for i in range(1, len(val_dices))]
        
        # Calculate training-validation gaps
        train_val_loss_gaps = []
        train_val_dice_gaps = []
        
        for i in range(len(recent_metrics)):
            if 'train_loss' in recent_metrics[i] and 'val_loss' in recent_metrics[i]:
                train_val_loss_gaps.append(recent_metrics[i]['train_loss'] - recent_metrics[i]['val_loss'])
            
            if 'train_dice' in recent_metrics[i] and 'val_dice' in recent_metrics[i]:
                train_val_dice_gaps.append(recent_metrics[i]['train_dice'] - recent_metrics[i]['val_dice'])
        
        # Calculate variance in metrics to detect instability
        val_loss_variance = np.var(val_losses) if len(val_losses) > 1 else 0
        val_dice_variance = np.var(val_dices) if len(val_dices) > 1 else 0
        
        # Calculate oscillation detection
        oscillation_detected = False
        if len(val_loss_diff) >= 3:
            # Check for alternating signs in differences
            sign_changes = sum(1 for i in range(1, len(val_loss_diff)) if val_loss_diff[i] * val_loss_diff[i-1] < 0)
            oscillation_detected = sign_changes >= len(val_loss_diff) // 2
        
        # Calculate convergence speed
        convergence_speed = 0
        if len(val_dice_diff) >= 2:
            # Average of recent improvements
            convergence_speed = sum(val_dice_diff) / len(val_dice_diff)
        
        # Check for gradient explosion
        gradient_explosion = False
        if len(grad_norms) >= 2 and all(g > 0 for g in grad_norms):
            gradient_explosion = grad_norms[-1] > 10 * grad_norms[0]
        
        # Check for gradient vanishing
        gradient_vanishing = False
        if len(grad_norms) >= 2 and all(g > 0 for g in grad_norms):
            gradient_vanishing = grad_norms[-1] < 0.1 * grad_norms[0]
        
        # DETECTION LOGIC
        
        # Check for diverging (rapidly increasing validation loss)
        if any(vl > 10 for vl in val_losses[-2:]) or gradient_explosion:
            return 'diverging'
        
        # Check for overfitting
        overfitting_signals = 0
        
        # Signal 1: Training dice much higher than validation dice
        if len(train_val_dice_gaps) >= 2 and all(gap > 0.1 for gap in train_val_dice_gaps[-2:]):
            overfitting_signals += 1
        
        # Signal 2: Validation loss increasing while training loss decreasing
        if (len(val_loss_diff) >= 2 and len(train_losses) >= 3 and 
            all(diff > 0 for diff in val_loss_diff[-2:]) and 
            train_losses[-1] < train_losses[-3]):
            overfitting_signals += 1
        
        # Signal 3: Validation dice decreasing while training dice increasing
        if (len(val_dice_diff) >= 2 and len(train_dices) >= 3 and 
            all(diff < 0 for diff in val_dice_diff[-2:]) and 
            train_dices[-1] > train_dices[-3]):
            overfitting_signals += 1
        
        if overfitting_signals >= 2:
            return 'overfitting'
        
        # Check for underfitting
        underfitting_signals = 0
        
        # Signal 1: Both training and validation dice are low
        if len(train_dices) >= 2 and len(val_dices) >= 2:
            if all(dice < 0.5 for dice in train_dices[-2:]) and all(dice < 0.5 for dice in val_dices[-2:]):
                underfitting_signals += 1
        
        # Signal 2: Both training and validation loss are high
        if len(train_losses) >= 2 and len(val_losses) >= 2:
            if all(loss > 0.7 for loss in train_losses[-2:]) and all(loss > 0.7 for loss in val_losses[-2:]):
                underfitting_signals += 1
        
        # Signal 3: Very slow convergence
        if convergence_speed < 0.01 and len(val_dices) >= 3 and val_dices[-1] < 0.6:
            underfitting_signals += 1
        
        # Signal 4: Gradient vanishing
        if gradient_vanishing:
            underfitting_signals += 1
        
        if underfitting_signals >= 2:
            return 'underfitting'
        
        # Check for unstable training
        if oscillation_detected or val_loss_variance > 0.05 or val_dice_variance > 0.05:
            return 'unstable'
        
        # Check for plateau
        plateau_signals = 0
        
        # Signal 1: Very small changes in validation loss
        if len(val_loss_diff) >= 3 and all(abs(diff) < 0.005 for diff in val_loss_diff[-3:]):
            plateau_signals += 1
        
        # Signal 2: Very small changes in validation dice
        if len(val_dice_diff) >= 3 and all(abs(diff) < 0.005 for diff in val_dice_diff[-3:]):
            plateau_signals += 1
        
        # Signal 3: Gradient norm is very small
        if len(grad_norms) >= 2 and all(norm < 0.01 for norm in grad_norms[-2:]):
            plateau_signals += 1
        
        if plateau_signals >= 2:
            return 'plateau'
        
        # Check for converging (consistently improving validation metrics)
        converging_signals = 0
        
        # Signal 1: Validation loss consistently decreasing
        if len(val_loss_diff) >= 3 and all(diff < 0 for diff in val_loss_diff[-3:]):
            converging_signals += 1
        
        # Signal 2: Validation dice consistently increasing
        if len(val_dice_diff) >= 3 and all(diff > 0 for diff in val_dice_diff[-3:]):
            converging_signals += 1
        
        # Signal 3: Good convergence speed
        if convergence_speed > 0.01:
            converging_signals += 1
        
        if converging_signals >= 2:
            return 'converging'
        
        # Default: improving (general case when no specific pattern is detected)
        return 'improving'
            
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calculate the reward for the previous action based on improvements in metrics
        
        Args:
            metrics: Dictionary of metrics from the current epoch
            
        Returns:
            float: Reward value
        """
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Get current and previous metrics
        current_metrics = metrics
        previous_metrics = self.metrics_history[-2]
        
        # Extract metrics
        current_dice = current_metrics.get('val_dice', 0.0)
        previous_dice = previous_metrics.get('val_dice', 0.0)
        
        current_loss = current_metrics.get('val_loss', float('inf'))
        previous_loss = previous_metrics.get('val_loss', float('inf'))
        
        # Get current state
        current_state = self._analyze_trend()
        
        # Base reward components
        dice_improvement = current_dice - previous_dice
        loss_improvement = previous_loss - current_loss  # Reversed since lower loss is better
        
        # Initialize base reward
        reward = 0.0
        
        # Reward for dice improvement (higher weight)
        reward += dice_improvement * 10.0
        
        # Reward for loss improvement
        reward += loss_improvement * 5.0
        
        # Reward for absolute performance
        if current_dice > 0.9:
            reward += 2.0  # Bonus for excellent performance
        elif current_dice > 0.8:
            reward += 1.0  # Bonus for good performance
        
        # Penalty for very poor performance
        if current_dice < 0.1 and self.current_epoch > 5:
            reward -= 2.0
        
        # State-specific rewards and penalties
        if current_state == 'overfitting':
            # Penalize overfitting state
            reward -= 1.5
            
            # Check if we're escaping overfitting
            if self.last_state == 'overfitting' and dice_improvement > 0:
                reward += 3.0  # Bonus for escaping overfitting
                
            # Additional penalty based on train-val gap
            if 'train_dice' in current_metrics and 'val_dice' in current_metrics:
                train_val_gap = current_metrics['train_dice'] - current_metrics['val_dice']
                if train_val_gap > 0.2:
                    reward -= train_val_gap * 2.0  # Penalty proportional to the gap
        
        elif current_state == 'underfitting':
            # Penalize underfitting state
            reward -= 1.0
            
            # Reward for escaping underfitting
            if self.last_state == 'underfitting' and dice_improvement > 0:
                reward += 2.0
                
            # Additional penalty for very high loss
            if current_loss > 0.8:
                reward -= 1.0
        
        elif current_state == 'diverging':
            # Severe penalty for diverging
            reward -= 3.0
            
            # Bonus for escaping divergence
            if self.last_state == 'diverging' and loss_improvement > 0:
                reward += 4.0
        
        elif current_state == 'plateau':
            # Slight penalty for plateau
            reward -= 0.5
            
            # Reward for escaping plateau
            if self.last_state == 'plateau' and (dice_improvement > 0.01 or loss_improvement > 0.01):
                reward += 2.0
        
        elif current_state == 'unstable':
            # Penalty for unstable training
            reward -= 1.0
            
            # Calculate stability improvement
            if len(self.metrics_history) >= 4:
                previous_variance = np.var([m.get('val_dice', 0.0) for m in self.metrics_history[-4:-1]])
                current_variance = np.var([m.get('val_dice', 0.0) for m in self.metrics_history[-3:]])
                
                if current_variance < previous_variance:
                    reward += 1.0  # Reward for improved stability
        
        elif current_state == 'converging':
            # Bonus for being in converging state
            reward += 1.0
            
            # Extra reward for fast convergence
            if dice_improvement > 0.02:
                reward += 1.0
        
        elif current_state == 'improving':
            # Small bonus for general improvement
            reward += 0.5
        
        # Reward for balanced improvement in both training and validation
        if ('train_dice' in current_metrics and 'train_dice' in previous_metrics and
            'val_dice' in current_metrics and 'val_dice' in previous_metrics):
            
            train_improvement = current_metrics['train_dice'] - previous_metrics['train_dice']
            val_improvement = current_metrics['val_dice'] - previous_metrics['val_dice']
            
            # Both improving is good
            if train_improvement > 0 and val_improvement > 0:
                reward += 1.0
                
                # Balanced improvement is even better (similar improvement in both)
                if abs(train_improvement - val_improvement) < 0.01:
                    reward += 0.5
        
        # Reward for efficiency (finding good hyperparameters quickly)
        if current_dice > 0.7 and self.current_epoch < 20:
            efficiency_bonus = (0.9 - (self.current_epoch / 50)) * 2.0  # Higher bonus for earlier good results
            reward += max(0, efficiency_bonus)
        
        # Penalty for excessive hyperparameter changes
        if hasattr(self, 'action_history') and len(self.action_history) > 5:
            recent_changes = sum(1 for a in self.action_history[-5:] if a['hyperparameters_changed'])
            if recent_changes >= 4:  # Too many changes in a short period
                reward -= 0.5
        
        # Reward for specific parameter improvements
        param, action = self.last_action
        
        # Special case for learning rate
        if param == 'learning_rate':
            # If we decreased learning rate and it helped with instability
            if action == 'decrease' and self.last_state in ['unstable', 'diverging'] and current_state not in ['unstable', 'diverging']:
                reward += 1.5
                
            # If we increased learning rate and escaped plateau
            elif action == 'increase' and self.last_state == 'plateau' and current_state != 'plateau':
                reward += 1.5
        
        # Special case for regularization parameters
        elif param in ['weight_decay', 'dropout_rate', 'class_weights']:
            # If we adjusted regularization and it helped with overfitting
            if self.last_state == 'overfitting' and current_state != 'overfitting':
                reward += 1.5
        
        # Special case for batch size
        elif param == 'batch_size':
            # If we changed batch size and it improved stability
            if self.last_state == 'unstable' and current_state not in ['unstable', 'diverging']:
                reward += 1.0
        
        # Clip reward to reasonable range
        reward = max(min(reward, 10.0), -10.0)
        
        if self.verbose:
            print(f"[RL AGENT] Reward: {reward:.4f} (dice: {current_dice:.4f}, loss: {current_loss:.4f}, state: {current_state})")
        
        return reward
        
    def _select_parameter(self, state: str) -> str:
        """
        Select which hyperparameter to adjust based on the current state
        
        Args:
            state: Current state of training
            
        Returns:
            str: Selected hyperparameter to adjust
        """
        # If in cooldown, don't change parameters
        if self.cooldown_counter > 0:
            return 'none'
            
        # Group parameters by function
        regularization_params = ['weight_decay', 'dropout_rate', 'class_weights', 'lambda_ce', 'lambda_dice']
        optimization_params = ['learning_rate', 'momentum', 'batch_size']
        architecture_params = ['normalization_type']
        augmentation_params = ['augmentation_intensity', 'augmentation_probability']
        loss_params = ['focal_gamma', 'include_background', 'threshold']
        
        # Analyze recent performance to identify problematic parameters
        problematic_params = self._identify_problematic_parameters()
        
        # If we have identified specific problematic parameters, prioritize them
        if problematic_params and np.random.random() < 0.7:  # 70% chance to address specific problems
            return np.random.choice(problematic_params)
        
        # State-specific parameter selection
        if state == 'overfitting':
            # For overfitting, prioritize regularization parameters
            if np.random.random() < 0.8:  # 80% chance to use regularization params
                # Weight the regularization parameters based on their effectiveness
                weights = []
                for param in regularization_params:
                    if param in self.reward_history[state] and self.reward_history[state][param]:
                        # Calculate average reward for this parameter in this state
                        avg_reward = np.mean([r for action_rewards in self.reward_history[state][param].values() 
                                             for r in action_rewards])
                        # Convert to positive weight with bias toward higher rewards
                        weights.append(max(0.1, avg_reward + 5))
                    else:
                        weights.append(1.0)  # Default weight
                        
                # Normalize weights
                if sum(weights) > 0:
                    weights = [w/sum(weights) for w in weights]
                    return np.random.choice(regularization_params, p=weights)
                
                return np.random.choice(regularization_params)
            else:
                # Sometimes try other parameters
                return np.random.choice(self.hyperparameters)
                
        elif state == 'underfitting':
            # For underfitting, prioritize capacity and learning parameters
            capacity_params = optimization_params + architecture_params
            
            if np.random.random() < 0.8:  # 80% chance to use capacity params
                # Weight parameters based on past performance in this state
                weights = []
                for param in capacity_params:
                    if param in self.reward_history[state] and self.reward_history[state][param]:
                        avg_reward = np.mean([r for action_rewards in self.reward_history[state][param].values() 
                                             for r in action_rewards])
                        weights.append(max(0.1, avg_reward + 5))
                    else:
                        weights.append(1.0)
                        
                # Normalize weights
                if sum(weights) > 0:
                    weights = [w/sum(weights) for w in weights]
                    return np.random.choice(capacity_params, p=weights)
                
                return np.random.choice(capacity_params)
            else:
                # Sometimes try augmentation parameters
                return np.random.choice(augmentation_params + loss_params)
                
        elif state == 'plateau':
            # For plateau, prioritize learning rate and loss function parameters
            plateau_params = ['learning_rate'] + loss_params
            
            if np.random.random() < 0.7:  # Higher chance to change learning rate on plateau
                return 'learning_rate'
            else:
                # Weight parameters based on past performance in this state
                weights = []
                for param in plateau_params:
                    if param in self.reward_history[state] and self.reward_history[state][param]:
                        avg_reward = np.mean([r for action_rewards in self.reward_history[state][param].values() 
                                             for r in action_rewards])
                        weights.append(max(0.1, avg_reward + 5))
                    else:
                        weights.append(1.0)
                        
                # Normalize weights
                if sum(weights) > 0:
                    weights = [w/sum(weights) for w in weights]
                    return np.random.choice(plateau_params, p=weights)
                
                return np.random.choice(plateau_params)
                
        elif state == 'unstable' or state == 'oscillating':
            # For unstable training, prioritize parameters that affect stability
            stabilizing_params = ['learning_rate', 'momentum', 'batch_size']
            
            if np.random.random() < 0.8:
                # Weight parameters based on past performance in this state
                weights = []
                for param in stabilizing_params:
                    if param in self.reward_history[state] and self.reward_history[state][param]:
                        avg_reward = np.mean([r for action_rewards in self.reward_history[state][param].values() 
                                             for r in action_rewards])
                        weights.append(max(0.1, avg_reward + 5))
                    else:
                        weights.append(1.0)
                        
                # Normalize weights
                if sum(weights) > 0:
                    weights = [w/sum(weights) for w in weights]
                    return np.random.choice(stabilizing_params, p=weights)
                
                return np.random.choice(stabilizing_params)
            else:
                return np.random.choice(self.hyperparameters)
            
        elif state == 'converging':
            # For converging, fine-tune with threshold and class weights
            fine_tuning_params = ['threshold', 'class_weights', 'focal_gamma']
            
            # Weight parameters based on past performance in this state
            weights = []
            for param in fine_tuning_params:
                if param in self.reward_history[state] and self.reward_history[state][param]:
                    avg_reward = np.mean([r for action_rewards in self.reward_history[state][param].values() 
                                         for r in action_rewards])
                    weights.append(max(0.1, avg_reward + 5))
                else:
                    weights.append(1.0)
                    
            # Normalize weights
            if sum(weights) > 0:
                weights = [w/sum(weights) for w in weights]
                return np.random.choice(fine_tuning_params, p=weights)
            
            return np.random.choice(fine_tuning_params)
            
        elif state == 'diverging':
            # Emergency intervention for diverging - always adjust learning rate
            return 'learning_rate'
            
        elif state == 'improving':
            # For general improvement, try a mix of parameters
            # Prefer parameters that have worked well in the past
            
            # Calculate success rates for each parameter
            param_success_rates = {}
            for param in self.hyperparameters:
                positive_rewards = 0
                total_actions = 0
                
                for s in self.reward_history:
                    if param in self.reward_history[s]:
                        for action, rewards in self.reward_history[s][param].items():
                            positive_rewards += sum(1 for r in rewards if r > 0)
                            total_actions += len(rewards)
                
                if total_actions > 0:
                    param_success_rates[param] = positive_rewards / total_actions
                else:
                    param_success_rates[param] = 0.5  # Default 50% success rate
            
            # Select parameter with probability proportional to success rate
            params = list(param_success_rates.keys())
            weights = list(param_success_rates.values())
            
            # Normalize weights
            if sum(weights) > 0:
                weights = [w/sum(weights) for w in weights]
                return np.random.choice(params, p=weights)
            
            # Fallback to random selection
            return np.random.choice(self.hyperparameters)
        
        # Default: random selection
        return np.random.choice(self.hyperparameters)
    
    def _identify_problematic_parameters(self) -> List[str]:
        """
        Identify specific parameters that might be causing problems based on recent performance
        
        Returns:
            List[str]: List of potentially problematic parameters
        """
        problematic_params = []
        
        # Need enough history to identify problems
        if len(self.metrics_history) < 5:
            return problematic_params
        
        recent_metrics = self.metrics_history[-5:]
        
        # Check for gradient explosion (learning rate too high)
        if any(m.get('grad_norm', 0) > 10 for m in recent_metrics):
            problematic_params.append('learning_rate')
        
        # Check for high train-val gap (regularization needed)
        train_val_gaps = []
        for m in recent_metrics:
            if 'train_dice' in m and 'val_dice' in m:
                train_val_gaps.append(m['train_dice'] - m['val_dice'])
        
        if train_val_gaps and np.mean(train_val_gaps) > 0.2:
            problematic_params.extend(['weight_decay', 'dropout_rate'])
        
        # Check for class imbalance issues
        class_dices = []
        for m in recent_metrics:
            if 'class_dice_0' in m and 'class_dice_1' in m and 'class_dice_2' in m:
                class_dices.append([m['class_dice_0'], m['class_dice_1'], m['class_dice_2']])
        
        if class_dices:
            avg_class_dices = np.mean(class_dices, axis=0)
            if max(avg_class_dices) - min(avg_class_dices) > 0.3:  # Large disparity between classes
                problematic_params.extend(['class_weights', 'focal_gamma'])
        
        # Check for precision-recall imbalance
        precisions = [m.get('precision', 0.5) for m in recent_metrics if 'precision' in m]
        recalls = [m.get('recall', 0.5) for m in recent_metrics if 'recall' in m]
        
        if precisions and recalls:
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            
            if avg_precision > avg_recall * 1.5:  # Much higher precision than recall
                problematic_params.extend(['threshold', 'focal_gamma'])
            elif avg_recall > avg_precision * 1.5:  # Much higher recall than precision
                problematic_params.extend(['threshold', 'focal_gamma'])
        
        # Check for batch size issues (high variance in loss)
        if len(recent_metrics) >= 3:
            losses = [m.get('val_loss', 1.0) for m in recent_metrics]
            if np.var(losses) > 0.05:  # High variance
                problematic_params.append('batch_size')
        
        return list(set(problematic_params))  # Remove duplicates
    
    def _select_action(self, state: str, parameter: str) -> str:
        """
        Select action using a sophisticated policy for a specific parameter
        
        Args:
            state: Current state of training
            parameter: Parameter to select action for
            
        Returns:
            str: Selected action
        """
        # If parameter is 'none', return 'maintain'
        if parameter == 'none':
            return 'maintain'
            
        # Get available actions for this parameter
        available_actions = self.actions[parameter]
        
        # Check if we have historical performance data for this state-parameter combination
        has_history = (state in self.reward_history and 
                      parameter in self.reward_history[state] and 
                      any(len(rewards) > 0 for rewards in self.reward_history[state][parameter].values()))
        
        # Use deterministic policies for certain critical state-parameter combinations
        if self._should_use_deterministic_policy(state, parameter):
            deterministic_action = self._get_deterministic_action(state, parameter)
            if deterministic_action:
                return deterministic_action
        
        # Exploration vs. exploitation
        if np.random.random() < self._get_exploration_rate(state):
            # Exploration: random action
            return np.random.choice(available_actions)
        else:
            # Exploitation: choose action with highest expected reward
            
            if has_history:
                # Use historical performance to guide action selection
                action_rewards = {}
                
                for action in available_actions:
                    if action in self.reward_history[state][parameter]:
                        # Calculate average reward for this action
                        rewards = self.reward_history[state][parameter][action]
                        if rewards:
                            # Weight recent rewards more heavily
                            weighted_rewards = [rewards[i] * (0.8 + 0.2 * i / len(rewards)) for i in range(len(rewards))]
                            action_rewards[action] = sum(weighted_rewards) / len(weighted_rewards)
                        else:
                            action_rewards[action] = 0.0
                    else:
                        # No history for this action, use Q-value
                        action_rewards[action] = self.q_table[state][parameter][action]
                
                # Find actions with the highest reward
                max_reward = max(action_rewards.values())
                best_actions = [a for a, r in action_rewards.items() if r == max_reward]
                
                # Choose randomly among the best actions
                return np.random.choice(best_actions)
            else:
                # Use Q-values for action selection
                action_q_values = {a: self.q_table[state][parameter][a] for a in available_actions}
                
                # Find actions with the highest Q-value
                max_q = max(action_q_values.values())
                best_actions = [a for a, q in action_q_values.items() if q == max_q]
                
                # Choose randomly among the best actions
                return np.random.choice(best_actions)
    
    def _should_use_deterministic_policy(self, state: str, parameter: str) -> bool:
        """
        Determine if we should use a deterministic policy for this state-parameter combination
        
        Args:
            state: Current state of training
            parameter: Parameter to check
            
        Returns:
            bool: True if we should use a deterministic policy
        """
        # Critical situations that require deterministic policies
        critical_situations = [
            (state == 'diverging' and parameter == 'learning_rate'),
            (state == 'plateau' and parameter == 'learning_rate' and self.current_epoch > 20),
            (state == 'overfitting' and parameter in ['weight_decay', 'dropout_rate']),
            (state == 'unstable' and parameter == 'batch_size'),
            (state == 'unstable' and parameter == 'learning_rate')
        ]
        
        return any(critical_situations)
    
    def _get_deterministic_action(self, state: str, parameter: str) -> Optional[str]:
        """
        Get the deterministic action for a critical state-parameter combination
        
        Args:
            state: Current state of training
            parameter: Parameter to get action for
            
        Returns:
            str: Deterministic action to take, or None if no deterministic action
        """
        # Handle critical situations with deterministic actions
        if state == 'diverging' and parameter == 'learning_rate':
            return 'decrease'  # Always decrease learning rate when diverging
            
        elif state == 'plateau' and parameter == 'learning_rate':
            # If we've been training for a while and hit a plateau, try increasing learning rate
            if self.current_epoch > 20:
                return 'increase'
                
        elif state == 'overfitting' and parameter == 'weight_decay':
            return 'increase'  # Always increase weight decay when overfitting
            
        elif state == 'overfitting' and parameter == 'dropout_rate':
            return 'increase'  # Always increase dropout rate when overfitting
            
        elif state == 'unstable' and parameter == 'batch_size':
            return 'increase'  # Larger batch sizes can help stability
            
        elif state == 'unstable' and parameter == 'learning_rate':
            return 'decrease'  # Lower learning rates can help stability
        
        # No deterministic action for this state-parameter combination
        return None
    
    def _get_exploration_rate(self, state: str) -> float:
        """
        Get the exploration rate based on the current state and training progress
        
        Args:
            state: Current state of training
            
        Returns:
            float: Exploration rate to use
        """
        # Base exploration rate that decays over time
        base_rate = max(self.min_exploration_rate, 
                        self.exploration_rate * (self.exploration_decay ** (self.current_epoch / 10)))
        
        # Adjust exploration based on state
        if state == 'diverging':
            return 0.1 * base_rate  # Very low exploration when diverging
        elif state == 'converging':
            return 0.5 * base_rate  # Lower exploration when converging
        elif state == 'plateau':
            return 1.5 * base_rate  # Higher exploration when plateaued
        elif state == 'unstable':
            return 0.7 * base_rate  # Lower exploration when unstable
        else:
            return base_rate
    
    def _log_action(self, action_info: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        Log the action taken by the agent
        
        Args:
            action_info: Information about the action
            metrics: Current metrics
        """
        # Ensure state is present in action_info
        if 'state' not in action_info:
            action_info['state'] = self._analyze_trend()
            
        # Extract values for logging
        parameter = action_info.get('parameter', 'none')
        action = action_info.get('action', 'maintain')
        reason = action_info.get('reason', 'default')
        old_value = action_info.get('old_value', 0)
        new_value = action_info.get('new_value', 0)
        
        # Format values for logging
        if parameter in ['learning_rate', 'lambda_ce', 'lambda_dice', 'threshold', 'weight_decay', 'momentum', 'dropout_rate', 'augmentation_intensity', 'augmentation_probability', 'focal_gamma']:
            old_value_str = f"{old_value:.6f}" if old_value is not None else "None"
            new_value_str = f"{new_value:.6f}" if new_value is not None else "None"
        elif parameter == 'class_weights':
            old_value_str = str(old_value) if old_value is not None else "None"
            new_value_str = str(new_value) if new_value is not None else "None"
        elif parameter == 'include_background':
            old_value_str = str(old_value) if old_value is not None else "None"
            new_value_str = str(new_value) if new_value is not None else "None"
        elif parameter == 'normalization_type':
            old_value_str = str(old_value) if old_value is not None else "None"
            new_value_str = str(new_value) if new_value is not None else "None"
        elif parameter == 'batch_size':
            old_value_str = str(old_value) if old_value is not None else "None"
            new_value_str = str(new_value) if new_value is not None else "None"
        else:
            old_value_str = str(old_value)
            new_value_str = str(new_value)
            
        # Get Q-value if available
        q_value = 0.0
        if self.last_state is not None and parameter is not None and action is not None:
            q_value = self.q_table.get(self.last_state, {}).get(parameter, {}).get(action, 0.0)
            
        # Log to CSV
        with open(self.log_file, 'a') as f:
            f.write(f"{self.current_epoch},{action_info['state']},{parameter},{action},{reason},{old_value_str},{new_value_str},{self.last_reward:.4f},{q_value:.4f},{metrics.get('dice_score', 0):.4f},{metrics.get('val_loss', 0):.4f}\n")
            
    def _save_q_table(self) -> None:
        """
        Save the Q-table to a JSON file
        """
        q_table_file = os.path.join(self.log_dir, "agent_logs", "q_table.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_q_table = {}
        for state in self.q_table:
            serializable_q_table[state] = {}
            for param in self.q_table[state]:
                serializable_q_table[state][param] = {}
                for action in self.q_table[state][param]:
                    serializable_q_table[state][param][action] = float(self.q_table[state][param][action])
                    
        with open(q_table_file, 'w') as f:
            json.dump(serializable_q_table, f, indent=2)
            
    def get_current_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the current hyperparameter values
        
        Returns:
            dict: Current hyperparameter values
        """
        return {
            'learning_rate': self.learning_rate,
            'lambda_ce': self.lambda_ce,
            'lambda_dice': self.lambda_dice,
            'class_weights': self.class_weights,
            'threshold': self.threshold,
            'include_background': self.include_background,
            'normalization_type': self.normalization_type,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'dropout_rate': self.dropout_rate,
            'augmentation_intensity': self.augmentation_intensity,
            'augmentation_probability': self.augmentation_probability,
            'focal_gamma': self.focal_gamma,
            'batch_size': self.batch_size
        }
        
    def plot_learning_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot learning curves for the agent
        
        Args:
            save_path: Path to save the plot (if None, plot is displayed)
        """
        if len(self.metrics_history) < 2:
            print("Not enough data to plot learning curves")
            return
            
        # Extract metrics from history
        epochs = list(range(1, len(self.metrics_history) + 1))
        dice_scores = [m.get('dice_score', 0) for m in self.metrics_history]
        losses = [m.get('val_loss', float('inf')) for m in self.metrics_history]
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot metrics
        axs[0].plot(epochs, dice_scores, 'b-', label='Dice Score')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Dice Score')
        axs[0].set_title('Dice Score vs Epoch')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(epochs, losses, 'r-', label='Validation Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Validation Loss vs Epoch')
        axs[1].grid(True)
        axs[1].legend()
        
        # Extract learning rate history
        lr_history = []
        lr = self.base_lr
        for action in self.action_history:
            if action['parameter'] == 'learning_rate' and action['new_value'] is not None:
                lr = action['new_value']
            lr_history.append(lr)
            
        # If we don't have enough action history, pad with base_lr
        if len(lr_history) < len(epochs):
            lr_history = [self.base_lr] * (len(epochs) - len(lr_history)) + lr_history
            
        # Plot learning rate
        axs[2].plot(epochs, lr_history, 'g-', label='Learning Rate')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Learning Rate')
        axs[2].set_title('Learning Rate vs Epoch')
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
