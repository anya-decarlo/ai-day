#!/bin/bash
# Script to compare baseline hyperparameters vs. RL agent hyperparameter optimization

# Create results directories
mkdir -p results_baseline
mkdir -p results_agent

# Remove existing synthetic data to force regeneration with harder settings
echo "Removing existing synthetic data..."
rm -rf SyntheticData

# Run training with baseline suboptimal hyperparameters (same starting point as the agent)
echo "Running training with baseline hyperparameters..."
python train_synthetic.py \
  --optimizer SGD \
  --loss Dice \
  --learning_rate 0.3 \
  --epochs 15 \
  --batch_size 4 \
  --augmentations None \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --gradient_clip 0.0 \
  > results_baseline/training_log.txt

# Run training with RL hyperparameter agent
echo "Running training with RL hyperparameter agent..."
python train_synthetic_with_hyperparameter_agent.py \
  --optimizer SGD \
  --loss Dice \
  --learning_rate 0.3 \
  --epochs 15 \
  --batch_size 4 \
  --augmentations None \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --gradient_clip 0.0 \
  --agent_learning_rate 0.3 \
  --agent_exploration_rate 0.7 \
  --agent_cooldown 0 \
  > results_agent/training_log.txt

# Compare results
echo "Comparison complete. Check results in results_baseline and results_agent directories."
echo "To visualize and compare the results, run: python compare_results.py"
