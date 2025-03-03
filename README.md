# MONAI Hippocampus Segmentation Baseline

Real-Time AI Agent for Dynamic Learning Rate Adjustment in MONA


This repository contains a baseline implementation for the Task04_Hippocampus dataset from the Medical Segmentation Decathlon using MONAI.

## Dataset

The Task04_Hippocampus dataset consists of 3D MRI scans of the human brain, with segmentation masks for the hippocampus. The hippocampus is segmented into two parts: anterior and posterior.

## Model

The baseline model is a 3D U-Net implemented using MONAI. The model architecture includes:
- 3D convolutions
- 5 resolution levels (16, 32, 64, 128, 256 channels)
- Residual connections
- Instance normalization

## Training

The training script (`train_hippocampus.py`) includes:
- Data loading and preprocessing using MONAI transforms
- Training with Dice loss
- Validation with Dice metric
- Model checkpointing (saves the best model based on validation Dice score)
- Comprehensive metrics tracking

## Metrics Tracking

The training process tracks and logs the following metrics to a CSV file (`training_metrics.csv`):

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| Epoch | Current epoch number | Helps track progress over time |
| Training Loss | Loss on training data (Dice loss) | Core measure of model learning |
| Validation Loss | Loss on validation data | Detects overfitting if it diverges from training loss |
| Dice Score | Measures overlap between predicted and ground truth segmentation | Standard metric for segmentation tasks |
| AUROC | Area Under ROC Curve | Higher AUROC = better discrimination between classes |
| AUPRC | Area Under Precision-Recall Curve | Important for medical imaging tasks with imbalanced datasets |
| PPV | Positive Predictive Value (Precision) | Useful for evaluating false positive rate |
| NNE | Number Needed to Evaluate (1/PPV) | Measures clinical usefulness (lower = better) |
| Learning Rate | Current learning rate used in training | Tracks optimization adjustments |
| Batch Size | Number of samples per training batch | Helps analyze training stability |
| Data Distribution | Mean and variance of pixel intensities | Detects dataset drift across epochs |

## Learning Rate Agent

The learning rate agent (`lr_agent.py`) monitors training metrics and dynamically adjusts the learning rate based on observed performance trends. It can:

- Detect when training is improving, worsening, plateauing, or oscillating
- Increase learning rate when progress is too slow
- Decrease learning rate when training becomes unstable
- Maintain the current learning rate when progress is optimal

## RL-Enhanced Learning Rate Agent

The RL-enhanced learning rate agent (`rl_lr_agent.py`) uses reinforcement learning to dynamically adjust the learning rate during training. Unlike traditional schedulers, it learns from past adjustments to make smarter decisions over time.

### Key Features:

- **State Recognition**: Detects training trends (improving, worsening, plateau, oscillating)
- **Action Selection**: Uses epsilon-greedy policy to balance exploration and exploitation
- **Reward Function**: Rewards actions that improve Dice score and reduce validation loss
- **Q-Learning**: Updates Q-values based on observed rewards and future expected rewards
- **Adaptive Exploration**: Gradually reduces exploration rate as training progresses
- **Visualization**: Provides plots of learning rate changes, metrics, and Q-values

### Parameters:

- `base_lr`: Initial learning rate
- `min_lr`, `max_lr`: Boundaries for learning rate adjustments
- `patience`: Epochs to wait before considering adjustment
- `cooldown`: Epochs to wait after an adjustment before making another
- `learning_rate`: Learning rate for the RL agent (alpha)
- `discount_factor`: Weight for future rewards (gamma)
- `exploration_rate`: Initial exploration probability (epsilon)
- `min_exploration_rate`: Minimum exploration probability
- `exploration_decay`: Rate at which exploration probability decreases

## Visualization

A separate script (`plot_metrics.py`) is provided to visualize the metrics tracked during training:

```bash
python plot_metrics.py --csv_file Task04_Hippocampus/training_metrics.csv --output_dir Task04_Hippocampus
```

This script generates several plots:
- `basic_metrics.png`: Shows training/validation loss, Dice score, learning rate, and data distribution
- `advanced_metrics.png`: Shows AUROC, AUPRC, PPV, and NNE metrics
- `combined_metrics.png`: Shows multiple performance metrics on a single plot for comparison

## Usage

To train the model:

```bash
python train_hippocampus.py
```

The script will:
1. Load the Task04_Hippocampus dataset
2. Train a 3D U-Net for 20 epochs
3. Save the best model based on validation Dice score
4. Save all metrics to `training_metrics.csv`

After training, visualize the results:

```bash
python plot_metrics.py
```

### Training with Learning Rate Agent

```bash
python train_hippocampus.py --use_agent --epochs 100 --learning_rate 1e-4
```

### Training with RL-Enhanced Learning Rate Agent

```bash
python train_hippocampus.py --use_rl_agent --epochs 100 --learning_rate 1e-4
```

### Agent Parameters

- `--use_agent`: Enable the learning rate agent (default: disabled)
- `--agent_min_lr`: Minimum learning rate for agent (default: 1e-6)
- `--agent_max_lr`: Maximum learning rate for agent (default: 1e-2)
- `--agent_patience`: Epochs to wait before adjusting LR (default: 3)
- `--agent_cooldown`: Cooldown period after adjustment (default: 5)

## Requirements

- MONAI
- PyTorch
- NumPy
- Matplotlib
- pandas (for metrics visualization)
- scikit-learn (for ROC and PR curve calculations)

## Results

After training, you'll find:
- `best_metric_model.pth`: The model weights with the best validation Dice score
- `final_model.pth`: The model weights after the final epoch
- `training_metrics.csv`: A CSV file containing all tracked metrics for each epoch
- Various visualization plots (when running the plotting script)



🛠️ What Our AI Agents Will Do
	1.	Hyperparameter Analyzer 🤖
	•	Reads training logs (CSV) and identifies trends in Dice Loss, AUROC, etc.
	•	Suggests better hyperparameters for learning rate, batch size, and augmentations.
	2.	Hyperparameter Optimizer 🤖
	•	Dynamically adjusts training settings (e.g., lowering LR if overfitting is detected).
	•	Uses Bayesian optimization or reinforcement learning for fine-tuning.
	3.	AI Model Monitor 🤖
	•	Tracks performance drift (e.g., if AUROC/AUPRC starts declining).
	•	Detects dataset shifts (e.g., patient population changing over time).
	•	Alerts if model degradation is detected.