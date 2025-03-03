#!/bin/bash
# Run the full experiment: training followed by visualization

echo "Starting MONAI Hippocampus segmentation experiment..."

# Run the training script
echo "Training model..."
python train_hippocampus.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    
    # Run the plotting script
    echo "Generating visualization plots..."
    python plot_metrics.py
    
    if [ $? -eq 0 ]; then
        echo "Visualization completed successfully."
        echo "Experiment completed. Check the Task04_Hippocampus directory for results."
    else
        echo "Error generating visualization plots."
    fi
else
    echo "Error during training."
fi
