#!/usr/bin/env python
# Script to compare different hyperparameter optimization methods

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import argparse
from collections import defaultdict
import time

def load_results(results_dir, method_name):
    """
    Load results from a specific optimization method's directory
    """
    results_path = os.path.join(results_dir, f"results_{method_name}")
    
    # Check if directory exists
    if not os.path.exists(results_path):
        print(f"Warning: Results directory for {method_name} not found at {results_path}")
        return None
    
    # Load CSV results
    csv_path = os.path.join(results_path, f"{method_name}_results.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: Results CSV for {method_name} not found at {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading results for {method_name}: {e}")
        return None

def load_best_config(results_dir, method_name):
    """
    Load the best configuration for a specific optimization method
    """
    results_path = os.path.join(results_dir, f"results_{method_name}")
    best_config_path = os.path.join(results_path, "best_config.json")
    
    if not os.path.exists(best_config_path):
        print(f"Warning: Best config for {method_name} not found at {best_config_path}")
        return None
    
    try:
        with open(best_config_path, "r") as f:
            best_config = json.load(f)
        return best_config
    except Exception as e:
        print(f"Error loading best config for {method_name}: {e}")
        return None

def plot_performance_comparison(results, output_dir):
    """
    Create a bar chart comparing the best performance of each method
    """
    methods = list(results.keys())
    best_scores = [results[method]["best_score"] if method in results else 0 for method in methods]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, best_scores, color=sns.color_palette("muted"))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title("Best Dice Score by Optimization Method", fontsize=14)
    plt.ylabel("Dice Score", fontsize=12)
    plt.xlabel("Optimization Method", fontsize=12)
    plt.ylim(0, max(best_scores) * 1.1)  # Add some space for the text
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_performance_comparison.png"), dpi=300)
    plt.close()

def plot_convergence_comparison(results, output_dir):
    """
    Create a line plot showing the convergence of each method over trials/iterations
    """
    plt.figure(figsize=(12, 7))
    
    for method, data in results.items():
        if "convergence" in data:
            iterations = range(1, len(data["convergence"]) + 1)
            plt.plot(iterations, data["convergence"], label=method, marker='o', markersize=4)
    
    plt.title("Convergence of Optimization Methods", fontsize=14)
    plt.xlabel("Trial/Iteration", fontsize=12)
    plt.ylabel("Best Dice Score So Far", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set x-axis to show integers only
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_comparison.png"), dpi=300)
    plt.close()

def plot_parameter_importance(results, output_dir):
    """
    Create visualizations for parameter importance across methods
    """
    # Collect parameters from all methods
    all_params = set()
    for method, data in results.items():
        if "best_config" in data and data["best_config"] is not None:
            all_params.update(data["best_config"].keys())
    
    # Filter out non-numeric parameters and parameters with special formats
    exclude_params = {"class_weights"}
    numeric_params = [p for p in all_params if p not in exclude_params]
    
    if not numeric_params:
        print("No suitable parameters found for parameter importance plot")
        return
    
    # Create a dataframe with parameter values for each method
    param_data = []
    for param in numeric_params:
        row = {"Parameter": param}
        for method, data in results.items():
            if "best_config" in data and data["best_config"] is not None and param in data["best_config"]:
                value = data["best_config"][param]
                # Convert to numeric if possible
                if isinstance(value, (int, float)):
                    row[method] = value
                elif isinstance(value, bool):
                    row[method] = 1 if value else 0
                elif isinstance(value, str) and value.lower() in ["true", "false"]:
                    row[method] = 1 if value.lower() == "true" else 0
        param_data.append(row)
    
    param_df = pd.DataFrame(param_data)
    
    if len(param_df) == 0:
        print("No parameter data available for plotting")
        return
    
    # Melt the dataframe for easier plotting
    param_df_melted = pd.melt(param_df, id_vars=["Parameter"], 
                             var_name="Method", value_name="Value")
    
    # Create a heatmap of parameter values
    plt.figure(figsize=(12, max(6, len(numeric_params) * 0.5)))
    
    # Pivot the dataframe for the heatmap
    heatmap_df = param_df.set_index("Parameter")
    
    # Normalize values for better visualization
    normalized_df = heatmap_df.copy()
    for col in normalized_df.columns:
        normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / \
                            (normalized_df[col].max() - normalized_df[col].min() + 1e-10)
    
    # Plot heatmap
    sns.heatmap(normalized_df, annot=heatmap_df, fmt=".3g", cmap="YlGnBu", 
                linewidths=0.5, cbar_kws={"label": "Normalized Value"})
    
    plt.title("Parameter Values Across Optimization Methods", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_comparison.png"), dpi=300)
    plt.close()
    
    # Create a grouped bar chart for selected important parameters
    # Choose a subset of parameters (e.g., top 5) for clarity
    important_params = ["learning_rate", "weight_decay", "dropout_rate", "batch_size", "threshold"]
    important_params = [p for p in important_params if p in param_df["Parameter"].values]
    
    if len(important_params) > 0:
        important_df = param_df[param_df["Parameter"].isin(important_params)]
        important_df_melted = pd.melt(important_df, id_vars=["Parameter"], 
                                     var_name="Method", value_name="Value")
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x="Parameter", y="Value", hue="Method", data=important_df_melted)
        plt.title("Key Parameter Values by Optimization Method", fontsize=14)
        plt.ylabel("Parameter Value", fontsize=12)
        plt.xlabel("Parameter", fontsize=12)
        plt.legend(title="Method")
        plt.yscale("log")  # Log scale for better visualization of different magnitudes
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "key_parameters_comparison.png"), dpi=300)
        plt.close()

def extract_convergence_data(df, method):
    """
    Extract convergence data from results dataframe
    """
    if df is None or len(df) == 0:
        return []
    
    # Different methods might have different column names for iterations/trials
    id_columns = ["trial_id", "config_id", "iteration", "trial"]
    score_columns = ["dice_score", "metric", "score", "best_value"]
    
    # Find the appropriate columns
    id_col = next((col for col in id_columns if col in df.columns), None)
    score_col = next((col for col in score_columns if col in df.columns), None)
    
    if id_col is None or score_col is None:
        print(f"Warning: Could not find appropriate columns for {method}")
        return []
    
    # Sort by ID column
    df = df.sort_values(by=id_col)
    
    # Calculate the best score so far at each iteration
    best_so_far = []
    best_score = float("-inf")
    
    for _, row in df.iterrows():
        score = row[score_col]
        if score > best_score:
            best_score = score
        best_so_far.append(best_score)
    
    return best_so_far

def main():
    parser = argparse.ArgumentParser(description="Compare hyperparameter optimization methods")
    parser.add_argument("--output_dir", type=str, default="optimization_comparison",
                        help="Directory to save comparison results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Methods to compare
    methods = ["grid_search", "random_search", "optuna", "hyperparameter_agent"]
    
    # Load results for each method
    results = {}
    
    for method in methods:
        print(f"Loading results for {method}...")
        df = load_results(os.getcwd(), method)
        best_config = load_best_config(os.getcwd(), method)
        
        if df is not None:
            # Extract best score
            score_columns = ["dice_score", "metric", "score", "best_value"]
            score_col = next((col for col in score_columns if col in df.columns), None)
            
            if score_col is not None:
                best_score = df[score_col].max()
                
                # Extract convergence data
                convergence = extract_convergence_data(df, method)
                
                results[method] = {
                    "best_score": best_score,
                    "best_config": best_config,
                    "convergence": convergence,
                    "data": df
                }
                
                print(f"  Best score: {best_score:.4f}")
                print(f"  Trials/iterations: {len(df)}")
            else:
                print(f"  Could not find score column in results for {method}")
        else:
            print(f"  No results found for {method}")
    
    # Create comparison visualizations
    if results:
        print("\nCreating comparison visualizations...")
        
        # Performance comparison
        plot_performance_comparison(results, args.output_dir)
        
        # Convergence comparison
        plot_convergence_comparison(results, args.output_dir)
        
        # Parameter importance
        plot_parameter_importance(results, args.output_dir)
        
        # Create summary table
        summary = {
            "Method": [],
            "Best Dice Score": [],
            "Trials/Iterations": [],
            "Key Parameters": []
        }
        
        for method in methods:
            if method in results:
                data = results[method]
                
                # Get key parameters (top 3 by importance if available)
                key_params = ""
                if "best_config" in data and data["best_config"] is not None:
                    important_params = ["learning_rate", "weight_decay", "dropout_rate"]
                    params = [f"{p}={data['best_config'][p]:.4f}" if isinstance(data['best_config'][p], float) 
                             else f"{p}={data['best_config'][p]}" 
                             for p in important_params if p in data['best_config']]
                    key_params = ", ".join(params[:3])
                
                summary["Method"].append(method)
                summary["Best Dice Score"].append(f"{data['best_score']:.4f}")
                summary["Trials/Iterations"].append(len(data["data"]))
                summary["Key Parameters"].append(key_params)
        
        # Convert to dataframe and save
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(args.output_dir, "optimization_summary.csv"), index=False)
        
        # Also save as markdown table for easy viewing
        with open(os.path.join(args.output_dir, "optimization_summary.md"), "w") as f:
            f.write("# Hyperparameter Optimization Methods Comparison\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n## Visualization Results\n\n")
            f.write("See the following files for detailed visualizations:\n")
            f.write("- best_performance_comparison.png: Bar chart of best performance by method\n")
            f.write("- convergence_comparison.png: Line plot showing convergence over trials\n")
            f.write("- parameter_comparison.png: Heatmap of parameter values across methods\n")
            f.write("- key_parameters_comparison.png: Bar chart of key parameter values\n")
        
        print(f"Comparison complete! Results saved to {args.output_dir}")
    else:
        print("No results available for comparison")

if __name__ == "__main__":
    main()
