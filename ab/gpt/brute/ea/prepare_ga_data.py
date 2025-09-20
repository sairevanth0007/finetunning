# prepare_ga_data.py
import os
import json
import pandas as pd
from pathlib import Path

# Paths to your existing data
GA_ARCHITECTURE_DIR = "ga_architecture"  # Where your ga-alexnet-X.py files are
STATS_DIR = "stats"  # Where your stats folders are

# Output path for the fine-tuning dataset
OUTPUT_DIR = Path("nn-gpt/out/nngpt/dataset")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prepare a list to store all training examples
training_examples = []

# First, find all stats folders that match the pattern
stats_folders = [f for f in os.listdir(STATS_DIR) 
                if f.startswith("img-classification_cifar-10_acc_ga-alexnet-")]

print(f"Found {len(stats_folders)} stats folders to process")

# Process each stats folder
for stats_folder in stats_folders:
    try:
        # Extract the architecture number from the folder name
        # Example: "img-classification_cifar-10_acc_ga-alexnet-0" -> 0
        arch_num = stats_folder.split("ga-alexnet-")[-1]
        
        # Check if there's a corresponding architecture file
        arch_file = f"ga-alexnet-{arch_num}.py"
        arch_path = os.path.join(GA_ARCHITECTURE_DIR, arch_file)
        
        if not os.path.exists(arch_path):
            print(f"Warning: Architecture file not found for {stats_folder} (looking for {arch_file})")
            continue
            
        # Find the highest epoch file in this stats folder
        stats_folder_path = os.path.join(STATS_DIR, stats_folder)
        epoch_files = [f for f in os.listdir(stats_folder_path) if f.endswith('.json')]
        
        if not epoch_files:
            print(f"Warning: No epoch files found in {stats_folder_path}")
            continue
            
        # Sort by epoch number (assuming filenames are like "0.json", "1.json", etc.)
        epoch_files.sort(key=lambda x: int(x.split('.')[0]))
        latest_epoch_file = epoch_files[-1]  # Get the last epoch
        
        # Read the stats from the latest epoch
        with open(os.path.join(stats_folder_path, latest_epoch_file), 'r') as f:
            stats = json.load(f)
            
        # The stats might be a list with one element (as in previous examples)
        if isinstance(stats, list) and len(stats) > 0:
            stats = stats[0]
            
        # Read the architecture code
        with open(arch_path, 'r') as f:
            code = f.read()
        
        # Create a training example
        example = {
            "input": f"Generate a CNN for CIFAR-10 with approximately {stats['accuracy']*100:.1f}% accuracy",
            "output": code
        }
        
        # Add hyperparameters as context
        hp_context = f"Hyperparameters: lr={stats['lr']}, momentum={stats['momentum']}, dropout={stats['dropout']}"
        example["input"] = f"{example['input']}. {hp_context}"
        
        training_examples.append(example)
        print(f"Successfully processed {arch_file} with stats from {stats_folder}/{latest_epoch_file}")
        
    except Exception as e:
        print(f"Error processing {stats_folder}: {e}")

# Save the training examples
if training_examples:
    with open(OUTPUT_DIR / "training_data.json", "w") as f:
        json.dump(training_examples, f, indent=2)
    print(f"\nPrepared {len(training_examples)} training examples")
else:
    print("\nNo training examples were prepared. Check your file structure.")

print(f"Training data saved to: {OUTPUT_DIR / 'training_data.json'}")