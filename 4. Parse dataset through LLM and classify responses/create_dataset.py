#!/usr/bin/env python3

"""
Combines a list of "good" prompts from an Excel file and a list of "malicious" 
prompts from a CSV file to create a new labeled dataset.

This script is intended for creating datasets for safety and security modeling,
such as training a classifier to detect malicious inputs.

Requirements:
    pip install pandas openpyxl

Example Usage:
    python create_prompt_dataset.py
"""

import pandas as pd
from typing import List, Dict, Any

def load_data(
    file_path: str, column_name: str, is_excel: bool = False
) -> List[str]:
    """Loads a column from a CSV or Excel file into a list of strings."""
    try:
        if is_excel:
            # Requires openpyxl to be installed
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
            
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in {file_path}.")
            print(f"Available columns are: {list(df.columns)}")
            exit(1)
            
        # Drop rows where the specified column is empty and convert to string
        prompts = df[column_name].dropna().astype(str).tolist()
        print(f"Successfully loaded {len(prompts)} prompts from {file_path}.")
        return prompts
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        exit(1)

def create_labeled_dataset(
    good_prompts: List[str], malicious_prompts: List[str]
) -> pd.DataFrame:
    """
    Creates a labeled dataset from good and malicious prompt lists.
    
    - Good prompts are added with malicious_flag = 0.
    - Malicious prompts are created by prefixing a good prompt with a malicious
      prompt. Each malicious prompt is used exactly once.
    """
    new_dataset: List[Dict[str, Any]] = []
    
    # 1. Add all good prompts with malicious_flag = 0
    for prompt in good_prompts:
        new_dataset.append({"prompt": prompt, "malicious": 0})
        
    num_good_prompts = len(good_prompts)
    num_malicious_prompts = len(malicious_prompts)
    
    if num_good_prompts == 0:
        print("Warning: No good prompts loaded. Cannot generate prefixed malicious prompts.")
        return pd.DataFrame(new_dataset)

    print(f"Generating {num_malicious_prompts} prefixed malicious prompts...")

    # 2. Add prefixed malicious prompts with malicious_flag = 1
    for i, malicious_prefix in enumerate(malicious_prompts):
        # Get a good prompt to use as the base.
        # We cycle through the good prompts if there are more malicious
        # prefixes than good prompts.
        base_good_prompt = good_prompts[i % num_good_prompts]
        
        # Combine the prefix and the base prompt
        # Adding a space for separation is generally a good idea.
        new_malicious_prompt = f"{malicious_prefix} {base_good_prompt}"
        
        new_dataset.append({"prompt": new_malicious_prompt, "malicious": 1})
        
    return pd.DataFrame(new_dataset)

def main():
    """Main function to run the script with hard-coded values."""

    # --- Hard-coded variables ---
    good_file_path = "safe prompts.xlsx"
    good_col_name = "Prompt "
    malicious_file_path = "jailbreak prompts.csv"
    malicious_col_name = "Prompt"
    output_file_path = "my_labeled_dataset.csv"

    print(f"Using hard-coded file paths:")
    print(f"  Good prompts file: {good_file_path} (Column: '{good_col_name}')")
    print(f"  Malicious prompts file: {malicious_file_path} (Column: '{malicious_col_name}')")
    print(f"  Output file: {output_file_path}\n")
    
    # 1. Load data
    good_prompts = load_data(
        good_file_path, good_col_name, is_excel=True
    )
    malicious_prompts = load_data(
        malicious_file_path, malicious_col_name, is_excel=False
    )
    
    if not good_prompts or not malicious_prompts:
        print("Error: One of the input files resulted in an empty list. Exiting.")
        return

    # 2. Create dataset
    print("Creating new dataset...")
    output_df = create_labeled_dataset(good_prompts, malicious_prompts)
    
    # 3. Shuffle the dataset
    # Shuffling is good practice for training datasets.
    print("Shuffling dataset...")
    output_df = output_df.sample(frac=1).reset_index(drop=True)
    
    # 4. Save to CSV
    try:
        output_df.to_csv(output_file_path, index=False)
        print(f"\nSuccessfully created dataset with {len(output_df)} entries.")
        print(f"Saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    main()

