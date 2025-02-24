import pandas as pd
import os

# Dataset Path (Update this if needed)
data_path = "data/"

# Read dataset files
def load_data(source_file, target_file):
    with open(source_file, 'r', encoding='utf-8') as src, open(target_file, 'r', encoding='utf-8') as tgt:
        sources = src.readlines()
        targets = tgt.readlines()
    return sources, targets

# Convert data to CSV format
def create_dataframe():
    datasets = {
        "train": ("train.wp_source", "train.wp_target"),
        "valid": ("valid.wp_source", "valid.wp_target"),
        "test": ("test.wp_source", "test.wp_target"),
    }

    output_dir = r"data/csv"  # Save CSVs here
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    for split, (source_file, target_file) in datasets.items():
        sources, targets = load_data(os.path.join(data_path, source_file),
                                     os.path.join(data_path, target_file))
        df = pd.DataFrame({"prompt": sources, "story": targets})
        csv_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

# Run preprocessing
create_dataframe()
