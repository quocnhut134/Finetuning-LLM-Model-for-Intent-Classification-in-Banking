from datasets import load_from_disk
import sys

def load_tokenized_data(processed_data_dir: str):
    try:
        tokenized_datasets = load_from_disk(processed_data_dir)
        return tokenized_datasets
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)