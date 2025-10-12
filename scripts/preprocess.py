import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv


load_dotenv()

def run_preprocessing(model_name: str, max_length: int):
    
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    output_dir = f"./data/processed/banking77-tokenized-{model_name.replace('/', '_')}"
    
    print("="*80)
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Max_length: {max_length}")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    raw_datasets = load_dataset("mteb/banking77", token=hf_token)
    print(raw_datasets)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=max_length
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function, 
        batched=True, 
        num_proc=8
    )

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
        
    # print(tokenized_datasets["train"][0])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenized_datasets.save_to_disk(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="distilbert/distilbert-base-uncased", 
        help="Model Name"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=64, 
        help="Max length for tokenization"
    )
    
    args = parser.parse_args()
    
    run_preprocessing(args.model_name, args.max_length)