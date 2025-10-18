import argparse
import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def run_evaluation(model_dir: str, processed_data_dir: str):
    
    output_artifacts_dir = f"{model_dir}/evaluation_results"
    if not os.path.exists(output_artifacts_dir):
        os.makedirs(output_artifacts_dir)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Use Device: {device.upper()}")

    # Model and Tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    except Exception as e:
        print(e)
        return

    # Loading Test Dataset
    test_dataset = load_from_disk(processed_data_dir)["test"]
    
    if 'label_text' in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns(["label_text"])

    raw_datasets = load_dataset("mteb/banking77")
    label_names = sorted(list(set(raw_datasets['train']['label_text'])))
        
    # Prediction
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=16, 
        collate_fn=data_collator
    )
    model.eval()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad(): 
        for batch in tqdm(test_dataloader, desc="Predicting"):
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    # Report
    report = classification_report(all_true_labels, all_predictions, target_names=label_names, digits=4)
    report_path = f"{output_artifacts_dir}/evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(25, 25))
    sns.heatmap(cm, annot=False, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    cm_path = f"{output_artifacts_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Misclassified Examples
    test_texts = raw_datasets['test']['text']
    df = pd.DataFrame({
        'text': test_texts,
        'true_label_id': all_true_labels,
        'predicted_label_id': all_predictions
    })
    df['true_label'] = df['true_label_id'].apply(lambda x: label_names[x])
    df['predicted_label'] = df['predicted_label_id'].apply(lambda x: label_names[x])
    
    misclassified_df = df[df['true_label_id'] != df['predicted_label_id']]
    misclassified_path = f"{output_artifacts_dir}/misclassified_examples.csv"
    misclassified_df[['text', 'true_label', 'predicted_label']].to_csv(misclassified_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="saved_models/distilbert_distilbert-base-uncased-finetuned-banking77",
        help="Path to the directory containing the best model."
    )
    parser.add_argument(
        "--processed_data_dir", 
        type=str, 
        default="data/processed/banking77-tokenized-distilbert_distilbert-base-uncased",
        help="Path to the preprocessed data."
    )
    args = parser.parse_args()
    run_evaluation(args.model_dir, args.processed_data_dir)