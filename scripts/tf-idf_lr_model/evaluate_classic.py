import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

rocessed_dir = "data/processed_classic"
model_dir = "saved_models/classic_ml"
model_name = "logistic_regression.joblib"
eval_dir = f"{model_dir}/evaluation_results"

def run_classic_evaluation():
    
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    model = joblib.load(f"{model_dir}/{model_name}")
    X_test = load_npz(f"{rocessed_dir}/X_test_tfidf.npz")
    y_true = np.load(f"{rocessed_dir}/y_test.npy")
    
    y_pred = model.predict(X_test)

    raw_datasets = load_dataset("mteb/banking77")
    
    raw_train = raw_datasets['train']
    mapping_df = pd.DataFrame({
        'label': raw_train['label'],
        'label_text': raw_train['label_text']
    }).drop_duplicates().set_index('label').sort_index()
    label_names = mapping_df['label_text'].tolist()
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    report_path = f"{eval_dir}/evaluation_report_classic.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(25, 25))
    sns.heatmap(cm, annot=False, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Ma trận nhầm lẫn (Classic ML - Logistic Regression)')
    cm_path = f"{eval_dir}/confusion_matrix_classic.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Misclassified Examples
    test_texts = raw_datasets['test']['text']
    df = pd.DataFrame({
        'text': test_texts,
        'true_label_id': y_true,
        'predicted_label_id': y_pred
    })
    df['true_label'] = df['true_label_id'].apply(lambda x: label_names[x])
    df['predicted_label'] = df['predicted_label_id'].apply(lambda x: label_names[x])
    
    misclassified_df = df[df['true_label_id'] != df['predicted_label_id']]
    misclassified_path = f"{eval_dir}/misclassified_examples_classic.csv"
    misclassified_df[['text', 'true_label', 'predicted_label']].to_csv(misclassified_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    run_classic_evaluation()