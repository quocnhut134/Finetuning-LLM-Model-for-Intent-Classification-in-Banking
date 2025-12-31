import os
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

processed_dir = "data/processed_classic"
max_features = 10000  

def run_classic_preprocessing():
    raw_datasets = load_dataset("mteb/banking77")
    
    X_train_raw = raw_datasets['train']['text']
    y_train = np.array(raw_datasets['train']['label'])
    
    X_test_raw = raw_datasets['test']['text']
    y_test = np.array(raw_datasets['test']['label'])
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',  
        ngram_range=(1, 2)    
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    save_npz(f"{processed_dir}/X_train_tfidf.npz", X_train_tfidf)
    save_npz(f"{processed_dir}/X_test_tfidf.npz", X_test_tfidf)
    
    np.save(f"{processed_dir}/y_train.npy", y_train)
    np.save(f"{processed_dir}/y_test.npy", y_test)
    
    joblib.dump(vectorizer, f"{processed_dir}/tfidf_vectorizer.joblib")

if __name__ == "__main__":
    run_classic_preprocessing()