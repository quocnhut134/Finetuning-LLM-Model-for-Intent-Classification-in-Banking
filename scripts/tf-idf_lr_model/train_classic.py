import os
import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression

processed_dir = "data/processed_classic"
model_dir = "saved_models/classic_ml"
model_name= "logistic_regression.joblib"

def run_classic_training():
    X_train = load_npz(f"{processed_dir}/X_train_tfidf.npz")
    y_train = np.load(f"{processed_dir}/y_train.npy")
    
    model = LogisticRegression(
        max_iter=1000, 
        solver='saga',
        n_jobs=-1, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = f"{model_dir}/{model_name}"
    joblib.dump(model, model_path)

if __name__ == "__main__":
    run_classic_training()