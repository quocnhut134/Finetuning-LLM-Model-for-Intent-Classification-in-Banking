from transformers import AutoModelForSequenceClassification
import torch

def load_model(model_name: str, device: str):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=77)
        model.to(device)
        return model
    except Exception as e:
        print(e)
        return None