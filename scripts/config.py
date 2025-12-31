default_model_name = "distilbert/distilbert-base-uncased"
processed_data_dir_template = "./data/processed/banking77-tokenized-{model_name_safe}"
output_dir_template = "saved_models/{model_name_safe}-finetuned-banking77"

training_args = {
    "num_train_epochs": 5,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "fp16": True, 
    "logging_strategy": "steps",
    "logging_steps": 100,
}

early_stopping_patience = 3