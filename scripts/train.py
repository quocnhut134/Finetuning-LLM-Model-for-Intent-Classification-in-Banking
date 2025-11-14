import argparse
import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer

from config import default_model_name, processed_data_dir_template, output_dir_template, training_args, early_stopping_patience
from data_utils import load_tokenized_data
from model_utils import load_model
from metrics import compute_metrics

def main(args):
    model_name_safe = args.model_name.replace('/', '_')
    processed_data_dir = processed_data_dir_template.format(model_name_safe=model_name_safe)
    output_dir = output_dir_template.format(model_name_safe=model_name_safe)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[INFO] Use Device: {device.upper()}")

    tokenized_datasets = load_tokenized_data(processed_data_dir)
    model = load_model(args.model_name, device)
    if model is None: return

    training_args_dict = training_args.copy()
    training_args_dict['output_dir'] = output_dir
    training_args_dict['num_train_epochs'] = args.epochs
    training_args_dict['per_device_train_batch_size'] = args.batch_size
    training_args_dict['per_device_eval_batch_size'] = args.batch_size
    training_args_dict['fp16'] = torch.cuda.is_available() 
    training_args_dict['logging_dir'] = f"{output_dir}/logs"
    
    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )
    
    trainer.train()
    
    trainer.save_model(output_dir)
    
    print(f"Best model is saved at: {output_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Intent Classification Model for Banking77.")
    parser.add_argument("--model_name", type=str, default=default_model_name, help="Model name on HuggingFace.")
    parser.add_argument("--epochs", type=int, default=training_args['num_train_epochs'], help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=training_args['per_device_train_batch_size'], help="Batch size.")

    args = parser.parse_args()
    main(args)