python scripts/preprocess.py --model_name "distilbert/distilbert-base-uncased" --max_length 128
python scripts/train.py --model_name "distilbert/distilbert-base-uncased" --epochs 20 | Tee-Object -FilePath "logs/distilbert_distilbert-base-uncased.log"
python scripts/model_evaluate.py --model_dir "saved_models/distilbert_distilbert-base-uncased-finetuned-banking77" --processed_data_dir "data/processed/banking77-tokenized-distilbert_distilbert-base-uncased"
