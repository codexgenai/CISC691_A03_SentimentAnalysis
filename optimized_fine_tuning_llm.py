import os
import logging
import json
import random
from datetime import datetime
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
)
from datasets import Dataset as HFDataset, load_dataset, load_from_disk

# Disable W&B
os.environ["WANDB_DISABLED"] = "true"

# --------------------------------------
#  Configure Logging
# --------------------------------------
def configure_logging(loglevel: str = "DEBUG"):
    """
    Configures logging to log messages to both the console and a log file.
    The log level can be adjusted via the `loglevel` parameter.
    """
    log_filename = f"fine_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, loglevel.upper(), None),
        format="[%(asctime)s] %(levelname)s %(name)s:%(funcName)s:%(lineno)d - %(message)s [%(relativeCreated)d ms]",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)],
    )

configure_logging("DEBUG")
logger = logging.getLogger(__name__)

# --------------------------------------
#  Constants
# --------------------------------------
RESULTS_DIR = Path("./results").resolve()
BASE_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
SENTIMENT_DATASET_NAME = "imdb"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --------------------------------------
#  Dataset Class
# --------------------------------------
class SentimentDataset(Dataset):
    """
    Custom PyTorch Dataset class to handle tokenized text data and labels for sentiment analysis.
    
    Attributes:
        encodings (dict): Dictionary containing tokenized input data.
        labels (Tensor): Tensor containing sentiment labels.
    
    Methods:
        __len__: Returns the number of samples in the dataset.
        __getitem__: Retrieves an item at the specified index, returning tokenized input and label.
    """
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val, dtype=torch.long) for key, val in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# --------------------------------------
#  Load Model
# --------------------------------------
def load_base_model():
    """
    Loads a pre-trained sentiment analysis model from Hugging Face if it does not already exist locally.
    Freezes all but the last two layers to allow fine-tuning while preserving learned weights.
    """
    model_path = RESULTS_DIR / "base_model_2"
    if model_path.exists():
        logger.info("Model already exists. Loading from local storage.")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model.to(device), tokenizer
    
    logger.info(f"Downloading model: {BASE_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # Freeze all layers except the last 2
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.layer[-2:].parameters():
        param.requires_grad = True
    
    model.to(device)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info("Model downloaded and saved.")
    return model, tokenizer

# --------------------------------------
#  Load Dataset
# --------------------------------------
def load_sentiment_dataset(tokenizer):
    """
    Downloads and processes the IMDB sentiment dataset.
    
    - Ensures a balanced subset of positive and negative samples.
    - Tokenizes the dataset for model training and evaluation.
    - Saves the processed dataset to disk.
    """
    logger.info(f"Loading dataset: {SENTIMENT_DATASET_NAME}")
    dataset = load_dataset(SENTIMENT_DATASET_NAME)
    
    def select_balanced_subset(data, size):
        """Selects an equal number of positive and negative samples from the dataset."""
        pos = [d for d in data if d["label"] == 1]
        neg = [d for d in data if d["label"] == 0]
        return random.sample(pos, size // 2) + random.sample(neg, size // 2)
    
    subset_train, subset_test = map(
        lambda d: HFDataset.from_list(select_balanced_subset(list(d), 2000 if "train" in d else 500)),
        [dataset["train"], dataset["test"]]
    )
    
    def tokenize(example):
        """Tokenizes the text data using the provided tokenizer."""
        return tokenizer(example["text"], padding="max_length", truncation=True)
    
    train_data, test_data = map(
        lambda d: d.map(tokenize, batched=True, remove_columns=["text"]),
        [subset_train, subset_test]
    )
    
    train_data.save_to_disk(RESULTS_DIR / "imdb_train_subset")
    test_data.save_to_disk(RESULTS_DIR / "imdb_test_subset")
    logger.info("Dataset loaded and saved.")


# --------------------------------------
#  Train Model
# --------------------------------------
def train_model(model, tokenizer, train_dataset, test_dataset):
    """
    Trains the sentiment classification model using the given datasets.

    Args:
        model: Pretrained transformer model for sequence classification.
        tokenizer: Tokenizer corresponding to the model.
        train_dataset: Tokenized training dataset.
        test_dataset: Tokenized test dataset.
    """
    logger.info("Starting training...")

    def convert_to_torch_dataset(dataset):
        """Converts Hugging Face dataset into a PyTorch Dataset."""
        encodings = {key: dataset[key] for key in dataset.features if key != "label"}
        return SentimentDataset(encodings, dataset["label"])

    train_data, test_data = map(convert_to_torch_dataset, [train_dataset, test_dataset])

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs=6,
        weight_decay=0.01,
        load_best_model_at_end=True,
        bf16=True if torch.cuda.is_available() else False,
    )

    def compute_metrics(pred):
        """Computes accuracy and F1-score for evaluation."""
        predictions = torch.argmax(torch.tensor(pred.predictions), dim=-1).cpu().numpy()
        return {"accuracy": accuracy_score(pred.label_ids, predictions), "f1": f1_score(pred.label_ids, predictions, average="weighted")}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    trainer.save_model(RESULTS_DIR / "fine_tuned_model")
    tokenizer.save_pretrained(RESULTS_DIR / "fine_tuned_model")
    logger.info("Training complete.")

def infer_on_json(json_path, model_path=RESULTS_DIR / "fine_tuned_model"):
    """
    Performs inference on a custom JSON file containing text data.
    
    Args:
        json_path (str): Path to the JSON file containing text data.
        model_path (str, optional): Path to the trained model. Defaults to fine-tuned model.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        texts = [entry["sentence"] for entry in data]
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error loading JSON file: {e}")
        return

    logger.info("Loading fine-tuned model for inference...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    predictions = []
    for text, entry in zip(texts, data):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        sentiment = "Positive" if torch.argmax(F.softmax(logits, dim=-1)).item() == 1 else "Negative"
        entry["predicted_sentiment"] = sentiment
        predictions.append(entry)

    output_path = json_path.replace(".json", "_predictions.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=4)
    
    logger.info(f"Inference complete. Predictions saved to {output_path}")
    return predictions

# --------------------------------------
#  Main Execution
# --------------------------------------
if __name__ == "__main__":
    logger.info("Starting sentiment analysis fine-tuning pipeline...")
    model, tokenizer = load_base_model()
    load_sentiment_dataset(tokenizer)
    train_dataset = load_from_disk(RESULTS_DIR / "imdb_train_subset")
    test_dataset = load_from_disk(RESULTS_DIR / "imdb_test_subset")
    train_model(model, tokenizer, train_dataset, test_dataset)
    logger.info("Pipeline execution complete.")
