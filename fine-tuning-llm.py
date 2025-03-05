# ==================================
# CISC 691 - NextGenAI
# Sample Fine tuning code for sentiment analysis
# ==================================

# standard python libraries
import logging
import json
import random
from datetime import datetime
from collections import Counter
from pathlib import Path

# HuggingFace libraries
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# sklearn, py-torch  libraries
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from scipy.special import softmax
import numpy as np


# --------------------------------------
#  Set up python logging to console and a log file
# --------------------------------------
def configure_logging(loglevel: str = "DEBUG"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"fine_tune_{current_time}.log"
    numeric_level = getattr(logging, loglevel.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_filename),  # File output
        ]
    )

LOG_LEVEL = "DEBUG"
configure_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)



# --------------------------------------
#  constants
# --------------------------------------
# location of intermediate results and final model
RESULTS_DIR = Path("./results").resolve()

# model and dataset names to download from HuggingFace
BASE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# BASE_MODEL_NAME = "distilbert-base-uncased-distilled-squad"
SENTIMENT_DATASET_NAME = "imdb"

# --------------------------------------
#  Optimization tweaks: refer to torch documentation
# --------------------------------------

# Ensure MPS (Apple Silicon GPU) is used
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --------------------------------------
#  Defines a class for storing the sentiment datasets used for training
# --------------------------------------
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val, dtype=torch.long) for key, val in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].to(device) for key, val in self.encodings.items()}  # Move data to MPS
        item["labels"] = self.labels[idx].to(device)
        return item

# --------------------------------------
#  Load the base model: downloads from HF, reduces its size, then stores locally
# --------------------------------------
def load_base_model(model_name = BASE_MODEL_NAME):
    logger.info(f"Loading model...{model_name}")
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Reduce model size
    # config = DistilBertConfig(
    #     n_layers=3,                 # Default is 6 → Reduce to 3 layers
    #     hidden_size=384,            # Default is 768 → Reduce to 384
    #     intermediate_size=768,      # Default is 3072 → Reduce to 768
    #     num_labels=2                # Binary classification (Positive/Negative)
    # )

    # Load the reduced model
    # base_model = DistilBertForSequenceClassification(config)
    base_model.to("mps")            # Move model to Apple GPU

    base_model.save_pretrained(f"{RESULTS_DIR}/base_model")
    tokenizer.save_pretrained(f"{RESULTS_DIR}/base_model")

    logger.info("done")

    return base_model, tokenizer

# --------------------------------------
#  Load the sentiment dataset: downloads from HF, extracts a subset then stores locally
# --------------------------------------
def load_sentiment_dataset(tokenizer, dataset_name = SENTIMENT_DATASET_NAME):
    logger.info(f"Loading dataset...{dataset_name}")
    dataset = load_dataset(dataset_name)

    # Convert dataset to list to filter by label
    train_data = list(dataset["train"])
    test_data = list(dataset["test"])

    # Separate positive and negative samples
    positive_train = [ex for ex in train_data if ex["label"] == 1]
    negative_train = [ex for ex in train_data if ex["label"] == 0]

    positive_test = [ex for ex in test_data if ex["label"] == 1]
    negative_test = [ex for ex in test_data if ex["label"] == 0]

    # Select equal numbers of positive and negative examples
    subset_train = random.sample(positive_train, 2000) + random.sample(negative_train, 2000)
    subset_test = random.sample(positive_test, 500) + random.sample(negative_test, 500)

    # Shuffle to mix classes
    random.shuffle(subset_train)
    random.shuffle(subset_test)

    # Check new label distribution
    print("New Train Distribution:", Counter([ex["label"] for ex in subset_train]))
    print("New Test Distribution:", Counter([ex["label"] for ex in subset_test]))

    # Convert back to Dataset format
    subset_train = Dataset.from_list(subset_train)
    subset_test = Dataset.from_list(subset_test)

    # Tokenization function
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

    # Tokenize dataset
    tokenized_train = subset_train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test = subset_test.map(tokenize_function, batched=True, remove_columns=["text"])

    logger.info(f"Saving sentiment dataset...")
    tokenized_train.save_to_disk(f"{RESULTS_DIR}/imdb_train_subset")
    tokenized_test.save_to_disk(f"{RESULTS_DIR}/imdb_test_subset")


# --------------------------------------
#  Run the training step
# --------------------------------------
def train_model(base_model, tokenizer, train_dataset, test_dataset):
    logger.info(f"Starting...")

    # Get the encodings for the training and test datasets
    train_encodings = {key: train_dataset[key] for key in train_dataset.features if key != "label"}
    train_labels = train_dataset["label"]
    train_data = SentimentDataset(train_encodings, train_labels)

    test_encodings = {key: test_dataset[key] for key in test_dataset.features if key != "label"}
    test_labels = test_dataset["label"]
    test_data = SentimentDataset(test_encodings, test_labels)

    # Set up the arguments. Each parm can impact time to train and the performance of the resulting model
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR.as_posix(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=6,
        weight_decay=0.01,
        load_best_model_at_end=True,
        bf16=True,
    )

    # Define a function to be used by the trainer, to see how it is progressing
    def compute_metrics(pred):
        predictions = torch.argmax(torch.tensor(pred.predictions), dim=-1).cpu().numpy()
        labels = pred.label_ids
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc, "f1": f1}

    # Create the Trainer class to be executed
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )
    logger.info(f"Running the training... This may take a while...")
    trainer.train()

    logs = trainer.state.log_history
    print("Training Logs:", logs)

    # Print the metrics for evaluating the training
    logger.info(f"Running the evaluation...")
    trainer.compute_metrics = compute_metrics
    results = trainer.evaluate(metric_key_prefix="eval")

    print(results)

    # Save the fine-tuned model and tokenizer for use later
    logger.info(f"Saving results...")
    trainer.save_model(f"{RESULTS_DIR}/fine_tuned_model")
    tokenizer.save_pretrained(f"{RESULTS_DIR}/fine_tuned_model")

    logger.info(f"Done...")

# --------------------------------------
# Function to predict sentiment
# --------------------------------------
def predict_sentiment(text, model, tokenizer, actual_sentiment):

    # If optimizing, make sure the data is on the GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device

    # run the encoded text through the model, normalize the probability, and translate to our sentiment label
    output = model(**inputs)
    scores = output.logits[0].detach().cpu().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)[::-1]
    config = AutoConfig.from_pretrained(f"{RESULTS_DIR}/base_model")
    sentiment = config.id2label[ranking[0]]
    best_score = scores[ranking[0]]
    logger.info(f"Predict: [{text:<100}] : \t Sentiment: {sentiment}, Score: {best_score:.2f}. Actual sentiment: {actual_sentiment}")
    return sentiment == actual_sentiment


def step_01_download_base_model():
    load_base_model()


def step_02_download_sentiment_dataset():
    tokenizer = AutoTokenizer.from_pretrained(f"{RESULTS_DIR}/base_model")
    load_sentiment_dataset(tokenizer)


def step_03_train_model():
    base_model = AutoModelForSequenceClassification.from_pretrained(f"{RESULTS_DIR}/base_model")
    tokenizer = AutoTokenizer.from_pretrained(f"{RESULTS_DIR}/base_model")

    train_dataset = load_from_disk(f"{RESULTS_DIR}/imdb_train_subset")
    test_dataset = load_from_disk(f"{RESULTS_DIR}/imdb_test_subset")

    train_model(base_model, tokenizer, train_dataset, test_dataset)

def step_04_predict_sentiment(model_type= "fine_tuned" ):
    logger.info(f"Using model: {model_type}...")
    test_sentences = [
        ("I loved this movie!", "positive"), # Should be Positive
        ("This was the worst experience ever.", "negative"),  # Should be Negative
        ("It was okay, but not great.",  "neutral") # Neutral (may vary)
    ]

    if model_type == "fine_tuned":
        model = AutoModelForSequenceClassification.from_pretrained(f"{RESULTS_DIR}/fine_tuned_model")
        tokenizer = AutoTokenizer.from_pretrained(f"{RESULTS_DIR}/fine_tuned_model")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(f"{RESULTS_DIR}/base_model")
        tokenizer = AutoTokenizer.from_pretrained(f"{RESULTS_DIR}/base_model")

    correct_predictions = 0
    for sentence, actual_sentiment in test_sentences:
        if predict_sentiment(sentence, model, tokenizer, actual_sentiment):
            correct_predictions += 1
    total_predictions = len(test_sentences)

    json_filename = f"{RESULTS_DIR}/sample_sentences_2.json"
    with open(json_filename, "r", encoding="utf-8") as file:
        sentences_data = json.load(file)
        for entry in sentences_data:
            sentence = entry["sentence"]
            if entry["sentiment"] > 0:
                actual_sentiment = "positive"
            elif entry["sentiment"] < 0:
                actual_sentiment = "negative"
            else:
                actual_sentiment = "neutral"
            if predict_sentiment(sentence, model, tokenizer, actual_sentiment):
                correct_predictions += 1
        total_predictions += len(sentences_data)
    logger.info(f"Accuracy on sample_sentences_2.json: {correct_predictions / total_predictions * 100:.2f}%")


if __name__ == "__main__":
    logger.info(f"Starting...")

    # uncomment as needed
    # step_01_download_base_model()

    # step_02_download_sentiment_dataset()

    # step_03_train_model()

    step_04_predict_sentiment("base")
    step_04_predict_sentiment("fine_tuned")

    logger.info(f"Done...")

