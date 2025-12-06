# nlp_final_proj.py
"""
NLP Final Project - FABSA ABSA

This file will eventually contain:
- Dataset & DataLoader definitions
- Model definition
- Training / Evaluation / Explainability

Right now we are ONLY doing:
Step 1: Load FABSA + inspect basic structure (EDA).
"""

# ============================
# IMPORTS
# ============================
from datasets import load_dataset
from pprint import pprint
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score

# ============================
# CONFIG
# ============================
MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# DATASET & DATALOADER
# ============================

def load_fabsa_raw():
    """
    Load the FABSA dataset from Hugging Face.
    Returns a DatasetDict with splits: train / validation / test. (datasets were split on Hugging Face already)
    """
    print("Loading FABSA dataset from Hugging Face...")
    fabsa = load_dataset("jordiclive/FABSA")
    return fabsa


def inspect_fabsa_basic(fabsa):
    """
    Returns
    - What splits we have
    - How many examples per split
    - What columns exist
    - What one example looks like
    """
    # Dataset info (splits and sizes)
    print("\n=== Dataset object ===")
    print(fabsa)

    # Look at the columns in the train split
    print("\n=== Train split columns ===")
    print(fabsa["train"].column_names)

    # A single example
    print("\n=== One example from train split ===")
    example = fabsa["train"][0]
    pprint(example)

    # The text field length and a small snippet
    text = example["text"]
    print("\nText length (chars):", len(text))
    print("Review text snippet:")
    print(text[:300], "...\n")

    return example

def analyze_fabsa_labels_basic(fabsa):
    """
    basic analysis of labels in the train split.
    - How many labels per review
    - What sentiments exist and how frequent
    - How many unique aspect categories
    """
    train = fabsa["train"]

    label_counts = []
    sentiment_counter = Counter()
    aspect_counter = Counter()

    for ex in train:
        labels = ex["labels"]
        label_counts.append(len(labels))

        for aspect, sentiment in labels:
            sentiment_counter[sentiment] += 1
            aspect_counter[aspect] += 1

    label_counts = np.array(label_counts)

    print("\n=== Basic label stats (train split) ===")
    print(f"Number of reviews in train: {len(train)}")
    print(f"Labels per review: min: {label_counts.min()}, "
          f"max: {label_counts.max()}, "
          f"mean: {label_counts.mean():.2f}")

    print("\nSentiment distribution (train):")
    for sent, count in sentiment_counter.most_common():
        print(f"  {sent:8s} : {count}")

    print(f"\nNumber of unique aspect categories (train): {len(aspect_counter)}")
    print("Some example aspect categories:")
    for aspect, count in list(aspect_counter.most_common(10)):
        print(f"  {aspect}  (count: {count})")


def analyze_aspect_sentiment_distribution(fabsa):
    """
    For each aspect, compute sentiment counts.
    """
    from collections import defaultdict, Counter

    train = fabsa["train"]

    aspect_sentiment_map = defaultdict(Counter)

    for ex in train:
        for aspect, sentiment in ex["labels"]:
            aspect_sentiment_map[aspect][sentiment] += 1

    print("\n=== Aspect-level sentiment distribution (train split) ===")
    for aspect, counter in aspect_sentiment_map.items():
        print(f"\nAspect: {aspect}")
        for sent, count in counter.items():
            print(f"  {sent:8s}: {count}")

def analyze_industry_distribution(fabsa):
    """
    Analyze how reviews and sentiments are distributed across industries.
    """

    train = fabsa["train"]

    from collections import defaultdict, Counter
    industry_review_count = Counter()
    industry_label_count = Counter()
    industry_sentiment_map = defaultdict(Counter)

    for ex in train:
        industry = ex["industry"]
        industry_review_count[industry] += 1

        labels = ex["labels"]
        industry_label_count[industry] += len(labels)

        for aspect, sentiment in labels:
            industry_sentiment_map[industry][sentiment] += 1

    print("\n=== Industry-level review counts (train split) ===")
    for ind, count in industry_review_count.items():
        print(f"{ind:30s} : {count}")

    print("\n=== Industry-level avg labels per review ===")
    for ind in industry_review_count:
        avg = industry_label_count[ind] / industry_review_count[ind]
        print(f"{ind:30s} : {avg:.2f}")

    print("\n=== Industry-level sentiment distribution ===")
    for ind, counter in industry_sentiment_map.items():
        print(f"\nIndustry: {ind}")
        for sent, count in counter.items():
            print(f"  {sent:8s}: {count}")

def flatten_fabsa(fabsa):
    """
    Convert FABSA's multi-label review format into a flat list of samples:
    Each sample = one (text, aspect, sentiment, industry, data_source).
    """
    train = fabsa["train"]
    val = fabsa["validation"]
    test = fabsa["test"]

    def flatten_split(split):
        rows = []
        for ex in split:
            text = ex["text"]
            industry = ex["industry"]
            source = ex["data_source"]

            for aspect, sentiment in ex["labels"]:
                rows.append({
                    "text": text,
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "industry": industry,
                    "source": source
                })
        return rows

    train_rows = flatten_split(train)
    val_rows = flatten_split(val)
    test_rows = flatten_split(test)

    print("\n=== Flattened dataset sizes ===")
    print("Train:", len(train_rows))
    print("Val:  ", len(val_rows))
    print("Test: ", len(test_rows))

    return train_rows, val_rows, test_rows

def analyze_text_lengths(rows):
    """
    Analyze basic text length statistics (chars & tokens),
    overall and by sentiment.
    """
    import numpy as np
    from collections import defaultdict

    # Overall lengths
    lengths_chars = []
    lengths_tokens = []

    # By sentiment
    lengths_by_sent_chars = defaultdict(list)
    lengths_by_sent_tokens = defaultdict(list)

    for r in rows:
        text = r["text"]
        sent = r["sentiment"]

        n_chars = len(text)
        n_tokens = len(text.split())

        lengths_chars.append(n_chars)
        lengths_tokens.append(n_tokens)

        lengths_by_sent_chars[sent].append(n_chars)
        lengths_by_sent_tokens[sent].append(n_tokens)

    def summarize(arr):
        arr = np.array(arr)
        return {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    print("\n=== Text length stats (train, overall) ===")
    stats_chars = summarize(lengths_chars)
    stats_tokens = summarize(lengths_tokens)
    print("Characters:", stats_chars)
    print("Tokens:    ", stats_tokens)

    print("\n=== Text length stats by sentiment (train) ===")
    for sent in sorted(lengths_by_sent_chars.keys()):
        sc = summarize(lengths_by_sent_chars[sent])
        st = summarize(lengths_by_sent_tokens[sent])
        print(f"\nSentiment = {sent}")
        print("  Characters:", sc)
        print("  Tokens:    ", st)

# ============================
# LABEL MAPPING (sentiment - id)
# ============================

def build_sentiment_label_mapping(train_rows):
    """
    Build a mapping like:
      {'negative': 0, 'neutral': 1, 'positive': 2}
    based on the sentiments present in the TRAIN rows.
    """
    sentiments = sorted(list({row["sentiment"] for row in train_rows}))
    label2id = {s: i for i, s in enumerate(sentiments)}
    id2label = {i: s for s, i in label2id.items()}

    print("\n=== Sentiment label mapping ===")
    print(label2id)

    return label2id, id2label


# ============================
# PyTorch Dataset
# ============================

class FABSAAspectDataset(Dataset):
    """
    Dataset returning:
    - tokenized inputs for (text, aspect)
    - label_id (sentiment as int)
    - metadata: raw text, aspect, sentiment_str, industry, source
    """

    def __init__(self, rows, tokenizer, label2id, max_len=128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        text = item["text"]
        aspect = item["aspect"]
        sentiment_str = item["sentiment"]
        industry = item["industry"]
        source = item["source"]

        # Encode as: CLS text SEP aspect SEP
        encoded = self.tokenizer(
            text,
            aspect,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        label_id = self.label2id[sentiment_str]

        # Remove the leading batch dim from tokenizer output
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_id": torch.tensor(label_id, dtype=torch.long),

            # metadata
            "text": text,
            "aspect": aspect,
            "sentiment_str": sentiment_str,
            "industry": industry,
            "source": source,
        }


# ============================
# DATALOADER BUILDER
# ============================

def create_dataloaders(train_rows, val_rows, test_rows):
    """
    Given flattened rows, build:
    - tokenizer
    - sentiment label mapping
    - PyTorch Datasets
    - DataLoaders for train/val/test
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label2id, id2label = build_sentiment_label_mapping(train_rows)

    train_dataset = FABSAAspectDataset(train_rows, tokenizer, label2id, max_len=MAX_LEN)
    val_dataset = FABSAAspectDataset(val_rows, tokenizer, label2id, max_len=MAX_LEN)
    test_dataset = FABSAAspectDataset(test_rows, tokenizer, label2id, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, label2id, id2label, tokenizer

# ============================
# MODEL
# ============================

def build_model(label2id, id2label):
    """
    Build a sequence classification model (BERT) for 3-class sentiment.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)
    return model


def run_one_forward_pass(train_loader, model):
    """
    Take a single batch from the train loader and run it through the model.
    No training, just a sanity check.
    """
    model.eval()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label_id"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # this makes the model compute loss too
        )

    logits = outputs.logits  # shape: (batch_size, num_labels)
    loss = outputs.loss

    preds = torch.argmax(logits, dim=-1)

    print("\n=== Forward pass sanity check ===")
    print("Logits shape:", logits.shape)
    print("Loss:", float(loss.item()))
    print("Preds (first 10):", preds[:10].tolist())
    print("True  (first 10):", labels[:10].tolist())

# ============================
# TRAINING & EVALUATION LOOPS
# ============================

def train_one_epoch(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_id"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        loss = loss_fn(logits, labels)  # <-- weighted loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_examples += input_ids.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def eval_one_epoch(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_id"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += input_ids.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc

def get_predictions_and_labels(model, data_loader):
    """
    Run model on all batches in data_loader and collect
    predicted labels and true labels as CPU numpy arrays.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_id"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels

def evaluate_on_split(model, data_loader, split_name, id2label):
    """
    Compute accuracy and macro-F1 on a given split (train/val/test).
    """
    preds, labels = get_predictions_and_labels(model, data_loader)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    print(f"\n===== {split_name} metrics =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    # Detailed per-class report
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=target_names))


# ============================
# MAIN
# ============================

def main():
    fabsa = load_fabsa_raw()
    inspect_fabsa_basic(fabsa)
    analyze_fabsa_labels_basic(fabsa)
    analyze_aspect_sentiment_distribution(fabsa)
    analyze_industry_distribution(fabsa)


    # Flatten dataset
    train_rows, val_rows, test_rows = flatten_fabsa(fabsa)

    print("\n=== One flattened sample ===")
    print(train_rows[0])
    sent_counts = Counter([row["sentiment"] for row in train_rows])
    total = sum(sent_counts.values())

    # inverse frequency
    weights = torch.tensor([
        total / sent_counts['negative'],
        total / sent_counts['neutral'],
        total / sent_counts['positive']
    ], dtype=torch.float).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print("Class weights:", weights)

    # Text length analysis on train
    analyze_text_lengths(train_rows)

    # Build PyTorch Datasets & DataLoaders
    train_loader, val_loader, test_loader, label2id, id2label, tokenizer = create_dataloaders(
        train_rows, val_rows, test_rows
    )

    # fetch one batch
    batch = next(iter(train_loader))
    print("\n=== One batch from train_loader ===")
    print("input_ids shape:     ", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("label_id shape:      ", batch["label_id"].shape)
    print("label_id (first 8):  ", batch["label_id"][:8])

    # Metadata example
    print("\nExample metadata from batch:")
    print("text[0]   :", batch["text"][0][:120], "...")
    print("aspect[0] :", batch["aspect"][0])
    print("sent[0]   :", batch["sentiment_str"][0])
    print("industry[0]:", batch["industry"][0])
    print("source[0] :", batch["source"][0])

    # Build model and run one forward pass
    model = build_model(label2id, id2label)
    run_one_forward_pass(train_loader, model)

    # Simple training loop (few epochs)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        val_loss, val_acc = eval_one_epoch(model, val_loader, loss_fn)
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

    print("\nTraining finished (basic loop).")

    # Final evaluation on test split
    evaluate_on_split(model, test_loader, split_name="TEST", id2label=id2label)

    output_dir = "fabsa_roberta_sentiment"

    print(f"\nSaving fine-tuned model and tokenizer to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
