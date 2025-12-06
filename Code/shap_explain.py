import torch
import shap
import numpy as np
from collections import Counter, defaultdict
import random

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================
# CONFIG
# ============================
MODEL_DIR = "fabsa_roberta_sentiment"
MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================
# Load FABSA and flatten
# ============================

def load_fabsa_raw():
    fabsa = load_dataset("jordiclive/FABSA")
    return fabsa


def flatten_fabsa(fabsa_dict):
    def flatten_split(split):
        rows = []
        for ex in split:
            text = ex["text"]
            industry = ex["industry"]
            source = ex["data_source"]
            for aspect, sentiment in ex["labels"]:
                rows.append(
                    {
                        "text": text,
                        "aspect": aspect,
                        "sentiment": sentiment,
                        "industry": industry,
                        "source": source,
                    }
                )
        return rows

    train_rows = flatten_split(fabsa_dict["train"])
    val_rows = flatten_split(fabsa_dict["validation"])
    test_rows = flatten_split(fabsa_dict["test"])
    return train_rows, val_rows, test_rows


# ============================
# Load the model and tokenizer
# ============================

print("\nLoading the model and tokenizer from:", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


# ============================
# Prediction function for SHAP
# ============================

def predict_proba(text_batch):
    """
    SHAP may pass:
      - a single string
      - a list of strings
      - a numpy array of strings
      - a list/array of token lists (['easy', 'to', 'use', ...])
    """

    if isinstance(text_batch, np.ndarray):
        text_batch = text_batch.tolist()

    if isinstance(text_batch, str):
        text_batch = [text_batch]

    if isinstance(text_batch, list) and len(text_batch) > 0 and not isinstance(text_batch[0], str):
        text_batch = [" ".join(tokens) for tokens in text_batch]

    enc = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    return probs



# ============================
# Pick a few examples for explanation
# ============================
fabsa = load_fabsa_raw()
train_rows, val_rows, test_rows = flatten_fabsa(fabsa)

print(f"\nFlattened sizes: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

# Picked:
# - one negative
# - one neutral
# - one positive
samples_to_explain = []
for s in ["negative", "neutral", "positive"]:
    for row in test_rows:
        if row["sentiment"] == s:
            samples_to_explain.append(row)
            break

print("\nSamples chosen for explanation:")
for i, row in enumerate(samples_to_explain):
    print(f"\n--- Sample {i} ({row['sentiment']}) ---")
    print("Aspect :", row["aspect"])
    print("Industry:", row["industry"])
    print("Source :", row["source"])
    print("Text   :", row["text"][:200], "...")

def clean_for_shap(s: str) -> str:
    """
    - Remove the artificial [SEP] marker
    - Remove bracket characters that create ugly tokens
    """
    s = s.replace("[SEP]", " ")
    s = s.replace("[", " ").replace("]", " ")
    # Optionally collapse extra whitespace
    s = " ".join(s.split())
    return s

input_texts_raw = [
    f"{row['aspect']} [SEP] {row['text']}"
    for row in samples_to_explain
]

input_texts = [clean_for_shap(t) for t in input_texts_raw]

probs = predict_proba(input_texts)

print("\nModel probabilities for [negative, neutral, positive]:")
for i, row in enumerate(samples_to_explain):
    print(f"Sample {i} true={row['sentiment']} probs={probs[i]}")


# ============================
# SHAP explainer
# ============================

masker = shap.maskers.Text(tokenizer=r"\W+")
explainer = shap.Explainer(predict_proba, masker)

print("\nComputing SHAP values for the chosen samples...")
shap_values = explainer(input_texts)

def show_top_tokens(shap_values, texts, class_idx, top_k=10):
    """
    Print top contributing tokens
    """
    for i, txt in enumerate(texts):
        print(f"\n===== Sample {i} – class index {class_idx} =====")
        print("Raw text:", txt[:200], "...")
        token_vals = shap_values[i].values[:, class_idx]  # shape: (n_tokens,)
        tokens = shap_values[i].data  # list of tokens

        # pair token with its shap value
        pairs = list(zip(tokens, token_vals))
        # sort by absolute importance
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

        print(f"Top {top_k} tokens by |SHAP| for this class:")
        for tok, val in pairs_sorted[:top_k]:
            print(f"  {tok!r:>10}  shap={val:+.4f}")

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

def compute_aspect_shap_importance(
    rows,
    explainer,
    max_aspects=12,
    max_samples_per_aspect=50,
    top_k_tokens=10,
):
    """
    For each of the aspects, compute which tokens
    push the model toward negative / neutral / positive, using SHAP.

    - rows: flattened FABSA rows (list of dicts with 'text', 'aspect', 'sentiment')
    - explainer: shap.Explainer object for predict_proba
    """

    aspect_counts = Counter(r["aspect"] for r in rows)
    top_aspects = [a for (a, _) in aspect_counts.most_common(max_aspects)]

    for a in top_aspects:
        print("  ", a, "(count =", aspect_counts[a], ")")

    for aspect in top_aspects:
        # Collect rows for this aspect
        aspect_rows = [r for r in rows if r["aspect"] == aspect]

        # Sample if too many
        if len(aspect_rows) > max_samples_per_aspect:
            random.seed(42)
            aspect_rows = random.sample(aspect_rows, max_samples_per_aspect)

        print("\n=====================================")
        print(f"ASPECT: {aspect}")
        print(f"Using {len(aspect_rows)} samples for SHAP")

        input_texts_raw = [
            f"{r['aspect']} [SEP] {r['text']}" for r in aspect_rows
        ]
        input_texts = [clean_for_shap(t) for t in input_texts_raw]

        # Compute SHAP values for this aspect
        shap_values_aspect = explainer(input_texts)

        # Aggregate SHAP values per token, per class
        # token_shap[class_idx][token] = list of shap values
        token_shap = {cls_idx: defaultdict(list) for cls_idx in range(len(label2id))}

        for i in range(len(input_texts)):
            tokens = shap_values_aspect[i].data
            values = shap_values_aspect[i].values

            for cls_idx in range(values.shape[1]):
                cls_vals = values[:, cls_idx]
                for tok, val in zip(tokens, cls_vals):
                    token_shap[cls_idx][tok].append(val)

        # For each class, compute mean SHAP per token and print top-k
        for cls_idx in range(len(label2id)):
            sentiment_name = id2label[cls_idx]
            print(f"\n--- Top tokens pushing towards {sentiment_name.upper()} ---")

            # Average SHAP value: how much token pushes toward this class on average
            avg_shap = []
            for tok, vals in token_shap[cls_idx].items():
                mean_val = float(np.mean(vals))
                avg_shap.append((tok, mean_val))

            # Sort by mean SHAP descending (strongest positive push to this class)
            avg_shap.sort(key=lambda x: x[1], reverse=True)

            for tok, val in avg_shap[:top_k_tokens]:
                print(f"  {tok!r:>15}  mean_shap={val:+.4f}")

for i, row in enumerate(samples_to_explain):
    cls_idx = label2id[row["sentiment"]]
    print(f"\n#############################")
    print(f"SHAP explanation for sample {i} (true={row['sentiment']})")
    show_top_tokens(shap_values, input_texts, class_idx=cls_idx, top_k=10)


# ==========================================
# Aspect-level SHAP analysis
# ==========================================

# Use all rows for better coverage
all_rows = train_rows + val_rows + test_rows

print("\n\n=== Aspect-level SHAP analysis ===")
compute_aspect_shap_importance(
    rows=all_rows,
    explainer=explainer,
    max_aspects=12,
    max_samples_per_aspect=50,
    top_k_tokens=10,
)