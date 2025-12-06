#%%
from datasets import load_dataset
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
from collections import Counter
import nltk

#%%
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nltk.download('punkt')
#%%
# Load FABSA from Hugging Face
print("Loading FABSA dataset...")
fabsa = load_dataset("jordiclive/FABSA")

print("\n=== Dataset object ===")
print(fabsa)

print("\n=== Train split columns ===")
print(fabsa["train"].column_names)

# one example
example = fabsa["train"][0]
print("\n=== One example from train split ===")
for k, v in example.items():
    print(f"{k}: {str(v)[:120]}{'...' if len(str(v)) > 120 else ''}")

#%%
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
                "source": source,
            })
    return rows

train_rows = flatten_split(fabsa["train"])
val_rows   = flatten_split(fabsa["validation"])
test_rows  = flatten_split(fabsa["test"])

# Convert to DataFrame
df_train = pd.DataFrame(train_rows)
df_val   = pd.DataFrame(val_rows)
df_test  = pd.DataFrame(test_rows)

print("Train flattened shape:", df_train.shape)
print("Validation flattened shape:", df_val.shape)
print("Test flattened shape:", df_test.shape)

print("\n=== First 5 rows of df_train ===")
print(df_train.head())

#%%
aspect_counts = df_train.groupby("text").size()

# Identify texts with <= 8 aspects
valid_texts = aspect_counts[aspect_counts <= 8].index

# Filter df_train to keep only valid reviews
df_train = df_train[df_train["text"].isin(valid_texts)].reset_index(drop=True)

#%%
# Count how many aspect annotations each review has
aspect_counts = df_train.groupby("text").size()

print("=== Aspect Count Summary ===")
print(aspect_counts.describe(), "\n")

# Count frequency of each aspect number
dist = aspect_counts.value_counts().sort_index()
dist_pct = dist / dist.sum() * 100

# Plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=dist.index, y=dist.values, color="steelblue")

ax.set_title("Distribution of Number of Aspects per Review")
ax.set_xlabel("Number of Aspects in Review")
ax.set_ylabel("Number of Reviews")

# Annotate percentages above bars
for i, (cnt, pct) in enumerate(zip(dist.values, dist_pct.values)):
    ax.text(i, cnt + dist.max() * 0.015,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

#%%
# Count occurrences of each data source
source_counts = df_train["source"].value_counts().sort_values(ascending=False)
source_pct = (source_counts / source_counts.sum()) * 100

print("=== Source Count Summary ===")
print(source_counts)
print("\n=== Source Percentages ===")
print(source_pct.round(2))

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x=source_counts.values,
    y=source_counts.index,
    palette="Blues_r"
)

ax.set_title("Distribution of Review Sources")
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Data Source")

# Annotate percentages beside bars
for i, (count, pct) in enumerate(zip(source_counts.values, source_pct.values)):
    ax.text(
        count + source_counts.max() * 0.01,   # slightly outside bar
        i,
        f"{pct:.1f}%",
        va="center",
        fontsize=10
    )

plt.tight_layout()
plt.show()

#%%
#
df = df_train_filtered if 'df_train_filtered' in globals() else df_train

review_groups = df.groupby("text").agg({
    "industry": "first",
    "aspect": "count"         # count how many aspects that review has
}).rename(columns={"aspect": "num_aspects"})

# Compute averages per industry
industry_aspect_avg = review_groups.groupby("industry")["num_aspects"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x=industry_aspect_avg.values,
    y=industry_aspect_avg.index,
    palette="viridis"
)

ax.set_title("Average Number of Aspects per Review by Industry")
ax.set_xlabel("Average # of Aspects")
ax.set_ylabel("Industry")

for i, v in enumerate(industry_aspect_avg.values):
    ax.text(v + 0.03, i, f"{v:.2f}", va='center')

plt.tight_layout()
plt.show()

#%%
df = df_train_filtered if 'df_train_filtered' in globals() else df_train

# Create cross-tab
heatmap_df = pd.crosstab(df["source"], df["sentiment"])

# Convert counts to row percentages
heatmap_pct = heatmap_df.div(heatmap_df.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12, 7))
sns.heatmap(
    heatmap_pct,
    cmap="RdYlGn",
    annot=True,
    fmt=".1f",
    linewidths=.5,
    cbar_kws={"label": "Percentage (%)"}
)

plt.title("Sentiment Distribution per Data Source")
plt.xlabel("Sentiment")
plt.ylabel("Data Source")

plt.tight_layout()
plt.show()

#%%
# Count sentiment frequencies
sent_counts = df_train["sentiment"].value_counts().sort_index()
sent_percent = (sent_counts / sent_counts.sum()) * 100

print("Sentiment counts:")
print(sent_counts)
print("\nSentiment %:")
print(sent_percent.round(2))

plt.figure(figsize=(6,4))
ax = sns.barplot(
    x=sent_counts.index,
    y=sent_counts.values,
    palette="viridis"
)

for i, value in enumerate(sent_counts.values):
    pct = sent_percent.values[i]
    ax.text(
        i,
        value + max(sent_counts)*0.02,
        f"{pct:.1f}%",
        ha='center',
        fontsize=12
    )

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#%%
# Count aspect frequencies
aspect_counts = df_train["aspect"].value_counts()

# Compute percentages
aspect_percent = (aspect_counts / aspect_counts.sum()) * 100

# Select top aspects
top_n = 20
top_aspects = aspect_counts.head(top_n)

# Combine counts + percentages into a DataFrame
df_aspects = pd.DataFrame({
    "count": top_aspects,
    "percent": aspect_percent.head(top_n)
})

# Sort so largest is on top
df_aspects = df_aspects.sort_values("count", ascending=False)

plt.figure(figsize=(10, 10))
ax = sns.barplot(
    x="count",
    y=df_aspects.index,
    data=df_aspects,
    palette="magma"
)

for i, (count, pct) in enumerate(zip(df_aspects["count"], df_aspects["percent"])):
    ax.text(
        count + max(df_aspects["count"]) * 0.01,
        i,
        f"{pct:.1f}%",
        va='center',
        fontsize=11
    )

plt.title(f"Most Frequent Aspect Categories")
plt.xlabel("Count")
plt.ylabel("Aspect Category")
plt.tight_layout()
plt.show()

#%%
top_aspects = df_train["aspect"].value_counts().head(top_n).index

df_top = df_train[df_train["aspect"].isin(top_aspects)]

# Create pivot table: aspect and sentiment
pivot = pd.crosstab(df_top["aspect"], df_top["sentiment"])

# Convert counts to percentages per aspect
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

print("\n=== Pivot table (counts) ===")
print(pivot.head())

print("\n=== Pivot table (percent %) ===")
print(pivot_pct.round(1).head())

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_pct,
    cmap="RdYlGn",
    annot=True,
    fmt=".1f",
    cbar_kws={'label': 'Percentage (%)'}
)

plt.title("Sentiment Distribution per Aspect")
plt.xlabel("Sentiment")
plt.ylabel("Aspect Category")
plt.tight_layout()
plt.show()

#%%
# Count industries
industry_counts = df_train["industry"].value_counts()
industry_percent = (industry_counts / industry_counts.sum()) * 100

top_ind = industry_counts.head(top_n)

df_ind = pd.DataFrame({
    "count": top_ind,
    "percent": industry_percent.head(top_n)
})

df_ind = df_ind.sort_values("count", ascending=False)

plt.figure(figsize=(10, 8))
ax = sns.barplot(
    x="count",
    y=df_ind.index,
    data=df_ind,
    palette="crest"
)

for i, (count, pct) in enumerate(zip(df_ind["count"], df_ind["percent"])):
    ax.text(
        count + max(df_ind["count"]) * 0.01,
        i,
        f"{pct:.1f}%",
        va="center",
        fontsize=11
    )

plt.title(f"Industries by Review Count")
plt.xlabel("Count")
plt.ylabel("Industry")
plt.tight_layout()
plt.show()

#%%
df_top_ind = df_train[df_train["industry"].isin(df_ind.index)]

pivot_ind = pd.crosstab(df_top_ind["industry"], df_top_ind["sentiment"])

# Convert to percentages per industry
pivot_ind_pct = pivot_ind.div(pivot_ind.sum(axis=1), axis=0) * 100

plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_ind_pct,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    cbar_kws={"label": "Percentage (%)"}
)

plt.title("Sentiment Distribution per Industry")
plt.xlabel("Sentiment")
plt.ylabel("Industry")
plt.tight_layout()
plt.show()

#%%
# Compute character and token lengths
df_train["len_chars"] = df_train["text"].str.len()
df_train["len_tokens"] = df_train["text"].str.split().apply(len)

# Summary statistics
print("\n=== Character Length Stats ===")
print(df_train["len_chars"].describe())

print("\n=== Token Length Stats ===")
print(df_train["len_tokens"].describe())

# ----- Plot: Character Length Distribution -----
plt.figure(figsize=(8,4))
sns.histplot(df_train["len_chars"], bins=40, kde=True, color="teal")
plt.axvline(df_train["len_chars"].median(), color="black", linestyle="--", label="Median")
plt.axvline(df_train["len_chars"].quantile(0.9), color="red", linestyle="--", label="90th percentile")
plt.title("Distribution of Review Lengths (Characters)")
plt.xlabel("Character Count")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

#%%
# ----- Plot: Token Length Distribution -----
plt.figure(figsize=(8,4))
sns.histplot(df_train["len_tokens"], binwidth=10, kde=True, color="purple")
plt.axvline(df_train["len_tokens"].median(), color="black", linestyle="--", label="Median")
plt.axvline(df_train["len_tokens"].mean(), color="red", linestyle="--", label="Mean")
plt.title("Distribution of Review Lengths (Tokens)")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

#%%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text)  # collapse spaces
    return text.strip()

def get_top_words(df, sentiment, n=15):
    texts = df[df["sentiment"] == sentiment]["text"].apply(clean_text)
    tokens = []
    for t in texts:
        tokens.extend(nltk.word_tokenize(t))

    # remove stopwords and irrelevant words
    stopwords = set(nltk.corpus.stopwords.words("english"))
    custom_remove = {"app", "website", "company", "service", "customer"}
    stopwords.update(custom_remove)

    filtered = [w for w in tokens if w not in stopwords and len(w) > 2]

    return Counter(filtered).most_common(n)


def plot_top_words(word_counts, sentiment):
    words, counts = zip(*word_counts)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(f"Top Words — {sentiment.capitalize()} Reviews")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.show()

# Get top words
top_pos = get_top_words(df_train, "positive")
top_neg = get_top_words(df_train, "negative")
top_neu = get_top_words(df_train, "neutral")

# Plot
plot_top_words(top_pos, "positive")
plot_top_words(top_neg, "negative")
plot_top_words(top_neu, "neutral")

#%%
def get_top_bigrams(df, sentiment, n=15):
    texts = df[df["sentiment"] == sentiment]["text"].apply(clean_text)
    bigram_counter = Counter()

    # same stopwords idea as before
    stopwords = set(nltk.corpus.stopwords.words("english"))
    custom_remove = {"app", "website", "company", "service", "customer"}
    stopwords.update(custom_remove)

    for t in texts:
        tokens = [w for w in nltk.word_tokenize(t) if w.lower() not in stopwords and len(w) > 2]
        # build bigrams from filtered tokens
        for w1, w2 in zip(tokens, tokens[1:]):
            bigram = f"{w1.lower()} {w2.lower()}"
            bigram_counter[bigram] += 1

    return bigram_counter.most_common(n)

def plot_top_bigrams(bigram_counts, sentiment):
    if not bigram_counts:
        print(f"No bigrams for sentiment={sentiment}")
        return
    phrases, counts = zip(*bigram_counts)
    plt.figure(figsize=(8,4))
    sns.barplot(x=list(counts), y=list(phrases), palette="mako")
    plt.title(f"Top Bigrams — {sentiment.capitalize()} Reviews")
    plt.xlabel("Frequency")
    plt.ylabel("Bigram")
    plt.tight_layout()
    plt.show()

# Get top bigrams
top_pos_bi = get_top_bigrams(df_train, "positive")
top_neg_bi = get_top_bigrams(df_train, "negative")
top_neu_bi = get_top_bigrams(df_train, "neutral")

# Plot
plot_top_bigrams(top_pos_bi, "positive")
plot_top_bigrams(top_neg_bi, "negative")
plot_top_bigrams(top_neu_bi, "neutral")

#%%
def clean_text_no_org(text):
    text = text.lower()
    # Remove anonymized placeholder "org" in all obvious cases
    text = re.sub(r"\borg\b", " ", text)          # 'org' as its own word
    text = re.sub(r"org\.?", " ", text)           # 'org.' or similar
    text = re.sub(r"[^a-zA-Z\s]", " ", text)      # keep only letters/spaces
    text = re.sub(r"\s+", " ", text)              # collapse multiple spaces
    return text.strip()
def get_top_words_no_org(df, sentiment, n=15):
    texts = df[df["sentiment"] == sentiment]["text"].apply(clean_text_no_org)
    tokens = []
    for t in texts:
        tokens.extend(nltk.word_tokenize(t))

    # remove stopwords and irrelevant words
    stopwords = set(nltk.corpus.stopwords.words("english"))
    custom_remove = {"app", "website", "company", "service", "customer", "org"}
    stopwords.update(custom_remove)

    filtered = [w for w in tokens if w not in stopwords and len(w) > 2 and w != "org"]

    return Counter(filtered).most_common(n)


def plot_top_words(word_counts, sentiment):
    words, counts = zip(*word_counts)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(f"Top Words — {sentiment.capitalize()} Reviews")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.show()


# Get top words
top_pos = get_top_words_no_org(df_train, "positive")
top_neg = get_top_words_no_org(df_train, "negative")
top_neu = get_top_words_no_org(df_train, "neutral")

# Plot
plot_top_words(top_pos, "positive")
plot_top_words(top_neg, "negative")
plot_top_words(top_neu, "neutral")


#%%
def get_top_bigrams_no_org(df, sentiment, n=15):
    texts = df[df["sentiment"] == sentiment]["text"].apply(clean_text_no_org)
    bigram_counter = Counter()

    # stopwords + 'org'
    stopwords = set(nltk.corpus.stopwords.words("english"))
    custom_remove = {"app", "website", "company", "service", "customer", "org"}
    stopwords.update(custom_remove)

    for t in texts:
        tokens = [w for w in nltk.word_tokenize(t)
                  if w not in stopwords and len(w) > 2 and w != "org"]

        # build bigrams
        for w1, w2 in zip(tokens, tokens[1:]):
            if "org" in (w1, w2):
                continue  # extra safety, skip any bigram involving 'org'
            bigram = f"{w1} {w2}"
            bigram_counter[bigram] += 1

    return bigram_counter.most_common(n)


def plot_top_bigrams(bigram_counts, sentiment):
    if not bigram_counts:
        print(f"No bigrams for sentiment={sentiment}")
        return
    phrases, counts = zip(*bigram_counts)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(counts), y=list(phrases), palette="mako")
    plt.title(f"Top Bigrams — {sentiment.capitalize()} Reviews")
    plt.xlabel("Frequency")
    plt.ylabel("Bigram")
    plt.tight_layout()
    plt.show()

# Compute cleaned bigrams
top_pos_bi = get_top_bigrams_no_org(df_train, "positive")
top_neg_bi = get_top_bigrams_no_org(df_train, "negative")
top_neu_bi = get_top_bigrams_no_org(df_train, "neutral")

# Plot cleaned bigrams
plot_top_bigrams(top_pos_bi, "positive")
plot_top_bigrams(top_neg_bi, "negative")
plot_top_bigrams(top_neu_bi, "neutral")

#%%
# Filter only positive sentiment rows
df_pos = df_train[df_train["sentiment"] == "positive"]

# Count industries within positive reviews
pos_industry_counts = df_pos["industry"].value_counts()
pos_industry_pct = (pos_industry_counts / pos_industry_counts.sum()) * 100

print("=== Positive Sentiment Industry Counts ===")
print(pos_industry_counts)
print("\n=== Positive Sentiment Industry Percentages ===")
print(pos_industry_pct.round(2))

# Plot
plt.figure(figsize=(10, 8))
ax = sns.barplot(
    x=pos_industry_counts.values,
    y=pos_industry_counts.index,
    palette="Greens"
)

ax.set_title("Industry Distribution within Positive Sentiment Reviews")
ax.set_xlabel("Count")
ax.set_ylabel("Industry")

# Add percentage labels to the right of bars
for i, (count, pct) in enumerate(zip(pos_industry_counts.values, pos_industry_pct.values)):
    ax.text(
        count + pos_industry_counts.max() * 0.01,
        i,
        f"{pct:.1f}%",
        va="center",
        fontsize=11
    )

plt.tight_layout()
plt.show()

#%%
# Filter dataset to only positive sentiment
df_pos = df_train[df_train["sentiment"] == "positive"]

# Count aspects within positive sentiment reviews
pos_aspect_counts = df_pos["aspect"].value_counts().sort_values(ascending=False)
pos_aspect_pct = (pos_aspect_counts / pos_aspect_counts.sum()) * 100

print("=== Positive Sentiment Aspect Counts ===")
print(pos_aspect_counts)
print("\n=== Positive Sentiment Aspect Percentages ===")
print(pos_aspect_pct.round(2))

top_pos_aspects = pos_aspect_counts.head(top_n)
top_pos_pct = pos_aspect_pct.head(top_n)

plt.figure(figsize=(10, 10))
ax = sns.barplot(
    x=top_pos_aspects.values,
    y=top_pos_aspects.index,
    palette="Greens"
)

ax.set_title(f"Top {top_n} Aspects Within Positive Sentiment Reviews")
ax.set_xlabel("Count")
ax.set_ylabel("Aspect")

# Add percentage labels beside bars
for i, (count, pct) in enumerate(zip(top_pos_aspects.values, top_pos_pct.values)):
    ax.text(
        count + top_pos_aspects.max() * 0.01,
        i,
        f"{pct:.1f}%",
        va="center",
        fontsize=11
    )

plt.tight_layout()
plt.show()

#%%
# Filter dataset to only negative sentiment
df_neg = df_train[df_train["sentiment"] == "negative"]

# Count industries within negative sentiment reviews
neg_industry_counts = df_neg["industry"].value_counts().sort_values(ascending=False)
neg_industry_pct = (neg_industry_counts / neg_industry_counts.sum()) * 100

print("=== Negative Sentiment Industry Counts ===")
print(neg_industry_counts)
print("\n=== Negative Sentiment Industry Percentages ===")
print(neg_industry_pct.round(2))

top_neg_industries = neg_industry_counts.head(top_n)
top_neg_pct = neg_industry_pct.head(top_n)

plt.figure(figsize=(10, 8))
ax = sns.barplot(
    x=top_neg_industries.values,
    y=top_neg_industries.index,
    palette="Reds"
)

ax.set_title(f"Industry Distribution Within Negative Sentiment Reviews")
ax.set_xlabel("Count")
ax.set_ylabel("Industry")

for i, (count, pct) in enumerate(zip(top_neg_industries.values, top_neg_pct.values)):
    ax.text(
        count + top_neg_industries.max() * 0.01,
        i,
        f"{pct:.1f}%",
        va="center",
        fontsize=11
    )

plt.tight_layout()
plt.show()

#%%
# Filter dataset to only negative sentiment
df_neg = df_train[df_train["sentiment"] == "negative"]

# Count aspects within negative reviews
neg_aspect_counts = df_neg["aspect"].value_counts().sort_values(ascending=False)
neg_aspect_pct = (neg_aspect_counts / neg_aspect_counts.sum()) * 100

print("=== Negative Sentiment Aspect Counts ===")
print(neg_aspect_counts)
print("\n=== Negative Sentiment Aspect Percentages ===")
print(neg_aspect_pct.round(2))

top_n = 20
top_neg_aspects = neg_aspect_counts.head(top_n)
top_neg_pct = neg_aspect_pct.head(top_n)

plt.figure(figsize=(10, 10))
ax = sns.barplot(
    x=top_neg_aspects.values,
    y=top_neg_aspects.index,
    palette="Reds"
)

ax.set_title(f"Top {top_n} Aspects Within Negative Sentiment Reviews")
ax.set_xlabel("Count")
ax.set_ylabel("Aspect")

# Add percentage labels beside bars
for i, (count, pct) in enumerate(zip(top_neg_aspects.values, top_neg_pct.values)):
    ax.text(
        count + top_neg_aspects.max() * 0.01,
        i,
        f"{pct:.1f}%",
        va="center",
        fontsize=11
    )

plt.tight_layout()
plt.show()
