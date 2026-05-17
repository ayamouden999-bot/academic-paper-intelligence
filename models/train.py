"""
models/train.py — Training script for TextCNN on arXiv dataset.
Run this in Google Colab (GPU runtime) for ~20-30 min training.
Saves model to models/textcnn_trained.pth
"""

import os, json, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

from models.textcnn import TextCNN
from config import (
    DATASET_NAME, MAX_SEQ_LEN, VOCAB_SIZE, BATCH_SIZE,
    NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, FIELD_LABELS
)


# ── Tokenizer (simple word-level) ─────────────────────────────────────────────

class Vocab:
    def __init__(self, max_size: int = VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts: list[str]):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"Vocabulary built: {len(self.word2idx):,} tokens")

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        # Pad to max_len
        ids += [0] * (max_len - len(ids))
        return ids

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path: str):
        with open(path) as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}


# ── Dataset ───────────────────────────────────────────────────────────────────

class ArxivDataset(Dataset):
    def __init__(self, texts, field_labels, novelty_labels, vocab: Vocab):
        self.texts   = texts
        self.fields  = field_labels
        self.novelty = novelty_labels
        self.vocab   = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.vocab.encode(self.texts[idx])
        return {
            "input_ids":    torch.tensor(tokens, dtype=torch.long),
            "field_label":  torch.tensor(self.fields[idx], dtype=torch.long),
            "novelty_label":torch.tensor(self.novelty[idx], dtype=torch.long),
        }


def assign_novelty_label(text: str) -> int:
    """
    Heuristic novelty labeling based on abstract keywords.
    Week 2 stretch: replace with a human-annotated subset.
    0=Incremental, 1=Moderate, 2=High, 3=Breakthrough
    """
    text = text.lower()
    breakthrough = ["novel", "first", "state-of-the-art", "outperform", "breakthrough", "surpass"]
    high         = ["propose", "new approach", "introduce", "significant", "substantially"]
    incremental  = ["extend", "improve upon", "build on", "adaptation", "variant"]

    if any(w in text for w in breakthrough): return 3
    if any(w in text for w in high):         return 2
    if any(w in text for w in incremental):  return 0
    return 1  # Moderate by default


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Training on: {device}")

    # Load dataset
    print("Loading arXiv dataset...")
    ds = load_dataset(DATASET_NAME)
    train_data = ds["train"]
    test_data  = ds["test"]

    texts_train  = train_data["text"]
    labels_train = train_data["label"]
    texts_test   = test_data["text"]
    labels_test  = test_data["label"]

    novelty_train = [assign_novelty_label(t) for t in texts_train]
    novelty_test  = [assign_novelty_label(t) for t in texts_test]

    # Build vocabulary
    vocab = Vocab()
    vocab.build(texts_train)
    os.makedirs("models", exist_ok=True)
    vocab.save("models/vocab.json")

    # DataLoaders
    train_ds = ArxivDataset(texts_train, labels_train, novelty_train, vocab)
    test_ds  = ArxivDataset(texts_test,  labels_test,  novelty_test,  vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model
    model = TextCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        t0 = time.time()

        for batch in train_loader:
            ids     = batch["input_ids"].to(device)
            f_label = batch["field_label"].to(device)
            n_label = batch["novelty_label"].to(device)

            field_logits, novelty_logits = model(ids)

            # Combined loss: both heads trained simultaneously
            loss = criterion(field_logits, f_label) + \
                   0.5 * criterion(novelty_logits, n_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (field_logits.argmax(1) == f_label).sum().item()
            total      += f_label.size(0)

        acc = correct / total
        history["train_loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(acc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {time.time()-t0:.1f}s")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

    # Evaluation
    evaluate(model, test_loader, device)
    plot_training(history)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids     = batch["input_ids"].to(device)
            f_label = batch["field_label"]
            field_logits, _ = model(ids)
            preds = field_logits.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(f_label.tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n📊 Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=list(FIELD_LABELS.values())
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=list(FIELD_LABELS.values()),
        yticklabels=list(FIELD_LABELS.values()),
        cmap="Blues"
    )
    plt.title("TextCNN Confusion Matrix — Research Field Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to outputs/confusion_matrix.png")


def plot_training(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], marker='o')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax2.plot(history["train_acc"], marker='o', color='green')
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150)
    print("Training curves saved to outputs/training_curves.png")


def text_to_tensor(text: str) -> torch.Tensor:
    """Helper used by classifier_tool.py at inference time."""
    vocab = Vocab()
    vocab.load("models/vocab.json")
    ids = vocab.encode(text)
    return torch.tensor(ids, dtype=torch.long)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    train()
