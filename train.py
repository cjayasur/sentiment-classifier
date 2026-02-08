"""
Training script for the Sentiment Transformer.
Uses the IMDB dataset - 50,000 real movie reviews.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import re
import random
from pathlib import Path
from model import SentimentTransformer, count_parameters


class SimpleTokenizer:
    """Basic word-level tokenizer"""
    def __init__(self, vocab_size=20000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.pad_idx = 0
        self.unk_idx = 1

    def fit(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        # Keep most common words
        for word, _ in word_counts.most_common(self.vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def _tokenize(self, text):
        """Simple tokenization: lowercase, remove punctuation, split"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def encode(self, text, max_len=256):
        """Convert text to token IDs"""
        words = self._tokenize(text)
        ids = [self.word2idx.get(w, self.unk_idx) for w in words]

        # Pad or truncate
        if len(ids) < max_len:
            ids = ids + [self.pad_idx] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids

    def decode(self, ids):
        """Convert token IDs back to text"""
        words = [self.idx2word.get(i, "<UNK>") for i in ids if i != self.pad_idx]
        return " ".join(words)

    def save(self, path):
        """Save tokenizer vocabulary"""
        torch.save({
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }, path)

    @classmethod
    def load(cls, path):
        """Load tokenizer from file"""
        data = torch.load(path, weights_only=False)
        tokenizer = cls(data['vocab_size'])
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = data['idx2word']
        return tokenizer


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification"""
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids = self.tokenizer.encode(text, self.max_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    random.seed(42)
    torch.manual_seed(42)

    # Configuration - scaled up for real data
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    MAX_SEQ_LEN = 256
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 4
    VOCAB_SIZE = 20000

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load IMDB dataset
    print("Loading IMDB dataset (50,000 reviews)...")
    from datasets import load_dataset
    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["test"]["text"]
    val_labels = dataset["test"]["label"]

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Build tokenizer on training data
    print("Building vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit(train_texts)
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Create model - bigger for real data
    model = SentimentTransformer(
        vocab_size=len(tokenizer.word2idx),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_MODEL * 4,
        max_seq_len=MAX_SEQ_LEN,
        n_classes=2,
        dropout=0.1,
        pad_idx=tokenizer.pad_idx
    ).to(device)

    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Architecture: {N_LAYERS} layers, {N_HEADS} heads, d_model={D_MODEL}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # Training loop
    print("\nTraining on IMDB...")
    print("-" * 60)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            tokenizer.save("checkpoints/tokenizer.pt")
            marker = " *"
        else:
            marker = ""

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{marker}")

    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"\nModel saved to checkpoints/best_model.pt")
    print(f"Tokenizer saved to checkpoints/tokenizer.pt")

    # Test with examples
    print("\n" + "=" * 60)
    print("Testing with new examples:")
    print("=" * 60)

    model.load_state_dict(torch.load("checkpoints/best_model.pt", weights_only=True))
    model.eval()

    test_texts = [
        "This is the best movie I've ever watched!",
        "Absolutely horrible. Don't waste your time.",
        "It was okay, nothing special.",
        "A cinematic triumph with stellar performances!",
        "Boring and predictable from start to finish.",
        "This movie sucks. Total garbage.",
        "I loved every minute, truly a masterpiece.",
        "Meh, I've seen better.",
        "An incredible journey that moved me to tears.",
        "What a waste of two hours of my life.",
    ]

    with torch.no_grad():
        for text in test_texts:
            input_ids = torch.tensor([tokenizer.encode(text, MAX_SEQ_LEN)]).to(device)
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
            confidence = probs[0][pred].item()

            sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
            print(f"\n'{text}'")
            print(f"  -> {sentiment} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
