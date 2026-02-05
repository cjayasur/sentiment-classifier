"""
Training script for the Sentiment Transformer.
Uses a built-in movie review dataset - no downloads needed!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import re
import random
from pathlib import Path
from model import SentimentTransformer, count_parameters


# Simple movie review dataset (embedded - no external downloads)
SAMPLE_REVIEWS = [
    # Positive reviews
    ("This movie was absolutely fantastic! Great acting and storyline.", 1),
    ("I loved every minute of this film. Highly recommend!", 1),
    ("Brilliant performances and beautiful cinematography.", 1),
    ("One of the best movies I've seen this year. Amazing!", 1),
    ("Wonderful story with excellent character development.", 1),
    ("A masterpiece of modern cinema. Truly outstanding.", 1),
    ("Incredibly entertaining from start to finish.", 1),
    ("The acting was superb and the plot kept me engaged.", 1),
    ("A delightful film that exceeded my expectations.", 1),
    ("Phenomenal movie with great emotional depth.", 1),
    ("Loved the humor and the heartwarming moments.", 1),
    ("An absolute joy to watch. Perfect entertainment.", 1),
    ("The director did an amazing job with this one.", 1),
    ("Captivating story and memorable characters.", 1),
    ("A feel-good movie that leaves you smiling.", 1),
    ("Excellent film with a powerful message.", 1),
    ("The chemistry between the actors was incredible.", 1),
    ("A must-see movie for everyone.", 1),
    ("Thoroughly enjoyed this cinematic experience.", 1),
    ("Beautiful storytelling and stunning visuals.", 1),
    # Negative reviews
    ("This movie was terrible. Complete waste of time.", 0),
    ("I couldn't even finish watching it. So boring.", 0),
    ("Awful acting and a predictable plot.", 0),
    ("One of the worst films I've ever seen.", 0),
    ("Disappointing and poorly executed.", 0),
    ("The storyline made no sense whatsoever.", 0),
    ("Boring, dull, and completely forgettable.", 0),
    ("I want my two hours back. Terrible movie.", 0),
    ("The acting was wooden and unconvincing.", 0),
    ("A complete disaster from beginning to end.", 0),
    ("Such a letdown. Had high hopes but was disappointed.", 0),
    ("Poorly written script with bad dialogue.", 0),
    ("The pacing was awful and the ending was weak.", 0),
    ("Not worth watching. Save your money.", 0),
    ("Cringeworthy performances throughout.", 0),
    ("A mess of a movie with no redeeming qualities.", 0),
    ("Felt like the longest movie ever. So tedious.", 0),
    ("The plot holes were impossible to ignore.", 0),
    ("Uninspired and derivative. Nothing new here.", 0),
    ("I regret watching this. Truly awful.", 0),
]

# Data augmentation - create more training examples
def augment_data(reviews, multiplier=10):
    """Simple augmentation by word shuffling and synonym-ish replacements"""
    augmented = list(reviews)

    positive_words = ["great", "amazing", "excellent", "wonderful", "fantastic", "brilliant", "superb", "outstanding"]
    negative_words = ["terrible", "awful", "horrible", "bad", "poor", "disappointing", "boring", "worst"]

    for text, label in reviews:
        for _ in range(multiplier):
            words = text.split()
            # Random word dropout
            if len(words) > 5:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)

            # Occasionally swap sentiment words
            if random.random() > 0.7:
                word_list = positive_words if label == 1 else negative_words
                for i, w in enumerate(words):
                    if w.lower() in positive_words + negative_words:
                        words[i] = random.choice(word_list)
                        break

            augmented.append((" ".join(words), label))

    return augmented


class SimpleTokenizer:
    """Basic word-level tokenizer"""
    def __init__(self, vocab_size=5000):
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

    def encode(self, text, max_len=64):
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
    def __init__(self, texts, labels, tokenizer, max_len=64):
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
    # Configuration
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    MAX_SEQ_LEN = 64
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    print("Preparing dataset...")
    data = augment_data(SAMPLE_REVIEWS, multiplier=15)
    random.shuffle(data)

    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    # Split data
    split_idx = int(len(texts) * 0.8)
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=2000)
    tokenizer.fit(train_texts)
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Create model
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
    print("\nTraining...")
    print("-" * 50)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            tokenizer.save("checkpoints/tokenizer.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("-" * 50)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"\nModel saved to checkpoints/best_model.pt")
    print(f"Tokenizer saved to checkpoints/tokenizer.pt")

    # Test with some examples
    print("\n" + "=" * 50)
    print("Testing with new examples:")
    print("=" * 50)

    model.load_state_dict(torch.load("checkpoints/best_model.pt", weights_only=True))
    model.eval()

    test_texts = [
        "This is the best movie I've ever watched!",
        "Absolutely horrible. Don't waste your time.",
        "It was okay, nothing special.",
        "A cinematic triumph with stellar performances!",
        "Boring and predictable from start to finish.",
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
            print(f"  → {sentiment} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
