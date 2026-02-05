"""
Transformer-based Sentiment Classifier - Built from scratch
So you can see those attention mechanisms you've been studying!
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    The heart of the transformer - scaled dot-product attention with multiple heads.
    Q, K, V projections -> attention scores -> weighted values
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.w_q(x)  # (batch, seq, d_model)
        K = self.w_k(x)
        V = self.w_v(x)

        # Reshape for multi-head: (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq, seq)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, heads, seq, d_k)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)

        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU like in BERT/GPT

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer encoder block: attention + feed-forward with residuals"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x, attn_weights


class SentimentTransformer(nn.Module):
    """
    Complete transformer for sentiment classification.

    Architecture:
    - Token embedding + positional encoding
    - N transformer encoder blocks
    - Global average pooling (or [CLS] token)
    - Classification head
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        max_seq_len: int = 256,
        n_classes: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, return_attention=False):
        batch_size, seq_len = input_ids.shape

        # Create padding mask
        mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)

        # Get positions
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embed tokens + positions
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, mask)
            attention_weights.append(attn)

        x = self.norm(x)

        # Global average pooling (ignoring padding)
        mask_expanded = (input_ids != self.pad_idx).unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Classify
        logits = self.classifier(x)

        if return_attention:
            return logits, attention_weights
        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
