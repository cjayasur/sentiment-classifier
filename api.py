"""
FastAPI server for the Sentiment Transformer.
Serves predictions via REST API.
"""

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from model import SentimentTransformer
from train import SimpleTokenizer

# Initialize FastAPI
app = FastAPI(
    title="Sentiment Classifier API",
    description="A transformer-based sentiment classifier built from scratch with PyTorch",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None
device = None

# Model configuration (must match training)
MODEL_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "max_seq_len": 256,
    "n_classes": 2,
    "dropout": 0.1
}


class TextInput(BaseModel):
    """Request model for single text prediction"""
    text: str
    return_attention: Optional[bool] = False


class BatchInput(BaseModel):
    """Request model for batch prediction"""
    texts: list[str]


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str
    model_params: Optional[int] = None


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global model, tokenizer, device

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint_path = Path("checkpoints")
    model_path = checkpoint_path / "best_model.pt"
    tokenizer_path = checkpoint_path / "tokenizer.pt"

    if not model_path.exists() or not tokenizer_path.exists():
        print("WARNING: Model checkpoints not found. Run train.py first!")
        print("API will return errors until model is trained.")
        return

    # Load tokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    # Load model
    model = SentimentTransformer(
        vocab_size=len(tokenizer.word2idx),
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_model"] * 4,
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        n_classes=MODEL_CONFIG["n_classes"],
        dropout=MODEL_CONFIG["dropout"],
        pad_idx=tokenizer.pad_idx
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully on {device}")
    print(f"Parameters: {param_count:,}")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    param_count = None
    if model is not None:
        param_count = sum(p.numel() for p in model.parameters())

    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "not initialized",
        model_params=param_count
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """
    Predict sentiment for a single text.

    Returns:
    - sentiment: "positive" or "negative"
    - confidence: probability of predicted class
    - probabilities: dict with both class probabilities
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first to create checkpoints."
        )

    # Tokenize
    input_ids = torch.tensor(
        [tokenizer.encode(input_data.text, MODEL_CONFIG["max_seq_len"])]
    ).to(device)

    # Predict
    with torch.no_grad():
        if input_data.return_attention:
            logits, attention = model(input_ids, return_attention=True)
        else:
            logits = model(input_ids)

        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
        confidence = probs[0][pred].item()

    sentiment = "positive" if pred == 1 else "negative"

    return PredictionResponse(
        text=input_data.text,
        sentiment=sentiment,
        confidence=confidence,
        probabilities={
            "negative": probs[0][0].item(),
            "positive": probs[0][1].item()
        }
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(input_data: BatchInput):
    """
    Predict sentiment for multiple texts at once.
    More efficient than calling /predict multiple times.
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first to create checkpoints."
        )

    if len(input_data.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 texts"
        )

    # Tokenize all texts
    input_ids = torch.tensor([
        tokenizer.encode(text, MODEL_CONFIG["max_seq_len"])
        for text in input_data.texts
    ]).to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    # Build responses
    predictions = []
    for i, text in enumerate(input_data.texts):
        pred = preds[i].item()
        confidence = probs[i][pred].item()
        sentiment = "positive" if pred == 1 else "negative"

        predictions.append(PredictionResponse(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities={
                "negative": probs[i][0].item(),
                "positive": probs[i][1].item()
            }
        ))

    return BatchResponse(predictions=predictions)


@app.get("/model/info")
async def model_info():
    """Get model architecture information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "architecture": "Transformer Encoder",
        "config": MODEL_CONFIG,
        "vocab_size": len(tokenizer.word2idx),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device)
    }


# Serve React frontend (must be after API routes)
frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="static")

    @app.get("/ui")
    async def serve_frontend():
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
