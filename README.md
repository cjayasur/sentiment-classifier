# Sentiment Classifier

A transformer-based sentiment classifier built from scratch with PyTorch, served via FastAPI.

## Architecture

- **Multi-head self-attention** with scaled dot-product attention
- **3 transformer encoder blocks** with residual connections and layer normalization
- **GELU activation** (like BERT/GPT)
- **Global average pooling** for sequence classification
- ~300K parameters

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Start the API server
python api.py
# or: uvicorn api:app --reload
```

## API Endpoints

Once the server is running at `http://localhost:8000`:

- `GET /` - Health check
- `POST /predict` - Single text prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model architecture info
- `GET /docs` - Interactive API documentation (Swagger UI)

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

## Project Structure

```
├── model.py          # Transformer architecture
├── train.py          # Training script with embedded dataset
├── api.py            # FastAPI server
├── requirements.txt  # Dependencies
└── checkpoints/      # Saved model weights (created after training)
```

## Built With

- PyTorch 2.x
- FastAPI
- Built from scratch - no pretrained models!
