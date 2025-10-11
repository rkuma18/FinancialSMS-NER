"""FastAPI service that wraps the fine-tuned NER model for inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer
from starlette.responses import JSONResponse

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
MODEL_DIRS: List[Path] = [Path("artifacts/best_ner_model"), Path("artifacts/ner_model")]

# --- Pydantic Models (Data Schema) ---
class NERRequest(BaseModel):
    """Schema for the input text payload."""
    text: str = Field(..., min_length=1, max_length=1024, description="Text to analyze (Max 1024 chars)")

class NEREntity(BaseModel):
    """Schema for a single extracted entity (matches pipeline output structure)."""
    entity_group: str
    start: int
    end: int
    word: str
    score: float

class NERResponse(BaseModel):
    """Schema for the final prediction response."""
    text: str
    entities: List[NEREntity]


# --- Service Initialization ---
def load_pipeline() -> Pipeline:
    """Load NER pipeline from the first available model directory."""
    logger.info("Attempting to load NER model pipeline...")

    # 1. Find the model path
    try:
        model_path = next(path for path in MODEL_DIRS if path.is_dir())
    except StopIteration:
        error_msg = f"Model directory not found in {MODEL_DIRS}. Please ensure artifacts are mounted."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Loading model and tokenizer from: {model_path}")
    
    # 2. Load components and create pipeline
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        return pipeline(
            "ner", 
            model=model, 
            tokenizer=tokenizer, 
            aggregation_strategy="simple"
        )
    except Exception as e:
        error_msg = f"Failed to load model components from {model_path}: {e}"
        logger.error(error_msg, exc_info=True)
        # Re-raise as RuntimeError to stop the app if loading fails
        raise RuntimeError(error_msg) from e

# Load the model once at startup
try:
    ner_pipeline: Pipeline = load_pipeline()
except RuntimeError:
    # If loading fails, set pipeline to None and let the health check handle the 503 error
    ner_pipeline = None
    logger.critical("NER pipeline failed to load. The prediction endpoint will be unavailable.")

app = FastAPI(title="NER Inference API", version="1.0.1")


# --- Endpoints ---
@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint. Confirms API is running and model is loaded."""
    if ner_pipeline is None:
        # If model failed to load at startup, return Service Unavailable (503)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": "Model not loaded successfully. Check logs."}
        )
    return {"status": "ok", "model_status": "ready"}


@app.post("/predict", response_model=NERResponse)
def predict(req: NERRequest) -> NERResponse:
    """Extract named entities from input text."""
    
    # 1. Critical Check: Is model loaded?
    if ner_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The NER model is not loaded. Please check server status."
        )

    text = req.text.strip()
    
    # Pydantic handles the empty text check, but an explicit check is good for custom detail
    if not text:
        # Pydantic validation error will be 422, but this custom check gives 400
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input text must not be empty.")

    logger.info(f"Processing request for text: '{text[:50]}...'")
    
    try:
        # 2. Inference
        predictions: List[Dict[str, Any]] = ner_pipeline(text)
        
        # 3. Response Formatting (Mapping pipeline output to Pydantic schema)
        entities = [
            NEREntity(
                entity_group=pred["entity_group"],
                start=pred["start"],
                end=pred["end"],
                word=pred["word"],
                score=pred["score"],
            )
            for pred in predictions
        ]
        
        logger.info(f"Successfully extracted {len(entities)} entities.")
        
        return NERResponse(text=text, entities=entities)

    except Exception as e:
        logger.error(f"Prediction failed. Error: {e}", exc_info=True)
        # Handle general inference failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during model inference."
        )