"""FastAPI service for Named Entity Recognition inference."""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---

# Model path is absolute inside the Docker container
MODEL_DIRS: List[Path] = [Path("/app/artifacts/best_ner_model")] 

# --- Model Loading Logic ---

def load_ner_pipeline():
    """Tries to load the NER model from predefined paths."""
    for model_path in MODEL_DIRS:
        if model_path.exists():
            logger.info(f"Attempting to load NER model pipeline from: {model_path}")
            try:
                # Load the pipeline using the local directory
                pipe = pipeline(
                    "ner", 
                    model=str(model_path), 
                    tokenizer=str(model_path),
                    aggregation_strategy="simple"
                )
                logger.info("Successfully loaded NER model pipeline.")
                return pipe
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
    
    logger.error(f"Model directory not found in {[str(p) for p in MODEL_DIRS]}. Please ensure artifacts are mounted.")
    return None

NER_PIPE = load_ner_pipeline()

# --- FastAPI App Setup and Schemas (Same as before) ---
app = FastAPI(
    title="Financial SMS NER Service",
    description="FastAPI service for Transactional SMS Named Entity Recognition.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    text: str = Field(..., description="The transactional SMS text to analyze.")

class Entity(BaseModel):
    entity_group: str = Field(..., description="The type of entity.")
    score: float = Field(..., description="The confidence score.")
    word: str = Field(..., description="The exact text snippet identified.")
    start: int = Field(..., description="Starting character index.")
    end: int = Field(..., description="Ending character index.")

class PredictionResponse(BaseModel):
    entities: List[Entity] = Field(..., description="List of detected entities.")

# --- Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint for health check."""
    return {"message": "NER Inference Service is operational."}

@app.post("/predict", response_model=PredictionResponse)
def predict_ner(request: PredictionRequest):
    """Endpoint to perform NER prediction."""
    if NER_PIPE is None:
        logger.critical("Prediction attempt failed: Model is not loaded.")
        raise HTTPException(
            status_code=503,
            detail="The NER model is not loaded. Please check server status."
        )

    results = NER_PIPE(request.text)
    return {"entities": results}