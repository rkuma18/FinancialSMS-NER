"""FastAPI service for Named Entity Recognition inference."""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure you have the transformers and torch libraries installed
from transformers import pipeline

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---

# Fix: Model ab container ke root mein /app/artifacts/ par hai
MODEL_DIRS: List[Path] = [Path("/app/artifacts/best_ner_model")] 
# Note: Agar aapne 'artifacts' ke andar 'ner_model' bhi rakha hai toh use bhi add kar sakte hain:
# MODEL_DIRS: List[Path] = [Path("/app/artifacts/best_ner_model"), Path("/app/artifacts/ner_model")]


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
                # Try next path if one fails
    
    # Agar saare paths fail ho jayein
    logger.error(f"Model directory not found in {[str(p) for p in MODEL_DIRS]}. Please ensure artifacts are mounted.")
    return None

# Global variable to hold the loaded model
NER_PIPE = load_ner_pipeline()

# --- FastAPI App Setup ---
app = FastAPI(
    title="Financial SMS NER Service",
    description="FastAPI service for Transactional SMS Named Entity Recognition.",
    version="1.0.0",
)

# --- Schemas ---
class PredictionRequest(BaseModel):
    """Schema for the input SMS text."""
    text: str = Field(..., description="The transactional SMS text to analyze.")

class Entity(BaseModel):
    """Schema for a detected entity."""
    entity_group: str = Field(..., description="The type of entity (e.g., AMOUNT, ACCOUNT).")
    score: float = Field(..., description="The confidence score of the prediction.")
    word: str = Field(..., description="The exact text snippet identified.")
    start: int = Field(..., description="Starting character index.")
    end: int = Field(..., description="Ending character index.")

class PredictionResponse(BaseModel):
    """Schema for the prediction result."""
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

    # Perform inference
    results = NER_PIPE(request.text)
    
    # Clean up and return
    return {"entities": results}