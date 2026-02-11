# Financial SMS Named Entity Recognition 

A production-ready Named Entity Recognition (NER) system fine-tuned on DistilBERT for extracting entities from SMS messages. Achieves **96.20% F1-score** on test data with robust preprocessing, training pipelines, and a Streamlit web interface.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

This project implements an end-to-end NER pipeline for SMS text analysis, featuring:

- **High Performance**: 96.20% F1-score, 96.23% precision, 96.18% recall
- **Production-Ready**: Complete training pipeline with logging, metrics tracking, and model versioning
- **Interactive UI**: Streamlit web application for real-time entity extraction
- **Robust Pipeline**: Data ingestion → Annotation → Transformation → Training → Evaluation
- **Efficient Model**: DistilBERT (40% smaller than BERT, 60% faster)

---

## Performance Metrics

Based on actual test results from `artifacts/ner_metrics.json`:

| Metric | Value |
|--------|-------|
| **Test F1-Score** | **96.20%** |
| Test Precision | 96.23% |
| Test Recall | 96.18% |
| Eval F1-Score | 96.13% |
| Training Loss | 0.161 |
| Eval Loss | 0.060 |
| Test Loss | 0.064 |
| Training Time | ~27 minutes (3 epochs) |
| Training Speed | 75.2 samples/sec |

### Training Curves

View the training loss curve at: `artifacts/ner_loss_curve.png`

---

## Project Structure

```
NER-Fine-tune-DistilBERT/
│
├── Data/
│   └── SMS-Data.csv              # Raw SMS dataset
│
├── artifacts/                     # Generated artifacts
│   ├── english_sms_texts.csv     # Processed SMS data
│   ├── annotated_sms.csv         # Annotated entities
│   ├── ner_splits/               # Train/val/test splits
│   ├── tokenized_dataset/        # Cached tokenized data
│   ├── ner_model/                # Trained model checkpoint
│   ├── best_ner_model/           # Best performing checkpoint
│   ├── ner_metrics.json          # Evaluation metrics
│   └── ner_loss_curve.png        # Training visualization
│
├── src/components/
│   ├── data_ingestion.py         # Load raw data
│   ├── data_annotation/          # Entity annotation tools
│   ├── data_transformation/      # Text preprocessing
│   ├── ner_dataset_prep.py       # Prepare NER datasets
│   ├── dataset_splitter.py       # Train/val/test splitting
│   ├── train_ner.py              # Tokenization & alignment
│   ├── fine_tuning.py            # Training pipeline
│   ├── logger.py                 # Logging utility
│   └── exception.py              # Custom exceptions
│
├── notebook/
│   └── 1_EDA.ipynb               # Exploratory data analysis
│
├── logs/                          # Training logs with timestamps
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Base Model** | distilbert-base-uncased |
| **Framework** | PyTorch |
| **Transformers** | Hugging Face Transformers + Accelerate |
| **Data Processing** | Pandas, NumPy, Datasets |
| **Evaluation** | seqeval (NER-specific metrics) |
| **NLP Tools** | NLTK, RapidFuzz |
| **Visualization** | Matplotlib, Seaborn, TensorBoard |
| **Deployment** | Streamlit |
| **Model Saving** | SafeTensors |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU training)
- 8GB RAM minimum (16GB recommended for training)

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/rkuma18/Fine-Tune-BERT-Model-for-NER.git
cd NER-Fine-tune-DistilBERT

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Dependencies (from requirements.txt)

```
pandas
numpy
torch
transformers[torch]
accelerate>=0.26.0
datasets
seqeval
scikit-learn
matplotlib
seaborn
tensorboard
nltk
rapidfuzz
tqdm
safetensors
packaging
streamlit
```

---

## Usage

### 1. Data Preparation

```bash
# Run data ingestion
python -m src.components.data_ingestion

# Preprocess and annotate (if needed)
# The pipeline handles data transformation automatically
```

### 2. Train the Model

The training pipeline is modular and handles:
- Data loading and splitting
- Tokenization with sub-word alignment
- Model fine-tuning with checkpointing
- Metrics logging and visualization

```bash
# Main training script
python -m src.components.fine_tuning

# Or use the complete training pipeline
python -m src.components.train_ner
```

**Key Training Features:**
- Automatic caching of tokenized datasets (in `artifacts/tokenized_dataset/`)
- Sub-word token alignment for BERT models
- Early stopping and best model checkpoint saving
- TensorBoard integration for real-time monitoring

```bash
# Monitor training with TensorBoard
tensorboard --logdir=logs
```

### 3. Model Inference

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Load the trained model
model = AutoModelForTokenClassification.from_pretrained("artifacts/best_ner_model")
tokenizer = AutoTokenizer.from_pretrained("artifacts/best_ner_model")

# Example SMS text
text = "Your account has been debited Rs.5000 at Amazon on 15-Jan-24"

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)

# Decode predictions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]

# Display results
for token, label in zip(tokens, predicted_labels):
    if label != "O":  # Skip 'Outside' labels
        print(f"{token}: {label}")
```

### 4. Streamlit Web App

Launch the interactive web interface for real-time entity extraction:

```bash
streamlit run streamlit_app.py
```

**App Features:**
- Real-time NER prediction on custom SMS text
- Visual highlighting of detected entities
- Confidence scores for predictions
- Batch processing support

---

## Model Architecture

```
Input SMS Text
      ↓
Tokenization (DistilBERT Tokenizer)
      ↓
DistilBERT Encoder (6 Transformer Layers)
      ↓
Token Classification Head
      ↓
Entity Labels (BIO Format)
```

**Model Details:**
- **Base Model**: `distilbert-base-uncased`
- **Parameters**: 66M (vs 110M for BERT-base)
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Layers**: 6
- **Max Sequence Length**: 512 tokens
- **Label Scheme**: BIO (Beginning-Inside-Outside)

---

## Training Configuration

```python
# Key hyperparameters (from fine_tuning.py)
MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512
WARMUP_STEPS = 500
```

---

## Evaluation Results

### Per-Entity Performance

The model was evaluated on multiple entity types commonly found in SMS messages. View detailed per-entity metrics in `artifacts/ner_metrics.json`.

### Confusion Matrix Analysis

Common prediction patterns:
- **High accuracy** on well-defined entities (amounts, dates)
- **Good performance** on contextual entities (merchant names, locations)
- **Minimal confusion** between similar entity types

---

## Key Features

### 1. Robust Data Pipeline
- **Data Ingestion**: Automated loading from `Data/SMS-Data.csv`
- **Annotation Tools**: Custom annotation pipeline in `src/components/data_annotation/`
- **Transformation**: Text cleaning, normalization, emoji removal
- **Smart Splitting**: Stratified train/val/test split preserving entity distributions

### 2. Advanced Training
- **Sub-word Alignment**: Proper handling of WordPiece tokenization
- **Label Smoothing**: Special tokens set to `IGNORED_LABEL_ID (-100)`
- **Caching**: Tokenized datasets cached for faster experimentation
- **Checkpointing**: Auto-save best model based on F1-score
- **Logging**: Comprehensive logging with timestamps in `logs/`

### 3. Production Features
- **Model Versioning**: Separate `ner_model/` and `best_ner_model/` folders
- **Metrics Tracking**: JSON export for reproducibility
- **Visualization**: Matplotlib plots for loss curves
- **Fast Inference**: Optimized for real-time predictions
- **Error Handling**: Custom exception classes for debugging

---

## Jupyter Notebooks

Explore the data and training process:

- `notebook/1_EDA.ipynb`: Exploratory Data Analysis
  - SMS length distribution
  - Entity frequency analysis
  - Data quality checks
  - Visualization of entity patterns

---

## Deployment

### Docker Deployment (Coming Soon)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

### API Endpoint (Coming Soon)

FastAPI wrapper for production deployment:

```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
ner_pipeline = pipeline("ner", model="artifacts/best_ner_model")

@app.post("/predict")
def predict_entities(text: str):
    entities = ner_pipeline(text)
    return {"entities": entities}
```

---

## Future Enhancements

- [ ] **Multi-Language Support**: Extend to Hindi, Spanish SMS
- [ ] **Entity Linking**: Connect extracted entities to knowledge bases
- [ ] **Active Learning**: Identify uncertain predictions for re-annotation
- [ ] **Model Quantization**: Reduce model size for mobile deployment
- [ ] **Ensemble Methods**: Combine multiple models for higher accuracy
- [ ] **Real-Time Streaming**: Process SMS streams with low latency
- [ ] **Cross-Lingual Transfer**: Fine-tune multilingual BERT variants
- [ ] **Explainability**: Add attention visualization for predictions

---

## Testing & Validation

### Run Evaluation

```bash
# Evaluate on test set
python -m src.components.fine_tuning --mode eval

# Results saved to: artifacts/ner_metrics.json
```

### Error Analysis

```python
# Analyze misclassifications
from src.components.fine_tuning import analyze_errors

errors = analyze_errors(
    test_dataset="artifacts/ner_splits/test.csv",
    model_path="artifacts/best_ner_model"
)

# Common error patterns:
# - Rare entities with limited training examples
# - Ambiguous contexts (e.g., "Apple" as company vs fruit)
# - Out-of-vocabulary tokens
```

---

## Model Card

**Model Name**: DistilBERT-NER-SMS
**Base Architecture**: distilbert-base-uncased
**Task**: Token Classification (Named Entity Recognition)
**Language**: English
**Training Data**: SMS messages with annotated entities
**Evaluation Metrics**: F1-Score (96.20%), Precision (96.23%), Recall (96.18%)
**Intended Use**: Extracting structured information from SMS text
**Limitations**: Performance may degrade on non-SMS text domains
**Ethical Considerations**: Should not be used on private SMS without consent

---

## Contributing

Contributions are welcome! Areas for improvement:

- Additional entity types (email addresses, URLs, phone numbers)
- Support for other languages
- Improved handling of abbreviations and slang
- Integration with downstream applications

**How to Contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-entity-type`)
3. Commit your changes (`git commit -m 'Add support for URL entities'`)
4. Push to the branch (`git push origin feature/new-entity-type`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Roushan Kumar**

- GitHub: [@rkuma18](https://github.com/rkuma18)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/rk0718/)
- Email: kumarroushan.18@gmail.com

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library and model hub
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) by Sanh et al. (2019)
- [seqeval](https://github.com/chakki-works/seqeval) for NER evaluation metrics
- The open-source community for dataset annotation tools

---

## References

1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

3. [Hugging Face Token Classification Guide](https://huggingface.co/docs/transformers/tasks/token_classification)

4. [Named Entity Recognition: A Survey](https://arxiv.org/abs/1812.09449)

---

## Support

For questions or issues:
- Open an issue on [GitHub Issues](https://github.com/rkuma18/Fine-Tune-BERT-Model-for-NER/issues)
- Contact: kumarroushan.18@gmail.com

---

**⭐ If this project helps your NER tasks, please give it a star on GitHub!**
