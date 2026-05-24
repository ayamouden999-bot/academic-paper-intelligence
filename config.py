"""
config.py — Central configuration for Academic Paper Intelligence System
All constants, paths, and settings live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "textcnn_trained.pth")

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME   = "ccdv/arxiv-classification"
MAX_SEQ_LEN    = 256        # max tokens fed to TextCNN
VOCAB_SIZE     = 30_000

# ── TextCNN Hyperparameters ───────────────────────────────────────────────────
EMBED_DIM      = 128
NUM_FILTERS    = 100
FILTER_SIZES   = [2, 3, 4]  # n-gram sizes
DROPOUT        = 0.5
BATCH_SIZE     = 64
NUM_EPOCHS     = 10
LEARNING_RATE  = 1e-3

# ── Research Fields (arXiv labels) ────────────────────────────────────────────
FIELD_LABELS = {
    0:  "Computer Science",
    1:  "Economics",
    2:  "Electrical Engineering",
    3:  "Mathematics",
    4:  "Physics",
    5:  "Quantitative Biology",
    6:  "Quantitative Finance",
    7:  "Statistics",
    8:  "CS - Other",
    9:  "Physics - Other",
    10: "Math - Other",
}
NUM_FIELDS = 11

# ── Novelty Score Thresholds ──────────────────────────────────────────────────
# TextCNN confidence below this triggers the HITL checkpoint
NOVELTY_CONFIDENCE_THRESHOLD = 0.65

NOVELTY_LABELS = {
    0: "Incremental",
    1: "Moderate",
    2: "High",
    3: "Breakthrough",
}

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "groq/llama-3.1-8b-instant"
# ── Semantic Scholar ──────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_MAX_RESULTS = 5

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT     = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
