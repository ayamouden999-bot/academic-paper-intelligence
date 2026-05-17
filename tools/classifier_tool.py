"""
tools/classifier_tool.py — Wraps the trained TextCNN as a CrewAI-compatible tool.
Week 1: returns a placeholder result.
Week 2: loads and runs the real trained model.
"""

import torch
import os
from config import (
    MODEL_SAVE_PATH, FIELD_LABELS, NOVELTY_LABELS,
    NOVELTY_CONFIDENCE_THRESHOLD, MAX_SEQ_LEN
)


def classify_paper(text: str) -> dict:
    """
    Classify a research paper abstract.
    Returns:
        field:            predicted research field (string)
        field_confidence: confidence score 0-1
        novelty:          predicted novelty level (string)
        novelty_confidence: confidence score 0-1
        requires_human_review: bool — True if confidence is low
    """

    # ── Week 2: swap this block with real model inference ─────────────────────
    if not os.path.exists(MODEL_SAVE_PATH):
        print("[ClassifierTool] Model not trained yet — returning placeholder.")
        return _placeholder_result(text)

    # ── Real inference (Week 2+) ──────────────────────────────────────────────
    from models.textcnn import TextCNN
    from models.train import text_to_tensor

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = TextCNN()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    model.to(device)

    with torch.no_grad():
        tensor = text_to_tensor(text[:MAX_SEQ_LEN]).unsqueeze(0).to(device)
        field_logits, novelty_logits = model(tensor)

        field_probs   = torch.softmax(field_logits, dim=1)
        novelty_probs = torch.softmax(novelty_logits, dim=1)

        field_conf, field_idx     = field_probs.max(dim=1)
        novelty_conf, novelty_idx = novelty_probs.max(dim=1)

    field_confidence   = field_conf.item()
    novelty_confidence = novelty_conf.item()

    return {
        "field":               FIELD_LABELS[field_idx.item()],
        "field_confidence":    round(field_confidence, 3),
        "novelty":             NOVELTY_LABELS[novelty_idx.item()],
        "novelty_confidence":  round(novelty_confidence, 3),
        # HITL trigger: low confidence OR Incremental novelty
        "requires_human_review": (
            novelty_confidence < NOVELTY_CONFIDENCE_THRESHOLD
            or NOVELTY_LABELS[novelty_idx.item()] == "Incremental"
        ),
    }


def _placeholder_result(text: str) -> dict:
    """Week 1 placeholder — rule-based heuristic for testing the pipeline."""
    text_lower = text.lower()
    field = "Computer Science"
    if any(w in text_lower for w in ["neural", "deep learning", "transformer", "cnn", "lstm"]):
        field = "Computer Science"
    elif any(w in text_lower for w in ["patient", "clinical", "disease", "therapy"]):
        field = "Medicine"
    elif any(w in text_lower for w in ["quantum", "particle", "relativity"]):
        field = "Physics"
    elif any(w in text_lower for w in ["market", "portfolio", "gdp", "inflation"]):
        field = "Economics"

    return {
        "field":               field,
        "field_confidence":    0.72,   # placeholder
        "novelty":             "Moderate",
        "novelty_confidence":  0.68,   # placeholder
        "requires_human_review": False,
        "note": "Placeholder result — train TextCNN in Week 2 to get real predictions.",
    }
