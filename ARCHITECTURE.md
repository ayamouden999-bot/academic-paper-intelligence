# Academic Paper Intelligence System
## Architecture Document — Week 1 Deliverable

**Team:** [Your names here]  
**Course:** AI & Big Data — S8 Integrated Project  
**Professor:** Prof. Hakim Hafidi  
**Date:** [Date]

---

## 1. System Overview

The Academic Paper Intelligence System is a multi-agent AI pipeline that analyzes research papers end-to-end. A user uploads any research PDF and receives a structured, peer-review style critique — covering field classification, novelty assessment, methodology extraction, citation analysis, and a final accept/revise/reject recommendation.

The system is designed around three core principles:
- **Every component exists for a reason** — no agent is redundant
- **The DL model is load-bearing** — both its outputs (field + novelty) directly control downstream agent behavior
- **Human oversight is genuine** — the HITL checkpoint is not cosmetic; it gates the most expensive part of the pipeline

---

## 2. Agent Roles & Responsibilities

| Agent | Role | Primary Tool | Input | Output |
|-------|------|-------------|-------|--------|
| **Classifier Agent** | Classifies paper field and novelty | TextCNN (PyTorch) | Abstract text | Field label + novelty score + confidence |
| **Extraction Agent** | Extracts structured info | Gemini 1.5 Flash | Full text + field label | Research Q, methodology, findings, limitations |
| **Citation Agent** | Validates references | Semantic Scholar API | Title + reference list | Missing citations, seminal works found |
| **Critique Agent** | Writes peer review | Gemini 1.5 Flash | All above outputs | Full structured critique + verdict |

**Orchestrator:** CrewAI sequential process manager. Routes outputs between agents and enforces the HITL checkpoint.

---

## 3. Communication Diagram

```
User (PDF upload)
       │
       ▼
  [PDF Reader Tool]
  Extracts: title, abstract, full text, references
       │
       ▼
  [Classifier Agent]  ◄── TextCNN Model (PyTorch) ──┐
  Outputs:                                            │
    - Research field (8 classes)                      │  Dual-output
    - Novelty level (4 classes)                       │  head design
    - Confidence scores                               │
    - requires_human_review flag                      │
       │
       ▼ (if requires_human_review = True)
  ┌─────────────────────┐
  │  HUMAN CHECKPOINT   │  ← Human approves or stops pipeline
  └─────────────────────┘
       │ (approved)
       ▼
  [Extraction Agent]  ◄── Gemini 1.5 Flash
  Uses field label to tailor extraction prompt
  Outputs structured paper analysis
       │
       ├──────────────────────────┐
       ▼                          ▼
  [Citation Agent]          [Continue to Critique]
  Semantic Scholar API
  Validates references
       │
       └──────────────────────────┐
                                  ▼
                          [Critique Agent]  ◄── Gemini 1.5 Flash
                          Synthesizes all outputs
                          Produces: peer-review critique
                                  │
                                  ▼
                          JSON log + Markdown report
                          saved to outputs/
```

---

## 4. DL Model Design Rationale

### Why TextCNN?

TextCNN (Kim, 2014) was chosen for three reasons:

1. **Efficiency** — Trains in ~20 minutes on Colab free T4 GPU. No risk of session timeout.
2. **Interpretability** — Filter activations can be visualized and explained during the oral defense. Unlike BERT, every component is explainable.
3. **Custom architecture** — We build it from scratch in PyTorch, demonstrating genuine DL knowledge rather than wrapping a pre-trained model.

### Why dual-output?

The model produces **two simultaneous outputs** from a shared feature representation:

- **Head 1: Research field** (8 classes) — used to tailor the extraction agent's prompt. A CS paper gets asked about datasets and benchmarks; a medicine paper gets asked about patient cohorts and p-values.
- **Head 2: Novelty level** (4 classes) — used as the HITL trigger. "Incremental" papers are flagged for human review before the expensive Gemini extraction runs.

This makes the DL model **load-bearing**, not decorative. Remove it and the pipeline breaks.

### Architecture details

```
Input (abstract tokens)
    → Embedding layer (128-dim)
    → Parallel Conv1D filters (bigram, trigram, 4-gram)
    → ReLU activation
    → Max-over-time pooling
    → Concatenation (300-dim feature vector)
    → Dropout (0.5)
    ├→ Head 1: Linear(300→128) → ReLU → Linear(128→8)  [field]
    └→ Head 2: Linear(300→64)  → ReLU → Linear(64→4)   [novelty]
```

Total trainable parameters: ~4.2M

---

## 5. Dataset Choice Rationale

**Dataset:** `ccdv/arxiv-classification` (HuggingFace)

- **Size:** ~50,000 research paper abstracts
- **Labels:** 8 research fields (CS, Economics, EE, Math, Physics, Q-Bio, Q-Finance, Statistics)
- **Pre-split:** train/validation/test — no manual splitting needed
- **Quality:** Clean, curated from arXiv, no preprocessing needed
- **Access:** 2 lines of code via HuggingFace `datasets` library

**Novelty labels:** Generated via keyword heuristics on the abstract (Week 1). Optionally refined with a small human-annotated subset (Week 2 stretch goal).

---

## 6. HITL Checkpoint Justification

The human checkpoint is placed **between the Classifier Agent and the Extraction Agent** for two reasons:

1. **Cost efficiency** — Gemini API calls cost tokens. Running extraction on a paper the classifier is uncertain about wastes resources. The human verifies the classification is correct before committing.

2. **Genuine necessity** — A paper classified as "Incremental" novelty will receive a more critical critique. This judgment affects the tone of the entire downstream pipeline, making human verification genuinely important.

**Trigger conditions:**
- Novelty confidence < 65% (model is uncertain)
- Novelty predicted as "Incremental" (consequential decision)

---

## 7. Tool Input/Output Schemas

### PDF Reader Tool
```
Input:  pdf_path: str
Output: {
  title:         str,
  abstract:      str,
  full_text:     str,
  references:    list[str],
  num_pages:     int,
  char_count:    int,
  classify_input: str,  # abstract or first 1500 chars
}
```

### TextCNN Classifier Tool
```
Input:  text: str (abstract, max 256 tokens)
Output: {
  field:                str,    # e.g. "Computer Science"
  field_confidence:     float,  # 0.0 - 1.0
  novelty:              str,    # e.g. "High"
  novelty_confidence:   float,
  requires_human_review: bool,
}
```

### Semantic Scholar Tool
```
Input:  query: str (title or keywords)
Output: list[{
  title:    str,
  authors:  list[str],
  year:     int,
  citations: int,
  abstract: str,
}]
```

---

## 8. Known Limitations & Mitigations

| Limitation | Mitigation |
|-----------|------------|
| Novelty labels are heuristic-based (keyword matching) | Manual annotation of 200-300 examples in Week 2 for fine-tuning |
| PDF text extraction fails on scanned/image PDFs | Error handling fallback with user-facing message |
| Gemini API rate limits on free tier | Retry logic with exponential backoff in all API calls |
| TextCNN has no cross-lingual support | System is English-only; stated as known limitation in report |
| Semantic Scholar API rate limit (100 req/5min) | 0.5s delay between calls; caching for repeated queries |
