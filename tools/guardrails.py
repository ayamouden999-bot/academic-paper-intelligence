"""
tools/guardrails.py — Input validation, error handling, and guardrails.
Prevents crashes from bad PDFs, empty abstracts, API failures.
"""

import os
import re
import logging

logger = logging.getLogger("APIS")


class PipelineError(Exception):
    """Raised when a guardrail blocks pipeline execution."""
    pass


def validate_pdf(pdf_path: str) -> dict:
    """
    Guardrail 1: Validate PDF before processing.
    Checks file exists, is readable, and is not empty.
    """
    errors = []

    if not os.path.exists(pdf_path):
        raise PipelineError(f"File not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise PipelineError(f"File must be a PDF: {pdf_path}")

    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if size_mb < 0.01:
        raise PipelineError(f"PDF appears empty (size: {size_mb:.2f}MB)")

    if size_mb > 50:
        raise PipelineError(f"PDF too large ({size_mb:.1f}MB). Max 50MB.")

    logger.info(f"[guardrail] PDF validated: {pdf_path} ({size_mb:.2f}MB)")
    return {"valid": True, "size_mb": round(size_mb, 2)}


def validate_extraction(paper: dict) -> dict:
    """
    Guardrail 2: Validate extracted paper content.
    Ensures abstract and text are usable.
    """
    issues = []

    if not paper.get("abstract") or len(paper["abstract"]) < 50:
        issues.append("Abstract too short or missing — PDF may be scanned/image-based")

    if not paper.get("full_text") or len(paper["full_text"]) < 200:
        issues.append("Could not extract readable text — PDF may be image-based")

    if not paper.get("title") or paper["title"] == "Unknown Title":
        issues.append("Could not extract paper title")
        # Non-fatal — continue with unknown title

    if issues:
        for issue in issues:
            logger.warning(f"[guardrail] Extraction issue: {issue}")
        # Fatal if text is unreadable
        if len(paper.get("full_text", "")) < 200:
            raise PipelineError(
                "Cannot process this PDF — text extraction failed. "
                "The PDF may be scanned or image-based. "
                "Please use a text-based PDF."
            )

    logger.info(f"[guardrail] Extraction validated: {len(paper['full_text'])} chars")
    return {"valid": True, "issues": issues}


def validate_classification(classification: dict) -> dict:
    """
    Guardrail 3: Check classification confidence.
    Flags very low confidence results.
    """
    warnings = []

    field_conf = classification.get("field_confidence", 0)
    novelty_conf = classification.get("novelty_confidence", 0)

    if field_conf < 0.25:
        warnings.append(
            f"Very low field confidence ({field_conf*100:.1f}%) — "
            "classification may be unreliable"
        )
        classification["requires_human_review"] = True

    if novelty_conf < 0.40:
        warnings.append(
            f"Low novelty confidence ({novelty_conf*100:.1f}%) — "
            "human review recommended"
        )
        classification["requires_human_review"] = True

    for w in warnings:
        logger.warning(f"[guardrail] Classification warning: {w}")

    return {"valid": True, "warnings": warnings}


def sanitize_text(text: str, max_chars: int = 10000) -> str:
    """
    Guardrail 4: Clean and truncate text before sending to LLM.
    Prevents token overflow and removes junk characters.
    """
    # Remove null bytes and non-printable chars
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text.strip()


def safe_api_call(func, *args, max_retries: int = 2, **kwargs):
    """
    Guardrail 5: Wrap any API call with retry logic.
    Handles transient failures gracefully.
    """
    import time

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Don't retry on auth errors
            if "401" in error_str or "403" in error_str or "api key" in error_str:
                logger.error(f"[guardrail] Auth error — not retrying: {e}")
                raise

            if attempt < max_retries:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s
                logger.warning(f"[guardrail] API call failed (attempt {attempt+1}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"[guardrail] API call failed after {max_retries+1} attempts: {e}")

    raise PipelineError(f"API call failed after {max_retries+1} attempts: {last_error}")