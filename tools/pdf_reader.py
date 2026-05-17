"""
tools/pdf_reader.py — Extracts structured text from research PDFs.
Input:  path to a .pdf file
Output: dict with title, abstract, sections, full_text
"""

import re
import PyPDF2
from typing import Optional


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract title, abstract, and full text from a research PDF.
    Returns a structured dict ready for downstream agents.
    """
    raw_text = ""

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"

    # ── Extract abstract ──────────────────────────────────────────────────────
    abstract = _extract_abstract(raw_text)

    # ── Extract title (first non-empty line heuristic) ────────────────────────
    title = _extract_title(raw_text)

    # ── Extract references section ────────────────────────────────────────────
    references = _extract_references(raw_text)

    return {
        "title":       title,
        "abstract":    abstract,
        "full_text":   raw_text,
        "references":  references,
        "num_pages":   num_pages,
        "char_count":  len(raw_text),
        # Use abstract for classification — shorter = faster TextCNN inference
        "classify_input": abstract if abstract else raw_text[:1500],
    }


def _extract_abstract(text: str) -> str:
    """Pull the abstract block from the paper."""
    patterns = [
        r'(?i)abstract[:\s\-–]+(.+?)(?=\n\n|\n1[\.\s]|\nIntroduction|\nKeywords)',
        r'(?i)abstract\n(.+?)(?=\n\n)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Clean up hyphenated line breaks common in PDFs
            abstract = re.sub(r'-\n', '', abstract)
            abstract = re.sub(r'\n', ' ', abstract)
            return abstract[:2000]  # cap at 2000 chars
    # Fallback: first 1000 chars
    return text[:1000].strip()


def _extract_title(text: str) -> str:
    """Extract full multi-line title from research paper."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    title_lines = []
    for i, line in enumerate(lines[:15]):
        # Skip very short lines, emails, dates, arxiv ids
        if len(line) < 8:
            continue
        if any(skip in line.lower() for skip in [
            '@', 'arxiv', 'abstract', 'introduction',
            'university', 'department', 'january', 'february',
            'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december',
            'submitted', 'preprint', 'doi'
        ]):
            continue
        
        # Start collecting title lines
        if not title_lines and len(line) > 10:
            title_lines.append(line)
        elif title_lines:
            # Continue if line looks like a title continuation
            # (no period at end, reasonable length, not an author line)
            if (len(line) > 5 and 
                not line.endswith('.') and
                not re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', line) and  # not "Firstname Lastname"
                len(title_lines) < 4):
                title_lines.append(line)
            else:
                break
    
    return ' '.join(title_lines) if title_lines else lines[0] if lines else "Unknown Title"


def _extract_references(text: str) -> list[str]:
    """Extract reference list from the end of the paper."""
    match = re.search(
        r'(?i)(references|bibliography)\n(.+)$', text, re.DOTALL
    )
    if match:
        ref_block = match.group(2)
        refs = [r.strip() for r in re.split(r'\n\[?\d+\]?\.?\s', ref_block) if r.strip()]
        return refs[:30]  # cap at 30 references
    return []


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tools/pdf_reader.py path/to/paper.pdf")
    else:
        result = extract_text_from_pdf(sys.argv[1])
        print(f"Title:    {result['title']}")
        print(f"Pages:    {result['num_pages']}")
        print(f"Abstract: {result['abstract'][:300]}...")
        print(f"Refs found: {len(result['references'])}")
