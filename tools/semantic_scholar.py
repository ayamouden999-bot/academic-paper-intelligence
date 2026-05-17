"""
tools/semantic_scholar.py — Free citation search via Semantic Scholar API.
No API key required for basic usage.
Input:  paper title or keywords
Output: list of related papers with citation counts
"""

import requests
import time
from config import SEMANTIC_SCHOLAR_BASE_URL, SEMANTIC_SCHOLAR_MAX_RESULTS
from typing import Optional

def search_related_papers(query: str, limit: int = SEMANTIC_SCHOLAR_MAX_RESULTS) -> list[dict]:
    """
    Search Semantic Scholar for papers related to a query.
    Returns a list of dicts with title, authors, year, citations, url.
    """
    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,citationCount,externalIds,abstract",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        papers = []
        for paper in data.get("data", []):
            papers.append({
                "title":        paper.get("title", "N/A"),
                "authors":      [a["name"] for a in paper.get("authors", [])[:3]],
                "year":         paper.get("year", "N/A"),
                "citations":    paper.get("citationCount", 0),
                "abstract":     (paper.get("abstract") or "")[:300],
                "arxiv_id":     paper.get("externalIds", {}).get("ArXiv", None),
            })
        return papers

    except requests.RequestException as e:
        print(f"[SemanticScholar] Request failed: {e}")
        return []


def get_paper_by_title(title: str) -> Optional[dict]:
    """Look up a specific paper by exact title."""
    results = search_related_papers(title, limit=1)
    return results[0] if results else None


def check_seminal_references(references: list[str]) -> dict:
    """
    Given a list of reference strings, check how many are highly cited.
    Returns a summary useful for the critique agent.
    """
    seminal_count = 0
    checked = []

    for ref in references[:10]:  # check first 10 to avoid rate limits
        time.sleep(0.5)          # be polite to the API
        results = search_related_papers(ref[:100], limit=1)
        if results and results[0]["citations"] > 500:
            seminal_count += 1
            checked.append({
                "reference": ref[:80],
                "citations": results[0]["citations"],
                "is_seminal": True,
            })

    return {
        "seminal_references_found": seminal_count,
        "total_checked": len(references[:10]),
        "details": checked,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = search_related_papers("transformer attention mechanism natural language processing")
    for r in results:
        print(f"[{r['year']}] {r['title']} — {r['citations']} citations")
