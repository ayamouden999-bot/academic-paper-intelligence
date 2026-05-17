"""
agents/citation_agent.py — Searches for related papers and validates citations.
Uses Semantic Scholar API as its tool (free, no key needed).
"""

from crewai import Agent
from config import GEMINI_MODEL
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))


def get_citation_agent() -> Agent:
    return Agent(
        role="Citation and Literature Analyst",
        goal=(
            "Search for related papers in the literature, verify that key "
            "seminal works are cited, and identify any important missing references "
            "that would strengthen the paper."
        ),
        backstory=(
            "You are a research librarian and literature expert who lives in "
            "academic databases. You know the seminal papers in every field and "
            "can immediately spot when an author has missed a crucial citation. "
            "You use the Semantic Scholar API to ground your analysis in real data, "
            "never making up paper titles or authors."
        ),
        verbose=True,
        allow_delegation=False,
        llm=f"gemini/{GEMINI_MODEL}",
    )
