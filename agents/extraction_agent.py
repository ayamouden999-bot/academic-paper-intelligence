"""
agents/extraction_agent.py — Extracts structured information from a research paper.
Prompt is dynamically tailored based on the field output from classifier_agent.
"""

from crewai import Agent
from config import GEMINI_MODEL
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))


def get_extraction_agent() -> Agent:
    return Agent(
        role="Research Paper Extraction Specialist",
        goal=(
            "Extract all key structured information from a research paper: "
            "research question, methodology, datasets, results, and limitations. "
            "Tailor the extraction depth based on the paper's research field."
        ),
        backstory=(
            "You are an expert research analyst who has reviewed thousands of "
            "academic papers. You know exactly what to look for in each discipline — "
            "a CS paper needs dataset details and benchmark comparisons, while a "
            "medicine paper needs patient cohort info and statistical significance. "
            "You extract information with surgical precision, never hallucinating facts."
        ),
        verbose=True,
        allow_delegation=False,
        llm=f"gemini/{GEMINI_MODEL}",
    )
