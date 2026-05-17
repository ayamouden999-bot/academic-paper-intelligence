"""
agents/classifier_agent.py — Specialist agent that classifies research papers.
Uses the TextCNN model as its primary tool.
Week 1: uses placeholder classifier.
Week 2: uses real trained TextCNN.
"""

from crewai import Agent
from tools.classifier_tool import classify_paper
from config import GEMINI_MODEL
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))


def get_classifier_agent() -> Agent:
    return Agent(
        role="Research Paper Classifier",
        goal=(
            "Accurately classify a research paper's field and novelty level "
            "using the TextCNN deep learning model, and flag papers that "
            "require human review."
        ),
        backstory=(
            "You are a specialist in academic paper analysis with deep expertise "
            "across all scientific disciplines. You use a custom-trained TextCNN "
            "neural network to classify papers by field and assess their novelty. "
            "You are precise, thorough, and always flag uncertain classifications "
            "for human review rather than guessing."
        ),
        verbose=True,
        allow_delegation=False,
        llm=f"gemini/{GEMINI_MODEL}",
    )
