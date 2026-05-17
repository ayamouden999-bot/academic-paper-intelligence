"""
agents/critique_agent.py — Synthesizes all agent outputs into a peer-review critique.
This is the final agent in the pipeline. It produces the structured report.
"""

from crewai import Agent
from config import GEMINI_MODEL
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))


def get_critique_agent() -> Agent:
    return Agent(
        role="Peer Review Critique Writer",
        goal=(
            "Synthesize all analysis from previous agents into a structured, "
            "fair, and actionable peer-review style critique. Produce a final "
            "accept / major revision / reject recommendation with clear justification."
        ),
        backstory=(
            "You are a senior academic reviewer who has published in top venues "
            "and reviewed for Nature, NeurIPS, ICML, and JAMA. You write critiques "
            "that are honest, specific, and constructive — never vague. You always "
            "back up every claim with evidence from the paper. Your reviews are "
            "known for being the most useful ones authors receive."
        ),
        verbose=True,
        allow_delegation=False,
        llm=f"gemini/{GEMINI_MODEL}",
    )
