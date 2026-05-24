"""
main.py — Academic Paper Intelligence System
Full multi-agent pipeline with orchestrator + HITL
"""

import os, re, json, datetime, logging
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

from tools.pdf_reader       import extract_text_from_pdf
from tools.classifier_tool  import classify_paper
from tools.semantic_scholar import search_related_papers
from tools.guardrails import (
    validate_pdf, validate_extraction,
    validate_classification, sanitize_text,
    safe_api_call, PipelineError
)
from config import GEMINI_MODEL, LOGS_DIR, OUTPUTS_DIR, FIELD_LABELS

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "system.log")),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("APIS")


# ── Logging helper ────────────────────────────────────────────────────────────
def log_action(agent_name: str, action: str, data: dict):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "agent":     agent_name,
        "action":    action,
        **data,
    }
    fname = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{agent_name}.json"
    with open(os.path.join(LOGS_DIR, fname), "w") as f:
        json.dump(entry, f, indent=2)
    logger.info(f"[{agent_name}] {action}")
    return entry


# ── Field extractor ───────────────────────────────────────────────────────────
def extract_corrected_field(result_text: str) -> str:
    """Extract the corrected research field from Agent 1's output."""
    text = str(result_text)

    # Check known fields first — most reliable
    known_fields = [
        "Internet of Things", "Mathematical Physics", "Theoretical Physics",
        "Computer Science", "Machine Learning", "Artificial Intelligence",
        "Pure Mathematics", "Applied Mathematics", "Electrical Engineering",
        "Mechanical Engineering", "Biology", "Economics", "Statistics",
        "Quantitative Finance", "Astrophysics", "Chemistry", "Medicine",
        "Data Science", "Cloud Computing", "Software Engineering",
    ]
    for field in known_fields:
        if field.lower() in text.lower():
            return field

    # Pattern: "Corrected Field: X"
    match = re.search(r'(?i)corrected field[:\s]+([A-Z][^,\n\.]{3,50})', text)
    if match:
        return match.group(1).strip().rstrip('.,)')

    # Pattern: "field of X" or "field is X"
    match = re.search(r'(?i)field (?:of|is)[:\s]+([A-Z][^,\n\.]{3,50})', text)
    if match:
        field = match.group(1).strip().rstrip('.,)')
        if len(field) < 60 and '\n' not in field:
            return field

    return "See Classifier Agent analysis below"


# ── HITL checkpoint ───────────────────────────────────────────────────────────
def human_checkpoint(classification: dict, paper_title: str) -> bool:
    print("\n" + "="*65)
    print("🛑  HUMAN REVIEW CHECKPOINT")
    print("="*65)
    print(f"  Paper:   {paper_title[:60]}")
    print(f"  TextCNN Field (preliminary): {classification['field']}")
    print(f"  Confidence:                  {classification['field_confidence']*100:.1f}%")
    print(f"  Novelty:                     {classification['novelty']} ({classification['novelty_confidence']*100:.1f}%)")
    print("="*65)
    print("  ⚠️  Note: The Classifier Agent will verify and correct")
    print("  the field label using reasoning on top of the TextCNN.")
    print()
    print("  Please approve to continue to the full analysis.")
    print()

    while True:
        response = input("  Type 'yes' to proceed or 'no' to stop: ").strip().lower()
        if response in ["yes", "no"]:
            break
        print("  Please type 'yes' or 'no'")

    approved = response == "yes"
    log_action("human_checkpoint", "decision", {
        "paper_title":    paper_title,
        "classification": classification,
        "approved":       approved,
    })

    if approved:
        print("  ✅ Approved — Classifier Agent will now verify the field...\n")
    else:
        print("  ❌ Rejected — pipeline stopped.\n")

    return approved


# ── Agent definitions ─────────────────────────────────────────────────────────
def make_agents():
    llm = GEMINI_MODEL

    classifier_agent = Agent(
        role="Research Paper Classifier",
        goal=(
            "Determine the correct research field of the paper. "
            "Start your response with 'Corrected Field: [field name]'. "
            "Be concise — maximum 150 words total."
        ),
        backstory=(
            "Expert classifier. You override incorrect TextCNN predictions "
            "using abstract reasoning. Always output corrected field first."
        ),
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        llm=llm,
    )

    extraction_agent = Agent(
        role="Research Paper Extraction Specialist",
        goal=(
            "Extract key information from the paper in maximum 200 words. "
            "Use the CORRECTED field from the Classifier Agent."
        ),
        backstory=(
            "Expert analyst. Extract concisely: research question, "
            "methodology, 3 key results, limitations, reproducibility."
        ),
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        llm=llm,
    )

    citation_agent = Agent(
        role="Citation and Literature Analyst",
        goal=(
            "Analyze citations in maximum 150 words. "
            "Give citation quality score 1-10 with brief justification."
        ),
        backstory=(
            "Expert librarian. Identify missing citations and assess "
            "recency and coverage briefly."
        ),
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        llm=llm,
    )

    critique_agent = Agent(
        role="Peer Review Critique Writer",
        goal=(
            "Write a structured peer review with exactly these sections: "
            "## Summary, ## Strengths, ## Weaknesses, ## Citation Assessment, "
            "## Reproducibility Score, ## Novelty Assessment, ## Final Verdict. "
            "Keep each section to 2-3 sentences maximum."
        ),
        backstory=(
            "Senior academic reviewer. Write concise, evidence-backed reviews. "
            "Final verdict must be exactly one of: Accept / Major Revision / Reject."
        ),
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        llm=llm,
    )

    return classifier_agent, extraction_agent, citation_agent, critique_agent


# ── Task definitions ──────────────────────────────────────────────────────────
def make_tasks(agents, paper, classification, related_papers):
    classifier_agent, extraction_agent, citation_agent, critique_agent = agents

    # Truncate inputs to reduce token usage
    abstract  = paper['abstract'][:500]
    excerpt   = paper['full_text'][:800]
    refs      = paper['references'][:5]

    task1 = Task(
        description=f"""Paper: {paper['title']}
Abstract: {abstract}
TextCNN preliminary: {classification['field']} ({classification['field_confidence']*100:.1f}% — may be wrong)
Start with 'Corrected Field: [name]'. Assess novelty. List 3 key terms. Max 150 words.""",
        agent=classifier_agent,
        expected_output="Corrected Field stated first, novelty, 3 key terms. Under 150 words.",
    )

    task2 = Task(
        description=f"""Paper: {paper['title']}
Use corrected field from Task 1.
Abstract: {abstract}
Excerpt: {excerpt}
Extract in under 200 words: 1.Research Question 2.Methodology 3.Key Results(3) 4.Datasets 5.Limitations(2) 6.Reproducibility""",
        agent=extraction_agent,
        expected_output="6-part structured extraction under 200 words.",
        context=[task1],
    )

    related_str = "\n".join([
        f"- {p['title']} ({p.get('year','?')}) — {p.get('citations',0)} citations"
        for p in related_papers[:3]
    ]) if related_papers else "No related papers found."

    task3 = Task(
        description=f"""Paper: {paper['title']}
Use corrected field from Task 1.
References: {len(paper['references'])} found. Sample: {chr(10).join([str(r)[:80] for r in refs])}
Related via Semantic Scholar: {related_str}
Analyze in under 150 words: citation coverage, 2 missing citations, recency, score 1-10.""",
        agent=citation_agent,
        expected_output="Citation analysis with score. Under 150 words.",
        context=[task1, task2],
    )

    task4 = Task(
        description=f"""Write peer review for: {paper['title']}
Use corrected field from Task 1 (NOT '{classification['field']}').
Keep each section 2-3 sentences max:
## Summary
## Strengths
## Weaknesses
## Citation Assessment
## Reproducibility Score
## Novelty Assessment (agree/disagree with '{classification['novelty']}')
## Final Verdict
**[Accept / Major Revision / Reject]**""",
        agent=critique_agent,
        expected_output="Complete peer review with all 7 sections, each 2-3 sentences.",
        context=[task1, task2, task3],
    )

    return [task1, task2, task3, task4]


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_full_pipeline(pdf_path: str):
    logger.info(f"Starting full pipeline | PDF: {pdf_path}")

    # Guardrail 1: Validate PDF
    try:
        validate_pdf(pdf_path)
    except PipelineError as e:
        print(f"\n❌ Guardrail blocked: {e}")
        log_action("guardrail", "pdf_rejected", {"reason": str(e)})
        return None

    # Step 1: Extract PDF
    print("\n📄 Step 1/5 — Extracting text from PDF...")
    try:
        paper = extract_text_from_pdf(pdf_path)
        validate_extraction(paper)
        paper["full_text"]      = sanitize_text(paper["full_text"])
        paper["abstract"]       = sanitize_text(paper["abstract"], max_chars=500)
        paper["classify_input"] = sanitize_text(paper["classify_input"], max_chars=500)
    except PipelineError as e:
        print(f"\n❌ Guardrail blocked: {e}")
        log_action("guardrail", "extraction_rejected", {"reason": str(e)})
        return None

    log_action("pdf_reader", "extraction_complete", {
        "title":      paper["title"],
        "pages":      paper["num_pages"],
        "chars":      paper["char_count"],
        "refs_found": len(paper["references"]),
    })
    print(f"   Title: {paper['title']}")
    print(f"   Pages: {paper['num_pages']}")
    print(f"   Refs:  {len(paper['references'])} found")

    # Step 2: TextCNN preliminary classification
    print("\n🧠 Step 2/5 — TextCNN preliminary classification...")
    try:
        classification = safe_api_call(classify_paper, paper["classify_input"])
        validate_classification(classification)
    except PipelineError as e:
        print(f"\n❌ Guardrail blocked: {e}")
        log_action("guardrail", "classification_failed", {"reason": str(e)})
        return None

    log_action("textcnn_classifier", "classification_complete", classification)
    print(f"   Preliminary field: {classification['field']} ({classification['field_confidence']*100:.1f}% confidence)")
    print(f"   Novelty:           {classification['novelty']} ({classification['novelty_confidence']*100:.1f}% confidence)")
    print(f"   ℹ️  Classifier Agent will verify and correct this field in Step 5.")

    # Step 3: HITL checkpoint
    print("\n🛑 Step 3/5 — Human checkpoint...")
    approved = human_checkpoint(classification, paper["title"])
    if not approved:
        print("Pipeline stopped by human reviewer.")
        return None

    # Step 4: Citation search
    print("\n🔍 Step 4/5 — Searching related papers via Semantic Scholar...")
    related_papers = search_related_papers(paper["title"], limit=3)
    log_action("citation_agent", "semantic_scholar_search", {
        "query":         paper["title"],
        "results_found": len(related_papers),
    })
    print(f"   Found {len(related_papers)} related papers")

    # Step 5: Multi-agent crew
    print("\n🤖 Step 5/5 — Running 4-agent crew...")
    print("   Agent 1 (Classifier) will determine the TRUE research field.")
    print("   Agents 2-4 will use the corrected field for their analysis.\n")

    agents = make_agents()
    tasks  = make_tasks(agents, paper, classification, related_papers)

    crew = Crew(
        agents=list(agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result     = crew.kickoff()
    result_str = str(result)
    corrected_field = extract_corrected_field(result_str)

    # Save JSON output
    output = {
        "timestamp":            datetime.datetime.now().isoformat(),
        "paper_title":          paper["title"],
        "pdf_path":             pdf_path,
        "textcnn_preliminary":  classification,
        "corrected_field":      corrected_field,
        "related_papers":       related_papers,
        "peer_review":          result_str,
    }

    fname    = f"review_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(OUTPUTS_DIR, fname)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save readable markdown report
    md_path = out_path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(f"# Peer Review: {paper['title']}\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Field:** {corrected_field}\n\n")
        f.write(f"**Novelty:** {classification['novelty']} ({classification['novelty_confidence']*100:.1f}%)\n\n")
        f.write("---\n\n")
        f.write(result_str)

    log_action("main", "pipeline_complete", {
        "output":          out_path,
        "corrected_field": corrected_field,
    })

    print("\n" + "="*65)
    print("✅ PIPELINE COMPLETE")
    print("="*65)
    print(f"   Field:       {corrected_field}")
    print(f"   JSON report: {out_path}")
    print(f"   MD report:   {md_path}")
    print()
    print(result_str)

    return output


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/paper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    run_full_pipeline(pdf_path)