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
            "You receive a preliminary TextCNN prediction but must use your own "
            "reasoning on the abstract to confirm or correct it. "
            "Always output the correct field regardless of what the TextCNN said."
        ),
        backstory=(
            "You are a specialist in academic paper classification with deep knowledge "
            "across all scientific disciplines. You receive a preliminary classification "
            "from a TextCNN model but you are the authoritative classifier — you use "
            "the paper abstract to determine the true field and novelty, correcting "
            "the model when necessary. You never output 'Uncertain' — you always "
            "determine the correct field from the content."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    extraction_agent = Agent(
        role="Research Paper Extraction Specialist",
        goal=(
            "Extract all key structured information from the research paper: "
            "research question, methodology, datasets used, key results, and limitations. "
            "Use the CORRECTED field from the Classifier Agent, not the preliminary TextCNN label."
        ),
        backstory=(
            "You are an expert research analyst who has reviewed thousands of papers. "
            "You know exactly what to look for in each discipline — a CS paper needs "
            "dataset details and benchmark comparisons, a math paper needs theorem "
            "statements and proof techniques. You never hallucinate facts."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    citation_agent = Agent(
        role="Citation and Literature Analyst",
        goal=(
            "Analyze the paper's references and identify important missing citations. "
            "Use the CORRECTED field from the Classifier Agent for context."
        ),
        backstory=(
            "You are a research librarian and literature expert. You know the seminal "
            "papers in every field and immediately spot missing crucial citations. "
            "You use real search results and never make up paper titles or authors."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    critique_agent = Agent(
        role="Peer Review Critique Writer",
        goal=(
            "Synthesize all analysis into a structured, fair peer-review critique. "
            "Use the CORRECTED field from the Classifier Agent in your review. "
            "Produce a final verdict: Accept / Major Revision / Reject."
        ),
        backstory=(
            "You are a senior academic reviewer who has published in top venues and "
            "reviewed for Nature, NeurIPS, ICML, and JAMA. Your reviews are honest, "
            "specific, and constructive. You always back every claim with evidence. "
            "Your verdict is always one of: Accept / Major Revision / Reject."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    return classifier_agent, extraction_agent, citation_agent, critique_agent


# ── Task definitions ──────────────────────────────────────────────────────────
def make_tasks(agents, paper, classification, related_papers):
    classifier_agent, extraction_agent, citation_agent, critique_agent = agents

    task1 = Task(
        description=f"""You are the authoritative classifier for this research paper.

        Paper title: {paper['title']}

        Abstract:
        {paper['abstract']}

        A TextCNN model gave this PRELIMINARY prediction (may be wrong):
        - Preliminary Field: {classification['field']} (confidence: {classification['field_confidence']*100:.1f}%)
        - Novelty Level: {classification['novelty']} (confidence: {classification['novelty_confidence']*100:.1f}%)

        The TextCNN confidence is low ({classification['field_confidence']*100:.1f}%), meaning the model
        is uncertain. YOU must determine the correct field from the abstract.

        Your job:
        1. Read the abstract carefully
        2. Determine the TRUE research field (e.g. Mathematical Physics, Computer Science,
           Internet of Things, Pure Mathematics, Theoretical Physics, Machine Learning,
           Biology, Economics, Electrical Engineering, etc.)
        3. Start your response with: "Corrected Field: [field name]"
        4. Confirm or correct the novelty level
        5. Provide a 2-sentence rationale
        6. List 3 key technical terms from the paper
        """,
        agent=classifier_agent,
        expected_output="Corrected Field clearly stated first, then novelty level, rationale, and 3 key technical terms.",
    )

    task2 = Task(
        description=f"""Extract structured information from this research paper.

        Paper title: {paper['title']}

        IMPORTANT: Use the corrected field from Task 1 (the Classifier Agent),
        NOT the preliminary TextCNN label '{classification['field']}'.

        Full abstract:
        {paper['abstract']}

        Paper excerpt (first 3000 chars):
        {paper['full_text'][:3000]}

        Extract and structure the following:
        1. **Research Question** — What problem does this paper solve? (2-3 sentences)
        2. **Methodology** — What approach/technique is used? Be specific.
        3. **Key Results** — 3 concrete findings with numbers if available
        4. **Datasets/Benchmarks** — What data was used? (if applicable)
        5. **Limitations** — 2-3 honest limitations of the work
        6. **Reproducibility** — Is there mention of code, data availability?
        """,
        agent=extraction_agent,
        expected_output="Structured extraction with research question, methodology, results, datasets, limitations, reproducibility.",
        context=[task1],
    )

    related_str = "\n".join([
        f"- {p['title']} ({p['year']}) — {p['citations']} citations"
        for p in related_papers[:5]
    ]) if related_papers else "No related papers found via Semantic Scholar."

    task3 = Task(
        description=f"""Analyze the citation landscape for this research paper.

        Paper title: {paper['title']}

        IMPORTANT: Use the corrected field from Task 1, not '{classification['field']}'.

        References found in paper ({len(paper['references'])} total):
        {chr(10).join(paper['references'][:10])}

        Related papers found via Semantic Scholar API search:
        {related_str}

        Your analysis:
        1. **Citation Coverage** — Does the paper cite the key works in this field?
        2. **Missing Citations** — Name 2-3 important papers that should be cited
        3. **Recency** — Are the references up to date or mostly outdated?
        4. **Overall citation quality** — Score 1-10 with justification
        """,
        agent=citation_agent,
        expected_output="Citation analysis with coverage, missing refs, recency, and quality score.",
        context=[task1, task2],
    )

    task4 = Task(
        description=f"""Write a complete peer-review critique for this research paper.

        You have received:
        - Classification analysis from Task 1 (use the CORRECTED field, not '{classification['field']}')
        - Structured extraction from Task 2
        - Citation analysis from Task 3

        Paper title: {paper['title']}

        Write a structured peer review with these exact sections:

        ## Summary
        (2-3 sentences summarizing the paper's contribution and its TRUE research field)

        ## Strengths
        (3 specific, evidence-backed strengths)

        ## Weaknesses
        (3 specific, evidence-backed weaknesses)

        ## Citation Assessment
        (Based on Task 3 findings)

        ## Reproducibility Score
        (1-10 with justification)

        ## Novelty Assessment
        (Agree or disagree with '{classification['novelty']}' classification — explain why)

        ## Final Verdict
        **[Accept / Major Revision / Reject]**
        (2-3 sentences justifying the verdict)
        """,
        agent=critique_agent,
        expected_output="Complete peer review using the corrected field, with all sections and a final verdict.",
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
        paper["abstract"]       = sanitize_text(paper["abstract"], max_chars=2000)
        paper["classify_input"] = sanitize_text(paper["classify_input"], max_chars=1500)
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
    related_papers = search_related_papers(paper["title"], limit=5)
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

    result = crew.kickoff()
    result_str = str(result)

    # Extract corrected field from Agent 1 output
    corrected_field = extract_corrected_field(result_str)

    # Save JSON output
    output = {
        "timestamp":           datetime.datetime.now().isoformat(),
        "paper_title":         paper["title"],
        "pdf_path":            pdf_path,
        "textcnn_preliminary": classification,
        "corrected_field":     corrected_field,
        "related_papers":      related_papers,
        "peer_review":         result_str,
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
        f.write(f"**Novelty:** {classification['novelty']} ({classification['novelty_confidence']*100:.1f}% — assessed by Classifier Agent in review)\n\n")
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