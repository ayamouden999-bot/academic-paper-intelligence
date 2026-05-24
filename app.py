"""
app.py — Flask web interface for Academic Paper Intelligence System
Wraps the existing pipeline with a professional web UI.
"""

import os, json, threading, uuid, time
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

jobs = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['pdf']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file — must be a PDF'}), 400

    job_id = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{job_id}_{file.filename}")
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(pdf_path)

    jobs[job_id] = {
        'status': 'running',
        'step': 0,
        'steps': [
            {'id': 1, 'label': 'Extracting text from PDF',        'status': 'pending'},
            {'id': 2, 'label': 'TextCNN classification',           'status': 'pending'},
            {'id': 3, 'label': 'Human checkpoint',                 'status': 'pending'},
            {'id': 4, 'label': 'Semantic Scholar citation search', 'status': 'pending'},
            {'id': 5, 'label': '4-agent crew running',             'status': 'pending'},
        ],
        'classification': None,
        'result': None,
        'error': None,
        'pdf_path': pdf_path,
        'output_path': None,
        'md_path': None,
    }

    thread = threading.Thread(target=run_pipeline_async, args=(job_id, pdf_path))
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id})


def update_step(job_id, step_id, status):
    for s in jobs[job_id]['steps']:
        if s['id'] == step_id:
            s['status'] = status
    jobs[job_id]['step'] = step_id


def run_pipeline_async(job_id, pdf_path):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from tools.pdf_reader import extract_text_from_pdf
        from tools.classifier_tool import classify_paper
        from tools.guardrails import (
            validate_pdf, validate_extraction, validate_classification,
            sanitize_text, safe_api_call, PipelineError
        )

        # Step 1: Extract PDF
        update_step(job_id, 1, 'running')
        validate_pdf(pdf_path)
        paper = extract_text_from_pdf(pdf_path)
        validate_extraction(paper)
        paper['full_text']      = sanitize_text(paper['full_text'])
        paper['abstract']       = sanitize_text(paper['abstract'], max_chars=2000)
        paper['classify_input'] = sanitize_text(paper['classify_input'], max_chars=1500)
        jobs[job_id]['paper'] = {
            'title':    paper['title'],
            'pages':    paper['num_pages'],
            'refs':     len(paper['references']),
            'abstract': paper['abstract'][:300] + '...',
        }
        update_step(job_id, 1, 'done')

        # Step 2: TextCNN classification
        update_step(job_id, 2, 'running')
        classification = safe_api_call(classify_paper, paper['classify_input'])
        validate_classification(classification)
        jobs[job_id]['classification'] = classification
        update_step(job_id, 2, 'done')

        # Step 3: Pause for human approval
        update_step(job_id, 3, 'waiting')
        jobs[job_id]['status'] = 'awaiting_approval'
        jobs[job_id]['_paper_obj'] = paper

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)


def resume_pipeline(job_id):
    paper = jobs[job_id]['_paper_obj']
    classification = jobs[job_id]['classification']

    try:
        from tools.semantic_scholar import search_related_papers
        from main import make_agents, make_tasks, extract_corrected_field
        from crewai import Crew, Process
        import datetime

        # Step 4: Citation search
        update_step(job_id, 3, 'done')
        update_step(job_id, 4, 'running')
        related_papers = search_related_papers(paper['title'], limit=5)
        update_step(job_id, 4, 'done')

        # Step 5: Multi-agent crew with retry on rate limit
        update_step(job_id, 5, 'running')
        jobs[job_id]['status'] = 'running'

        agents = make_agents()
        # Truncate to reduce token usage
        # Aggressively truncate to stay under rate limits
        paper['abstract']  = paper['abstract'][:500]
        paper['full_text'] = paper['full_text'][:1000]
        tasks  = make_tasks(agents, paper, classification, related_papers)

        crew = Crew(
            agents=list(agents),
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )

        result = None
        for attempt in range(3):
            try:
                result = crew.kickoff()
                break
            except Exception as e:
                err = str(e).lower()
                if 'rate_limit' in err or 'ratelimit' in err or '429' in err:
                    wait = 30 * (attempt + 1)
                    print(f"Rate limit hit — waiting {wait}s (attempt {attempt+1}/3)")
                    jobs[job_id]['steps'][4]['label'] = f'4-agent crew — rate limit, retrying in {wait}s...'
                    time.sleep(wait)
                    jobs[job_id]['steps'][4]['label'] = '4-agent crew running'
                    if attempt == 2:
                        raise
                else:
                    raise

        result_str = str(result)
        corrected_field = extract_corrected_field(result_str)

        # Save outputs
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_json = f"outputs/review_{ts}.json"
        out_md   = f"outputs/review_{ts}.md"

        output = {
            'timestamp':       datetime.datetime.now().isoformat(),
            'paper_title':     paper['title'],
            'classification':  classification,
            'corrected_field': corrected_field,
            'related_papers':  related_papers,
            'peer_review':     result_str,
        }

        with open(out_json, 'w') as f:
            json.dump(output, f, indent=2)

        with open(out_md, 'w') as f:
            f.write(f"# Peer Review: {paper['title']}\n\n")
            f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**Field:** {corrected_field}\n\n")
            f.write(f"**Novelty:** {classification['novelty']} ({classification['novelty_confidence']*100:.1f}%)\n\n")
            f.write("---\n\n")
            f.write(result_str)

        parsed = parse_review(result_str)
        parsed['field']              = corrected_field
        parsed['novelty']            = classification['novelty']
        parsed['novelty_confidence'] = round(classification['novelty_confidence'] * 100, 1)
        parsed['title']              = paper['title']

        jobs[job_id]['result']      = parsed
        jobs[job_id]['output_path'] = out_json
        jobs[job_id]['md_path']     = out_md
        jobs[job_id]['status']      = 'done'
        update_step(job_id, 5, 'done')

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)


def parse_review(text):
    import re

    def extract_section(heading, txt):
        pattern = rf'##\s+{heading}\s*\n(.*?)(?=\n##\s|\Z)'
        match = re.search(pattern, txt, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ''

    verdict = 'Unknown'
    m = re.search(r'\*\*(Accept|Major Revision|Reject)\*\*', text, re.IGNORECASE)
    if m:
        verdict = m.group(1)

    citation_score = 'N/A'
    cs = re.search(r'(\d+)\s*(?:out of|/)\s*10', extract_section('Citation Assessment', text))
    if cs:
        citation_score = f"{cs.group(1)}/10"

    repro_score = 'N/A'
    rs = re.search(r'(\d+)\s*(?:out of|/)\s*10', extract_section('Reproducibility Score', text))
    if rs:
        repro_score = f"{rs.group(1)}/10"

    return {
        'summary':            extract_section('Summary', text),
        'strengths':          extract_section('Strengths', text),
        'weaknesses':         extract_section('Weaknesses', text),
        'citations':          extract_section('Citation Assessment', text),
        'repro':              extract_section('Reproducibility Score', text),
        'novelty_assessment': extract_section('Novelty Assessment', text),
        'verdict_section':    extract_section('Final Verdict', text),
        'verdict':            verdict,
        'citation_score':     citation_score,
        'repro_score':        repro_score,
        'full_text':          text,
    }


@app.route('/approve/<job_id>', methods=['POST'])
def approve(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    if jobs[job_id]['status'] != 'awaiting_approval':
        return jsonify({'error': 'Not awaiting approval'}), 400
    thread = threading.Thread(target=resume_pipeline, args=(job_id,))
    thread.daemon = True
    thread.start()
    return jsonify({'ok': True})


@app.route('/reject/<job_id>', methods=['POST'])
def reject(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    jobs[job_id]['status'] = 'rejected'
    update_step(job_id, 3, 'rejected')
    return jsonify({'ok': True})


@app.route('/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    return jsonify({
        'status':         job['status'],
        'steps':          job['steps'],
        'classification': job.get('classification'),
        'paper':          job.get('paper'),
        'result':         job.get('result'),
        'error':          job.get('error'),
    })


@app.route('/download/<job_id>/<filetype>')
def download(job_id, filetype):
    if job_id not in jobs:
        return jsonify({'error': 'Not found'}), 404
    path = jobs[job_id].get('output_path') if filetype == 'json' else jobs[job_id].get('md_path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not ready'}), 404
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='127.0.0.1')
