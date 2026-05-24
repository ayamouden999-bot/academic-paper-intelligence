## Run in Google Colab
[![Open In Colab]
https://colab.research.google.com/drive/1HpKdw7IiekZv9GolXtd-YfWIkQ2dzNTD?usp=sharing
# 🎓 Academic Paper Intelligence System
> A multi-agent AI system that reads research papers and produces peer-review style critiques.

**Course:** AI & Big Data — S8 Integrated Project | UIR 2025–2026  
**Team:** [Names here]  
**Professor:** Prof. Hakim Hafidi

---

## What it does

Drop any research PDF → the system classifies its field and novelty level using a custom-trained TextCNN → extracts methodology and findings → validates citations via Semantic Scholar → generates a structured peer-review critique with a final verdict.

---

## Setup (5 minutes)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/academic-paper-intelligence.git
cd academic-paper-intelligence
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key
Get a free key at https://aistudio.google.com
```bash
cp .env.example .env
# Edit .env and paste your key
```

### 5. Run the Week 1 prototype
```bash
python main.py path/to/your_paper.pdf
```

---

## Project Structure

```
academic-paper-intelligence/
├── agents/             ← CrewAI agent definitions
├── models/             ← TextCNN architecture + training script
├── tools/              ← PDF reader, classifier tool, Semantic Scholar
├── data/               ← arXiv dataset (not tracked in git)
├── logs/               ← JSON action logs (auto-generated)
├── outputs/            ← Reports and evaluation plots
├── config.py           ← All constants and settings
├── main.py             ← Entry point
└── ARCHITECTURE.md     ← System design document
```

---

## Training the TextCNN (Week 2)

Open `models/train.py` in Google Colab with GPU runtime:
```bash
# In Colab:
!python models/train.py
# Trains in ~20-30 minutes on T4 GPU
# Saves model to models/textcnn_trained.pth
```

---

## Week-by-week progress

- [x] **W1** — Environment, dataset, single-agent prototype, architecture doc
- [ ] **W2** — TextCNN training, evaluation, wrapped as CrewAI tool
- [ ] **W3** — Full multi-agent pipeline, HITL, end-to-end demo
- [ ] **W4** — Guardrails, logging, final report, demo video

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | CrewAI |
| LLM backend | Gemini 1.5 Flash (free) |
| DL model | TextCNN — PyTorch ≥ 2.0 |
| Dataset | arXiv classification (HuggingFace) |
| Citation tool | Semantic Scholar API (free) |
| Language | Python 3.10+ |
