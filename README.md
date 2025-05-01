# Microsoft GenAI Pipeline

This repository implements a unified, multi-stage workflow for biomedical knowledge extraction and hypothesis generation, using LangGraph and Google Gemini LLMs.

## Overview

The pipeline consists of:
1. **Ontology analysis** (`OntologistAgent`): processes subgraph JSONs and produces domain analyses.
2. **PubMed expansion** (`PubMedExpanderAgent`): generates and executes PubMed queries to fetch relevant articles.
3. **Question generation** (`QuestionGenAgent`): creates questions from abstracts using a T5-based model.
4. **Embedding fine-tuning** (`FinetuneEmbed`): fine-tunes a SentenceTransformer on question-abstract pairs.
5. **Embedding update** (`UpdateEmbedding`): updates embeddings in ChromaDB.
6. **Hypothesis generation** (`HypotesisGenerator`): proposes testable hypotheses via LLM.
7. **Hypothesis evaluation** (`HypotesisEvaluator`): evaluates hypotheses against retrieved evidence.
8. **Hypothesis refinement** (`HypotesisRefiner`): refines hypotheses iteratively.

All steps are orchestrated by `Part3_main.py` and share a single `StateDict` and centralized configuration (`config.py`).

## Prerequisites

- Python 3.8 or newer (tested on 3.12)
- Linux environment
- A `.env` file at the project root with the following variables:
  - `GEMINI_API_KEY` (Google Gemini key)
  - `ENTREZ_EMAIL` (Entrez user email)
  - `PubMed_API_KEY` and `PubMed_email`

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Microsoft_genAI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
config.py            # Central paths, ChromaDB client, embedding function
Part3_main.py        # Unified pipeline entrypoint
agents/              # LLM-based agents for each workflow step
Tools/               # Utilities (graph runner, search, evaluation, etc.)
Inputs/              # Input subgraphs and CSV data
Outputs/             # Generated outputs (models, embeddings, hypotheses)
``` 

## Usage

1. Populate `.env` with required API keys.
2. Run the main script:
   ```bash
   python Part3_main.py
   ```
3. Hypotheses will be saved as JSON files `hypotheses_<key>.json` in the working directory.

## License

MIT License (or your preferred license)

---
**Created for:** Microsoft AI Agents Hackathon 2025  
**Assistance by:** GitHub Copilot
