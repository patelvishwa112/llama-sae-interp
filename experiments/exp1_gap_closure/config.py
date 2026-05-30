"""Shared project paths. Import from any script to get repo-relative paths."""
from pathlib import Path

# Project root is 3 levels up from experiments/exp1_gap_closure/
PROJECT_ROOT = Path(__file__).parent.parent.parent

MODEL_PATH = str(PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct")
SAE_PATH = PROJECT_ROOT / "saes" / "sae-Llama-3.2-1B-131k"
QUESTIONS_PATH = str(PROJECT_ROOT / "data" / "factual_questions.jsonl")
OUTPUT_DIR = Path(__file__).parent / "results"
