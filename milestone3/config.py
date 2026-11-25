"""Shared settings for the Milestone 3 assistant: paths, secrets, model ids and thresholds.

Secrets never live in the repository: the Groq key comes from the environment or the
git-ignored `.env` file at the repository root, and the Neo4j credentials are read from
milestone2/config.txt (the same file the Milestone 2 graph builder uses).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

M3_DIR = Path(__file__).resolve().parent
REPO_DIR = M3_DIR.parent
load_dotenv(REPO_DIR / ".env")

# --- paths -------------------------------------------------------------------
MODELS_DIR = M3_DIR / "models"          # saved Milestone 1 model + SHAP artifacts
EVAL_DIR = M3_DIR / "eval"              # test sets and evaluation scripts
RESULTS_DIR = EVAL_DIR / "results"      # metrics, confusion matrices, sheets
LOG_FILE = M3_DIR / "logs" / "turns.jsonl"
MODEL_TABLE = MODELS_DIR / "model_table.csv"   # M1 leakage-safe feature rows (saved by the export cell)


# --- Neo4j ---------------------------------------------------------------------
def neo4j_settings():
    """KEY=VALUE pairs from milestone2/config.txt (URI, USERNAME, PASSWORD, optional DATABASE)."""
    settings = {}
    with open(REPO_DIR / "milestone2" / "config.txt", encoding="utf-8-sig") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
    return settings


# --- language models (all served by Groq's free tier) -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODELS = [
    "allam-2-7b",                 # SDAIA, small (7B)
    "qwen/qwen3.8-27b",           # Alibaba Qwen, medium (27B)
    "openai/gpt-oss-20b",         # OpenAI open-weight, medium (20B, mixture of experts)
]
API_ERROR = "API connection error"   # the only text shown when a call fails
# Prompt variants deployed after the before-and-after run (eval/results/llm_summary.md): the "strict"
# explanation prompt raised explanation fidelity from about 0.3 to 0.96 for qwen and gpt-oss (0.67 to
# 0.74 for allam); the "strict" answer rules gave mixed results (better for allam, slightly worse for
# gpt-oss), so answers keep "base".
ANSWER_VARIANT = "base"
EXPLAIN_VARIANT = "strict"

# --- intent classification -------------------------------------------------------
# Winner of the 3-model benchmark in eval/eval_intents.py; thresholds picked by leave-one-out
# on the classifier examples (not the test set). Deployed as the fallback after the rules.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MIN_SIMILARITY = 0.45   # below this the embedding classifier asks the user to rephrase
MIN_MARGIN = 0.005      # best minus second-best intent score must exceed this
