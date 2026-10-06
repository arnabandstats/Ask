"""App-wide settings. Everything here is a plain constant; nothing heavy runs on import."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("ASK_DATA_DIR", ROOT_DIR / "ask_data"))
DB_PATH = DATA_DIR / "chats.db"
OUTPUT_DIR = DATA_DIR / "outputs"

# Models (same env vars the original app used).
DEFAULT_MODEL = os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-5.6-luna")
DEEP_MODEL = os.getenv("ASK_DEEP_MODEL", "gpt-5.6-sol")
DEEP_REASONING_EFFORT = os.getenv("ASK_REASONING_EFFORT", "medium")   # low | medium | high

# Validation library (ask/validation).
VALIDATION_SEED = int(os.getenv("ASK_VALIDATION_SEED", "20240601"))   # every random step uses this
JUDGE_MODEL = os.getenv("ASK_JUDGE_MODEL", "")        # LLM-judge tests; blank = the default model
MAX_TABLE_ROWS = int(os.getenv("ASK_MAX_TABLE_ROWS", "2000000"))      # above this, tables are hash-sampled
MAX_SUITE_TESTS = 60          # tests one run_validation_suite call may run

# Agent limits.
MAX_AGENT_STEPS = 20          # tool-calling rounds per answer
MAX_REPAIR_STEPS = 5          # extra rounds for the citation-repair pass
HISTORY_TURNS = 12            # past user/assistant turns sent with each question
MAX_TOOL_OUTPUT_CHARS = 24_000

# Images: PDF pages with figures/scans and image files are read by the vision model.
READ_IMAGES = os.getenv("ASK_READ_IMAGES", "1").lower() not in {"0", "false", "no"}
VISION_MODEL = os.getenv("ASK_VISION_MODEL", "")      # blank = the default model
MAX_VISION_PAGES = 40         # PDF pages read by the vision model per document
MAX_VISION_IMAGES = 60        # image files read per folder
MIN_IMAGE_SIDE = 120          # smaller embedded images (logos, icons) don't trigger a page read

# Loading limits.
MAX_FILE_BYTES = 5_000_000    # skip text files larger than this
MAX_REPO_FILES = 6_000
MAX_READ_LINES = 400          # lines returned by one read_file call
