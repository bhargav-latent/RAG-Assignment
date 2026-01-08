"""Configuration for preprocessing pipeline."""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
RAW_PDFS_DIR = ROOT_DIR / "raw_pdfs"

# Output aligned with agent's expected structure
PAPERS_DIR = ROOT_DIR / "papers"  # Agent reads from here

# API Settings
NVAI_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-parse"
MAX_PARALLEL = 5
TIMEOUT = 180  # seconds

# Processing
DPI = 300
IMAGE_FORMAT = "png"
