"""
Centralized path management for the project.
Uses pathlib to ensure cross-platform compatibility and absolute path resolution.
"""
import os
from pathlib import Path

# Get the directory of the current file (config/paths.py)
CONFIG_DIR = Path(__file__).resolve().parent

# Root directory of the project
ROOT_DIR = CONFIG_DIR.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Models directory
MODELS_DIR = ROOT_DIR / "outputs" / "models"
RANKER_MODEL_DIR = MODELS_DIR / "rankers"
CANDIDATE_MODEL_DIR = MODELS_DIR / "candidates"

# Output directories
OUTPUTS_DIR = ROOT_DIR / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure all directories exist
def ensure_directories():
    """Create all necessary output directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        RANKER_MODEL_DIR,
        CANDIDATE_MODEL_DIR,
        SUBMISSIONS_DIR,
        LOGS_DIR,
        REPORTS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Run directory creation on import
ensure_directories()
