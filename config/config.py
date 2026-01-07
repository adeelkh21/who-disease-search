"""
Configuration file for WHO Vector Database Project

Centralized configuration for paths, model settings, and parameters.
All scripts import from here to avoid hardcoding.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Input data files
CATALOGUE_PATH = DATA_DIR / "who_diseases_Full_Catalogue.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "who_embeddings.jsonl"
SCRAPING_REPORT_PATH = DATA_DIR / "scraping_report.txt"

# Output index files
FAISS_INDEX_PATH = DATA_DIR / "who_index.faiss"
METADATA_PATH = DATA_DIR / "index_metadata.json"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Embedding model
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
EMBEDDING_DIM = 768
MAX_SEQUENCE_LENGTH = 512

# Device configuration (auto-detect)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# INDEXING CONFIGURATION
# =============================================================================

# FAISS index type options:
# - "IndexFlatL2": Exact L2 distance search (for normalized vectors → cosine similarity)
# - "IndexFlatIP": Exact inner product search
# - "IndexHNSW": Approximate nearest neighbor (faster, larger datasets)
INDEX_TYPE = "IndexFlatL2"

# Whether to normalize embeddings (required for cosine similarity with L2 index)
NORMALIZE_EMBEDDINGS = True

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = 16

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

# Default number of results
DEFAULT_TOP_K = 5

# Reranking configuration
RERANK_MULTIPLIER = 5  # Fetch top_k * multiplier candidates for reranking

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log format
LOG_FORMAT = "%(asctime)s | %(levelname)8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """
    Validate configuration settings.
    
    Raises:
        ValueError: If configuration is invalid
    """
    if EMBEDDING_DIM <= 0:
        raise ValueError(f"Invalid EMBEDDING_DIM: {EMBEDDING_DIM}")
    
    if INDEX_TYPE not in ["IndexFlatL2", "IndexFlatIP", "IndexHNSW"]:
        raise ValueError(f"Invalid INDEX_TYPE: {INDEX_TYPE}")
    
    if EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError(f"Invalid EMBEDDING_BATCH_SIZE: {EMBEDDING_BATCH_SIZE}")
    
    if DEFAULT_TOP_K <= 0:
        raise ValueError(f"Invalid DEFAULT_TOP_K: {DEFAULT_TOP_K}")


# Validate on import
validate_config()

