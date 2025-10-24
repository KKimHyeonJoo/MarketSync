# config.py

import logging
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
PATH_DATA_DIR = PROJECT_ROOT / 'data'
PATH_VECTORSTORE_DIR = PROJECT_ROOT / 'vectorstore'
ASSETS =  PROJECT_ROOT / "assets"

# Data Files
PATH_FINAL_DF = PATH_DATA_DIR / 'final_df.csv'
PATH_FESTIVAL_DF = PATH_DATA_DIR / 'festival_df.csv'

# Vectorstore Paths
PATH_FAISS_MARKETING = PATH_VECTORSTORE_DIR / 'faiss_marketing'
PATH_FAISS_FESTIVAL = PATH_VECTORSTORE_DIR / 'faiss_festival'


# --- API ---
API_SERVER_URL = "http://127.0.0.1:8000"
API_PROFILE_ENDPOINT = f"{API_SERVER_URL}/profile"
API_MERCHANTS_ENDPOINT = f"{API_SERVER_URL}/merchants"


# --- Models ---
LLM_MODEL_NAME = "gemini-2.5-flash" 
EMBEDDING_MODEL = "dragonkue/BGE-m3-ko"


# --- RAG Weights ---
FESTIVAL_EMBEDDING_WEIGHT = 0.4
FESTIVAL_DYNAMIC_WEIGHT = 0.6


# --- Logging ---
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "%(asctime)s - [%(levelname)s] - %(name)s (%(funcName)s): %(message)s"

def get_logger(name: str):
    """
    표준화된 포맷으로 로거를 반환합니다.
    """
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    return logger
