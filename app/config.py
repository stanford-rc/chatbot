import os
import yaml
from typing import Dict, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

class Settings(BaseSettings):
    """Application settings loaded from config.yaml and environment variables"""
    
    # App settings
    APP_TITLE: str = config['app']['title']
    APP_DESCRIPTION: str = config['app']['description']
    APP_VERSION: str = config['app']['version']
    
    # Model settings
    MODEL_PATH: str = Field(default=config['model']['path'], env="MODEL_PATH")
    MODEL_TYPE: str = config['model']['type']
    MODEL_DEVICE: str = os.environ.get('WORKER_GPU', config['model']['device'])
    USE_QUANTIZATION: bool = config['model']['use_quantization']
    LOCAL_FILES_ONLY: bool = config['model']['local_files_only']
    
    # Generation settings
    MAX_NEW_TOKENS: int = config['generation']['max_new_tokens']
    
    # Cluster paths
    CLUSTERS: Dict[str, str] = config['clusters']

    # Shared docs path — merged into every cluster's retriever at startup.
    # Set to '' to disable.
    SHARED_DOCS_PATH: str = config.get('shared_docs', '')
    
    # API settings
    CORS_ORIGINS: List[str] = config['api']['cors_origins']
    
    # Caching settings
    SEMANTIC_CACHE_ENABLED: bool = config.get('caching', {}).get('SEMANTIC_CACHE_ENABLED', True)
    SEMANTIC_CACHE_THRESHOLD: float = config.get('caching', {}).get('SEMANTIC_CACHE_THRESHOLD', 0.70)
    SEMANTIC_CACHE_DB: str = config.get('caching', {}).get('SEMANTIC_CACHE_DB', '.response_cache.db')
    LANGCHAIN_CACHE_DB: str = config.get('caching', {}).get('LANGCHAIN_CACHE_DB', '.langchain.db')
    
    # Retrieval settings
    MAX_RETRIEVED_DOCS: int = config.get('retrieval', {}).get('MAX_RETRIEVED_DOCS', 5)
    MIN_BM25_SCORE: float = config.get('retrieval', {}).get('MIN_BM25_SCORE', 2.0)
    HYBRID_ENABLED: bool = config.get('retrieval', {}).get('HYBRID_ENABLED', False)
    VECTOR_MODEL: str = config.get('retrieval', {}).get('VECTOR_MODEL', 'all-MiniLM-L6-v2')
    RRF_K: int = config.get('retrieval', {}).get('RRF_K', 60)

    # Grounding check settings
    GROUNDING_CHECK_ENABLED: bool = config.get('grounding', {}).get('GROUNDING_CHECK_ENABLED', True)
    REFUSAL_DISCLAIMER: str = config.get('grounding', {}).get('REFUSAL_DISCLAIMER',
        "Note: This answer may not reflect your cluster's specific configuration. "
        "Please verify with the documentation or contact srcc-support@stanford.edu.")

    # Server settings
    API_PORT: int = Field(default=config.get('server', {}).get('api_port', 8000), env="API_PORT")
    API_HOST: str = Field(default=config.get('server', {}).get('host', 'localhost'), env="API_HOST")

    # Logging settings
    LOG_DIR: str = Field(default=config.get('logging', {}).get('log_dir', 'logs'), env="LOG_DIR")

    # Worker configuration (multi-GPU mode)
    WORKERS: list = config.get('workers', [
        {"url": "http://localhost:8001", "port": 8001, "gpu": "cuda:0"},
        {"url": "http://localhost:8002", "port": 8002, "gpu": "cuda:1"},
    ])

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
