import os
import yaml
from typing import Dict, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

def _find_config() -> str:
    """Locate config.yaml.

    Resolution order:
      1. ADA_CONFIG env var (explicit path, e.g. /etc/ada-chatbot/config.yaml)
      2. config.yaml alongside the app package (development default)
    """
    explicit = os.environ.get('ADA_CONFIG')
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"ADA_CONFIG points to missing file: {explicit}")
        return explicit
    fallback = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    if not os.path.isfile(fallback):
        raise FileNotFoundError(
            f"config.yaml not found at {fallback}. "
            "Set ADA_CONFIG=/path/to/config.yaml to specify its location."
        )
    return fallback


def load_config():
    """Load configuration from config.yaml (location resolved by _find_config)."""
    config_path = _find_config()
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    # Stash the resolved config path so other modules can resolve relative paths
    data['_config_path'] = config_path
    return data

config = load_config()

# Base directory for resolving relative paths in config.yaml.
# Relative doc/log paths are resolved from here so the app works
# regardless of the CWD it's invoked from.
_CONFIG_DIR = os.path.dirname(os.path.abspath(config['_config_path']))

# Runtime data directory — writable location for cache, logs, HF cache.
# Empty means use the config file's directory (development default).
_DATA_DIR = config.get('data_dir', '') or _CONFIG_DIR


def _resolve(path: str) -> str:
    """Resolve a config path to an absolute path.

    Resolution order:
      - Already absolute: returned as-is.
      - Starts with /workspace/: remapped to DATA_DIR (so production configs
        written with /workspace/ prefixes work when data_dir is set).
      - Relative: resolved from the config file's directory.
    """
    if not path:
        return path
    if path.startswith('/workspace/'):
        return os.path.join(_DATA_DIR, path[len('/workspace/'):])
    if path == '/workspace':
        return _DATA_DIR
    return path if os.path.isabs(path) else os.path.join(_CONFIG_DIR, path)


class Settings(BaseSettings):
    """Application settings loaded from config.yaml and environment variables"""
    
    # App settings
    APP_TITLE: str = config['app']['title']
    APP_DESCRIPTION: str = config['app']['description']
    APP_VERSION: str = config['app']['version']
    
    # Model settings
    MODEL_PATH: str = Field(default=config['model']['path'], env="MODEL_PATH")
    MODEL_TYPE: str = config['model']['type']
    MODEL_DTYPE: str = config['model'].get('dtype', 'auto')
    MODEL_DEVICE: str = os.environ.get('WORKER_GPU', config['model']['device'])
    USE_QUANTIZATION: bool = config['model']['use_quantization']
    LOCAL_FILES_ONLY: bool = config['model']['local_files_only']
    
    # Generation settings
    MAX_NEW_TOKENS: int = config['generation']['max_new_tokens']
    
    # Cluster paths — relative paths resolved from the config file's directory
    CLUSTERS: Dict[str, str] = {k: _resolve(v) for k, v in config['clusters'].items()}

    # Shared docs path — merged into every cluster's retriever at startup.
    # Set to '' to disable.
    SHARED_DOCS_PATH: str = _resolve(config.get('shared_docs', ''))
    
    # API settings
    CORS_ORIGINS: List[str] = config['api']['cors_origins']
    
    # Caching settings
    SEMANTIC_CACHE_ENABLED: bool = config.get('caching', {}).get('SEMANTIC_CACHE_ENABLED', True)
    SEMANTIC_CACHE_THRESHOLD: float = config.get('caching', {}).get('SEMANTIC_CACHE_THRESHOLD', 0.70)
    SEMANTIC_CACHE_DB: str = _resolve(config.get('caching', {}).get('SEMANTIC_CACHE_DB', '.response_cache.db'))
    SEMANTIC_CACHE_CLEAR_ON_STARTUP: bool = config.get('caching', {}).get('SEMANTIC_CACHE_CLEAR_ON_STARTUP', False)
    LANGCHAIN_CACHE_DB: str = _resolve(config.get('caching', {}).get('LANGCHAIN_CACHE_DB', '.langchain.db'))
    
    # Retrieval settings
    MAX_RETRIEVED_DOCS: int = config.get('retrieval', {}).get('MAX_RETRIEVED_DOCS', 5)
    MIN_BM25_SCORE: float = config.get('retrieval', {}).get('MIN_BM25_SCORE', 2.0)
    HYBRID_ENABLED: bool = config.get('retrieval', {}).get('HYBRID_ENABLED', False)
    VECTOR_MODEL: str = config.get('retrieval', {}).get('VECTOR_MODEL', 'all-MiniLM-L6-v2')
    CHUNK_SIZE: int = config.get('retrieval', {}).get('CHUNK_SIZE', 2000)
    CHUNK_OVERLAP: int = config.get('retrieval', {}).get('CHUNK_OVERLAP', 200)
    RRF_K: int = config.get('retrieval', {}).get('RRF_K', 60)
    FAISS_RRF_WEIGHT: float = config.get('retrieval', {}).get('FAISS_RRF_WEIGHT', 1.5)

    # Grounding check settings
    GROUNDING_CHECK_ENABLED: bool = config.get('grounding', {}).get('GROUNDING_CHECK_ENABLED', True)
    REFUSAL_DISCLAIMER: str = config.get('grounding', {}).get('REFUSAL_DISCLAIMER',
        "Note: This answer may not reflect your cluster's specific configuration. "
        "Please verify with the documentation or contact srcc-support@stanford.edu.")

    # Server settings
    API_PORT: int = Field(default=config.get('server', {}).get('api_port', 8000), env="API_PORT")
    API_HOST: str = Field(default=config.get('server', {}).get('host', 'localhost'), env="API_HOST")

    # Logging settings
    LOG_DIR: str = Field(default=_resolve(config.get('logging', {}).get('log_dir', 'logs')), env="LOG_DIR")
    STATS_LOG: str = _resolve(config.get('logging', {}).get('stats_log', ''))

    # HuggingFace cache directory — must be writable inside the container.
    HF_HOME: str = _resolve(config.get('hf_home', '/workspace/.hf_cache'))

    # Worker configuration (multi-GPU mode)
    WORKERS: list = config.get('workers', [
        {"url": "http://localhost:8001", "port": 8001, "gpu": "cuda:0"},
        {"url": "http://localhost:8002", "port": 8002, "gpu": "cuda:1"},
    ])

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
