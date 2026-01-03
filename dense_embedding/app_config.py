"""Configuration for the dense embedding service."""

from dataclasses import dataclass
from enum import StrEnum


class IndexType(StrEnum):
    ANNOY = "annoy"
    HNSW = "hnsw"


@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 4240

    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True
    max_seq_length: int = 8192

    index_type: IndexType = IndexType.HNSW
    max_documents: int = 100000

    # Annoy specific settings
    annoy_n_trees: int = 50
    annoy_metric: str = "angular"
    # HNSW specific settings
    hnsw_ef_construction: int = 200
    hnsw_M: int = 16
    hnsw_ef_search: int = 50
    hnsw_space: str = "cosine"

    # Logging settings
    log_level: str = "INFO"
    debug: bool = False
    show_embeddings: bool = False
