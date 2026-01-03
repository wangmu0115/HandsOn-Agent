"""Main FastAPI application for vector similarity search service."""

import argparse
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from app_config import IndexType, ServiceConfig
from app_logger import VectorSearchLogger, setup_logger
from document_store import DocumentStore
from embedding_service import EmbeddingService
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from indexing import VectorIndex
from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    text: str = Field(..., description="Text content to index")
    doc_id: Optional[str] = Field(None, description="Optional document ID")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    return_documents: bool = Field(default=True, description="Whether to return full documents")


class DeleteRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID to delete")


class IndexResponse(BaseModel):
    success: bool
    message: str
    doc_id: str
    index_size: int


class SearchResponse(BaseModel):
    class SearchResult(BaseModel):
        doc_id: str
        score: float
        text: Optional[str] = None
        metadata: Optional[dict[str, Any]] = None
        rank: int

    success: bool
    query: str
    total_results: int
    results: list[SearchResult]
    search_cost_time: float


class DeleteResponse(BaseModel):
    success: bool
    message: str
    index_size: int


class StatsResponse(BaseModel):
    index_type: str
    index_size: int
    document_count: int
    embedding_dimension: int
    model_name: str


# Global instances
config: ServiceConfig = None
logger: logging.Logger = None
vec_logger: VectorSearchLogger = None
embedding_service: EmbeddingService = None
vector_index: VectorIndex = None
document_store: DocumentStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service, vector_index, document_store

    logger.info("=" * 80)
    logger.info("🚀 Starting Vector Similarity Search Service")
    logger.info("=" * 80)

    # Initialize embedding service
    logger.info("Initializing BGE-M3 embedding service...")
    embedding_service = EmbeddingService(
        model_name=config.model_name,
        use_fp16=config.use_fp16,
        max_seq_length=config.max_seq_length,
        logger=vec_logger,
    )
    # Initialize vector index based on configuration
    embedding_dim = embedding_service.get_embedding_dimension()
    logger.info("Initializing %s vector index...", config.index_type.value.upper())

    # Initialize document store
    logger.info("Initializing document store...")
    document_store = DocumentStore(logger=vec_logger)

    logger.info("=" * 80)
    logger.info("✅ Service initialized successfully!")
    logger.info(f"📍 API available at http://{config.host}:{config.port}")
    logger.info(f"📚 Docs available at http://{config.host}:{config.port}/docs")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("Shutting down service...")


# Create FastAPI app
app = FastAPI(
    title="Vector Similarity Search Service",
    description="Vector similarity search using BGE-M3 embeddings with ANNOY/HNSW indexing",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():
    parser = argparse.ArgumentParser(description="Vector Similarity Search Service")
    parser.add_argument("--index-type", type=str, choices=["annoy", "hnsw"], default="hnsw", help="Type of index to use (default: hnsw)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=4240, help="Port to bind to (default: 4240)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--show-embeddings", action="store_true", help="Show embedding vectors in logs")

    args = parser.parse_args()

    global config, logger, vec_logger

    # Create configuration
    config = ServiceConfig(
        index_type=IndexType(args.index_type),
        host=args.host,
        port=args.port,
        debug=args.debug,
        show_embeddings=args.show_embeddings,
    )
    # Setup logging
    logger = setup_logger("vector_search", config.log_level)
    vec_logger = VectorSearchLogger(logger, config.show_embeddings)

    # Run the service
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
