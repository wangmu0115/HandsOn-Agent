import logging
import sys
import time
from functools import wraps
from typing import Literal, Optional

import colorlog

loggerLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def setup_logger(name: str = "vector_search", level: loggerLevel = "DEBUG") -> logging.Logger:
    """
    Set up a colorful and informative logger

    Args:
        name: Logger name
        level: Logger level(DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    level = getattr(logging, level)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []  # Clear existing handlers
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    log_format = "%(log_color)s%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s%(reset)s"
    formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time

    Args:
        logger: Logger instance to use
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger("vector_search")
            logger.debug("Starting execution of %s", func.__name__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                logger.info("✅ %s completed successfully in %.4f seconds", func.__name__, time.perf_counter() - start)
                return result
            except Exception as e:
                logger.exception("❌ %s failed after %.4f seconds: ", func.__name__, time.perf_counter() - start, e)
                raise

        return wrapper

    return decorator


class VectorSearchLogger:
    """Logger for vector search operations with detailed debugging."""

    def __init__(self, logger: logging.Logger, show_embeddings: bool = False):
        self.logger = logger
        self.show_embeddings = show_embeddings

    def log_indexing_start(self, doc_id: str, text: str):
        self.logger.debug("=" * 80)
        self.logger.info("📝 Starting INDEXING operation")
        self.logger.debug("Document ID: %s", doc_id)
        self.logger.debug("Text length: {%d} characters", len(text))
        if len(text) > 100:
            self.logger.debug("Text preview: %s...", text[:100])
        else:
            self.logger.debug("Text: %s", text)

    def log_embedding_generation(self, text: str, embedding_shape: tuple, time_taken: float):
        self.logger.info("🧮 Generating embeddings using BGE-M3 model")
        self.logger.debug("Input text length: %d characters", len(text))
        if len(text) > 100:
            self.logger.debug("Text preview: %s...", text[:100])
        else:
            self.logger.debug("Text: %s", text)
        self.logger.debug("Embedding shape: %s", str(embedding_shape))
        self.logger.debug("Embedding generation time: %.4f seconds", time_taken)

    def log_embedding_vector(self, embedding, sample_size: int = 10):
        if self.show_embeddings:
            self.logger.debug("Embedding vector (first %d dimensions): %s", sample_size, embedding[:sample_size])
            self.logger.debug("Embedding statistics - Min: %.6f, Max: %.6f, Mean: %.6f", embedding.min(), embedding.max(), embedding.mean())

    def log_index_update(self, index_type: str, doc_id: str, current_size: int):
        self.logger.info("📊 Updating %s index", index_type.upper())
        self.logger.debug("Adding document %s to index", doc_id)
        self.logger.debug("Current index size: %d documents", current_size)

    def log_search_start(self, query: str, top_k: int):
        self.logger.debug("=" * 80)
        self.logger.info("🔍 Starting SEARCH operation")
        self.logger.debug("Query: %s", query)
        self.logger.debug("Retrieving top %d results", top_k)

    def log_search_results(self, results: list, distances: list, time_taken: float):
        self.logger.info("✨ Search completed in %.4f seconds", time_taken)
        self.logger.debug("Found %d matching documents", len(results))

        for i, (doc_id, distance) in enumerate(zip(results, distances), 1):
            self.logger.debug("  Rank %d: Document %s (distance: %.6f)", i, doc_id, distance)

    def log_deletion(self, doc_id: str):
        """Log document deletion."""
        self.logger.debug("=" * 80)
        self.logger.info("🗑️  Starting DELETE operation")
        self.logger.debug("Deleting document: %s", doc_id)

    def log_error(self, operation: str, ex: Exception):
        """Log errors with context."""
        self.logger.error("❌ Error during %s: %s: %s", operation, ex.__class__.__name__, str(ex))
        self.logger.exception("Full error details:", ex, exc_info=True)

    def log_index_build(self, index_type: str, num_documents: int, parameters: dict):
        self.logger.info("🏗️  Building %s index", index_type.upper())
        self.logger.debug("Number of documents: %d", num_documents)
        self.logger.debug("Index parameters: %s", parameters)
