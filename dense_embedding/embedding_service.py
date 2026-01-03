"""Embedding service using BGE-M3 model."""

import time
from typing import Literal, Optional

import numpy as np
from app_logger import VectorSearchLogger
from FlagEmbedding import BGEM3FlagModel


class EmbeddingService:
    """Service for generating embeddings using BGE-M3 model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        max_seq_length: int = 512,
        logger: Optional[VectorSearchLogger] = None,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_seq_length = max_seq_length
        self.logger = logger
        # Initialize the model
        self._initialize_model()

    def _initialize_model(self):
        start = time.perf_counter()
        if self.logger:
            self.logger.logger.info("🚀 Initializing Embedding model: %s", self.model_name)
            self.logger.logger.info("  - Using FP16: %s", self.use_fp16)
            self.logger.logger.info("  - Max sequence length: %d", self.max_seq_length)
        try:
            self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
            # Try a test sentence
            test_embedding = self.model.encode(["test"])
            if isinstance(test_embedding, dict):
                self.embedding_dim = test_embedding["dense_vecs"].shape[1]
            else:
                self.embedding_dim = test_embedding.shape[1]

            load_cost_time = time.perf_counter() - start
            if self.logger:
                self.logger.logger.info("✅ model loaded successfully in %.4f seconds", load_cost_time)
                self.logger.logger.info("  - Embedding dimension: %d", self.embedding_dim)
                self.logger.logger.info("  - Model supports: dense, sparse, and multi-vector retrieval")
        except Exception as e:
            if self.logger:
                self.logger.logger.exception("Failed to load model: ", e)
            raise

    def encode_text(
        self,
        text: str,
        return_sparse: bool = False,
        return_colbert: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Encode a single text into embeddings.

        Args:
            text: Input text to encode
            return_sparse: Whether to return sparse embeddings
            return_colbert: Whether to return ColBERT embeddings

        Returns:
            Dictionary containing different types of embeddings
        """
        embeddings = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )
        print(embeddings)
        dense_vec = embeddings["dense_vecs"][0]
        if return_sparse and "lexical_weights" in embeddings:
            pass

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim

    def compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: Literal["cosine", "euclidean", "dot"] = "cosine",
    ):
        """
        Compute similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric ("cosine", "edclidean", "dot")

        Returns:
            Similarity score
        """
        match metric.lower():
            case "cosine":
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                similarity = dot_product / (norm1 * norm2)
            case "euclidean":
                similarity = -np.linalg.norm(vec1 - vec2)
            case "dot":
                similarity = np.dot(vec1, vec2)
            case _:
                raise ValueError(f"Unsupported similarity metric: {metric}, valid metrics: cosine, edclidean, dot")
        if self.logger:
            self.logger.logger.debug("  Similarity (%s): %.6f", metric, similarity)
        return float(similarity)


if __name__ == "__main__":
    EmbeddingService().encode_text("test", True, True)
