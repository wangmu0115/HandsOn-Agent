"""Vector index implementations using ANNOY and HNSW."""

from abc import ABC, abstractmethod
from typing import Optional

import annoy
import numpy as np
from app_logger import VectorSearchLogger


class VectorIndex(ABC):
    """Abstract Base Class for vector indexes."""

    @abstractmethod
    def add_item(self, doc_id: str, vector: np.ndarray) -> None:
        """Add an item to the index."""
        pass

    @abstractmethod
    def delete_item(self, doc_id: str) -> bool:
        """Delete an item from the index."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        """Search for top-k similar items."""
        pass

    @abstractmethod
    def get_size(self) -> int:
        """Get the current number of items in the index."""
        pass

    @abstractmethod
    def rebuild_index(self) -> None:
        """Rebuild the index if necessary."""
        pass


class AnnoyIndex(VectorIndex):
    """ANNOY-based vector index implementation."""

    def __init__(
        self,
        dimension: int,
        n_trees: int = 50,
        metric: str = "angular",
        logger: Optional[VectorSearchLogger] = None,
    ):
        """
        Initialize ANNOY index.

        Args:
            dimension: Dimension of vectors
            n_trees: Number of trees for ANNOY (affects precision/speed tradeoff)
            metric: Distance metric ('angular', 'euclidean', 'manhattan', 'hamming', 'dot')
            logger: Logger instance for educational output
        """
        self.dimension = dimension
        self.n_trees = n_trees
        self.metric = metric
        self.logger = logger

        # Create index
        self.index = annoy.AnnoyIndex(dimension, metric)

        # Mapping between internal indices and document IDs
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.vectors_cache: Dict[int, np.ndarray] = {}
        self.next_index = 0
        self.is_built = False

        if self.logger:
            self.logger.logger.info(f"📚 Initialized ANNOY index")
            self.logger.logger.debug(f"  - Dimension: {dimension}")
            self.logger.logger.debug(f"  - Number of trees: {n_trees}")
            self.logger.logger.debug(f"  - Metric: {metric}")


class HNSWIndex(VectorIndex):
    """HNSW-based vector index implementation."""
