"""In-memory document store for managing documents."""

import uuid
from datetime import datetime
from typing import Any, Optional

from app_logger import VectorSearchLogger


class Document:
    def __init__(
        self,
        id: str,
        text: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        embedding: Optional[list[float]] = None,
    ):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.created_at = created_at or datetime.now(tz=datetime.now().astimezone().tzinfo)
        self.embedding = embedding


class DocumentStore:
    """In-memory document storage."""

    def __init__(self, logger: Optional[VectorSearchLogger] = None):
        self.documents: dict[str, Document] = dict()
        self.logger = logger
        if self.logger:
            self.logger.logger.info("Initialized in-memory document store")

    def add_document(self, text: str, doc_id: Optional[str] = None, metadata: Optional[dict] = None) -> str:
        """
        Add a document to the store.

        Args:
            text: Document text
            doc_id: Optional document ID (will be generated if not provided)
            metadata: Optional metadata dictionary

        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        if doc_id in self.documents:
            if self.logger:
                self.logger.logger.warning("Document %s already exist, updating...", doc_id)
        # Store document, maybe updated
        self.documents[doc_id] = Document(doc_id, text, metadata=metadata)
        if self.logger:
            self.logger.logger.debug("Stored document %s", doc_id)

        return doc_id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document or None if not found
        """
        doc = self.documents.get(doc_id, None)

        if self.logger:
            if doc:
                self.logger.logger.debug("Retrieved document %s", doc_id)
            else:
                self.logger.logger.warning("Document %s not found", doc_id)

        return doc

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        if doc_id in self.documents:
            del self.documents[doc_id]

            if self.logger:
                self.logger.logger.debug("Deleted document %s, remaining documents: %d", doc_id, len(self.documents))

            return True
        else:
            return False

    def list_documents(self, limit: Optional[int] = None) -> list[Document]:
        """
        List all documents in the store.

        Args:
            limit: Maximum number of documents to return

        Returns:
            List of documents
        """
        all_docs = list(self.documents.values())
        if limit is not None:
            all_docs = all_docs[:limit]
        if self.logger:
            self.logger.logger.debug("Listing %d documents", len(all_docs))
        return all_docs

    def get_documents_by_ids(self, doc_ids: list[str]) -> list[Document]:
        """
        Retrieve documents by IDs.

        Args:
            doc_ids: List of document IDs

        Returns:
            List of documents (only those found)
        """
        docs = []
        for doc_id in doc_ids:
            doc = self.documents.get(doc_id, None)
            if doc is not None:
                docs.append(doc)
        if self.logger:
            self.logger.logger.debug("Retrived %d/%d documents", len(docs), len(doc_ids))
        return docs

    def update_document_embedding(self, doc_id: str, embedding: list[float]) -> bool:
        """
        Update the embedding for a document.

        Args:
            doc_id: Document ID
            embedding: Embedding vector

        Returns:
            True if updated, False if not found
        """
        if doc_id in self.documents:
            self.documents[doc_id].embedding = embedding

            if self.logger:
                self.logger.logger.debug("Updated embedding for document %s", doc_id)

            return True
        else:
            return False

    def get_size(self) -> int:
        """Get the number of documents in the store."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear all documents from the store."""
        count = len(self.documents)
        self.documents.clear()
        if self.logger:
            self.logger.logger.debug("Cleared %d documents from store", count)
