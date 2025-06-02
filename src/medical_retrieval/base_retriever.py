"""Base retriever class with common functionality."""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .config import DEFAULTS
from .utils import ensure_directory


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    def __init__(
        self, 
        retriever_name: str, 
        corpus_name: str, 
        db_dir: str = "./corpus", 
        **kwargs
    ):
        """
        Initialize base retriever.
        
        Args:
            retriever_name: Name of the retriever type
            corpus_name: Name of the corpus
            db_dir: Database directory path
            **kwargs: Additional arguments
        """
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        
        ensure_directory(self.db_dir)
        
        # Set up directory paths
        self.corpus_dir = os.path.join(self.db_dir, self.corpus_name)
        self.chunk_dir = os.path.join(self.corpus_dir, "chunk")
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", "bm25")
        
        self._initialize_corpus()
        self._initialize_index()
    
    @abstractmethod
    def get_relevant_documents(
        self, 
        question: str, 
        k: int = DEFAULTS["retrieval_k"], 
        id_only: bool = False, 
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            id_only: Whether to return only document IDs
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (documents, scores)
        """
        pass
    
    def _initialize_corpus(self) -> None:
        """Initialize corpus by downloading if necessary."""
        if not os.path.exists(self.chunk_dir):
            self._setup_corpus()
    
    def _setup_corpus(self) -> None:
        """Set up corpus by cloning or downloading."""
        from .utils import clone_corpus, download_statpearls
        
        print(f"Setting up {self.corpus_name} corpus...")
        ensure_directory(os.path.dirname(self.chunk_dir))
        
        if self.corpus_name == "selfcorpus":
            print("Skipping setup for selfcorpus")
            return
        
        # Clone the repository
        success = clone_corpus(self.corpus_name, self.corpus_dir)
        if not success:
            raise RuntimeError(f"Failed to clone corpus {self.corpus_name}")
        
        # Handle special case for StatPearls
        if self.corpus_name == "statpearls":
            success = download_statpearls(self.corpus_dir)
            if not success:
                raise RuntimeError("Failed to download StatPearls corpus")
    
    @abstractmethod
    def _initialize_index(self) -> None:
        """Initialize the search index."""
        pass
    
    def _validate_question(self, question: str) -> None:
        """Validate that question is a string."""
        if not isinstance(question, str):
            raise TypeError("Question must be a string")
    
    def _parse_document_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Parse document ID to extract source and index.
        
        Args:
            doc_id: Document ID string
            
        Returns:
            Dictionary with source and index
        """
        try:
            parts = doc_id.split('_')
            index = int(parts[-1])
            source = '_'.join(parts[:-1])
            return {"source": source, "index": index}
        except (ValueError, IndexError):
            # If parsing fails, use the whole ID
            return {"source": doc_id, "index": 0}
    
    def _get_docs_direct(self, indices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Try to get documents directly from the JSONL files.
        
        Args:
            indices: List of document indices with source and index
            
        Returns:
            List of documents
        """
        results = []
        
        for item in indices:
            try:
                file_path = os.path.join(self.chunk_dir, f"{item['source']}.jsonl")
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if not content:
                    continue
                    
                lines = content.split('\n')
                if item['index'] < len(lines):
                    doc = json.loads(lines[item['index']])
                    # Ensure document has required fields
                    doc = self._normalize_document(doc)
                    results.append(doc)
                    
            except Exception as e:
                print(f"Error retrieving document {item['source']}_{item['index']}: {e}")
                
        return results
    
    def _normalize_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize document to ensure it has required fields.
        
        Args:
            doc: Raw document dictionary
            
        Returns:
            Normalized document
        """
        # Ensure document has title and content
        if 'title' not in doc:
            doc['title'] = doc.get('id', 'Untitled')
        if 'content' not in doc:
            doc['content'] = doc.get('contents', '')
        
        return doc
    
    def _create_error_document(self, doc_id: str, error: str) -> Dict[str, Any]:
        """
        Create an error document when retrieval fails.
        
        Args:
            doc_id: Document ID
            error: Error message
            
        Returns:
            Error document
        """
        return {
            'id': doc_id,
            'title': 'Error',
            'content': f'Error retrieving document: {error}'
        }
    
    def _create_missing_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Create a missing document placeholder.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Missing document placeholder
        """
        return {
            'id': doc_id,
            'title': 'Unknown',
            'content': f'Document with ID {doc_id} not found in index'
        }