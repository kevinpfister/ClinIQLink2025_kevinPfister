"""BM25 retriever implementation using Lucene."""

import json
import os
from typing import List, Dict, Any, Tuple, Optional

from pyserini.search.lucene import LuceneSearcher

from .base_retriever import BaseRetriever
from .config import DEFAULTS
from .utils import (
    batch_index_files, 
    create_minimal_index, 
    determine_batch_size, 
    get_jsonl_files,
    clean_directory,
    ensure_directory
)


class BM25Retriever(BaseRetriever):
    """BM25 retriever using Lucene search index."""
    
    def __init__(
        self, 
        retriever_name: str = "bm25", 
        corpus_name: str = "textbooks", 
        db_dir: str = "./corpus", 
        **kwargs
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            retriever_name: Name of the retriever
            corpus_name: Name of the corpus
            db_dir: Database directory path
            **kwargs: Additional arguments
        """
        super().__init__(retriever_name, corpus_name, db_dir, **kwargs)
    
    def _initialize_index(self) -> None:
        """Initialize or load the BM25 index."""
        try:
            if self._index_exists() and self._index_is_valid():
                print(f"Loading existing BM25 index for {self.corpus_name}...")
                self.index = LuceneSearcher(self.index_dir)
                print(f"Successfully loaded BM25 index for {self.corpus_name}")
            else:
                print(f"Creating BM25 index for {self.corpus_name}...")
                self._create_index()
        except Exception as e:
            print(f"Error during index initialization: {e}")
            print("Attempting to create a working index...")
            self._create_index()
    
    def _index_exists(self) -> bool:
        """Check if index directory exists and is not empty."""
        return os.path.exists(self.index_dir) and os.listdir(self.index_dir)
    
    def _index_is_valid(self) -> bool:
        """Check if the existing index can be loaded."""
        try:
            test_searcher = LuceneSearcher(self.index_dir)
            # Try a simple search to validate
            test_searcher.search("test", k=1)
            return True
        except Exception:
            return False
    
    def _create_index(self) -> None:
        """Create BM25 index for the corpus using batch processing."""
        # Clean up any existing corrupted index
        clean_directory(self.index_dir)
        ensure_directory(os.path.dirname(self.index_dir))
        
        # Validate chunk directory
        if not os.path.exists(self.chunk_dir) or not os.listdir(self.chunk_dir):
            raise RuntimeError(
                f"Chunk directory {self.chunk_dir} is empty. Cannot create index."
            )
        
        # Count JSONL files and determine batch size
        jsonl_files = get_jsonl_files(self.chunk_dir)
        file_count = len(jsonl_files)
        print(f"Found {file_count} JSONL files to index in {self.chunk_dir}")
        
        batch_size = determine_batch_size(file_count)
        if batch_size is None:
            batch_size = file_count  # Process all at once
        
        print(f"Using batch size of {batch_size} files")
        
        # Try batch indexing
        success = batch_index_files(
            input_dir=self.chunk_dir,
            output_dir=self.index_dir,
            batch_size=batch_size
        )
        
        # If batch indexing fails, try minimal index as last resort
        if not success:
            print("Batch indexing failed. Creating minimal working index...")
            success = create_minimal_index(
                input_dir=self.chunk_dir,
                output_dir=self.index_dir,
                file_limit=DEFAULTS["minimal_index_files"]
            )
            
            if not success:
                raise RuntimeError(f"Failed to create any working index for {self.corpus_name}")
        
        # Load the created index
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the created index and validate it works."""
        try:
            self.index = LuceneSearcher(self.index_dir)
            print(f"Successfully loaded BM25 index for {self.corpus_name}")
        except Exception as e:
            raise RuntimeError(f"Created index but failed to load it: {e}")
    
    def get_relevant_documents(
        self, 
        question: str, 
        k: int = DEFAULTS["retrieval_k"], 
        id_only: bool = False, 
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant documents for a question using BM25.
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            id_only: Whether to return only document IDs
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (documents, scores)
        """
        self._validate_question(question)
        
        try:
            hits = self.index.search(question, k=k)
            
            if not hits:
                return [], []
            
            # Extract scores and document IDs
            scores = [hit.score for hit in hits]
            ids = [hit.docid for hit in hits]
            
            # Parse document indices
            indices = [self._parse_document_id(hit.docid) for hit in hits]
            
            # Return IDs only if requested
            if id_only:
                return [{"id": doc_id} for doc_id in ids], scores
            
            # Try to retrieve full documents
            docs = self._get_documents(indices, ids)
            return docs, scores
            
        except Exception as e:
            print(f"Error during search: {e}")
            return [], []
    
    def _get_documents(
        self, 
        indices: List[Dict[str, Any]], 
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get full documents either directly from files or from index.
        
        Args:
            indices: List of parsed document indices
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        try:
            docs = self._get_docs_direct(indices)
            if docs:
                return docs
        except Exception as e:
            print(f"Error retrieving documents from files: {e}")
        
        # Fallback to getting docs from the index
        return self._get_docs_from_index(ids)
    
    def _get_docs_from_index(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents directly from the index.
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        docs = []
        for doc_id in ids:
            try:
                doc = self.index.doc(doc_id)
                if doc:
                    parsed_doc = self._parse_index_document(doc, doc_id)
                    docs.append(parsed_doc)
                else:
                    docs.append(self._create_missing_document(doc_id))
            except Exception as e:
                print(f"Error retrieving document {doc_id}: {e}")
                docs.append(self._create_error_document(doc_id, str(e)))
        
        return docs
    
    def _parse_index_document(self, doc, doc_id: str) -> Dict[str, Any]:
        """
        Parse document retrieved from index.
        
        Args:
            doc: Document from index
            doc_id: Document ID
            
        Returns:
            Parsed document
        """
        try:
            # Try to parse as JSON
            content = json.loads(doc.raw())
            return self._normalize_document(content)
        except json.JSONDecodeError:
            # If parsing fails, use raw content
            return {
                'id': doc_id,
                'title': 'Document',
                'content': doc.raw()
            }