"""SPLADE retriever using query expansion with existing Lucene index."""

from typing import List, Dict, Any, Tuple, Optional

from .bm25_retriever import BM25Retriever
from .config import MODEL_CONFIGS, DEFAULTS


class SPLADERetriever(BM25Retriever):
    """SPLADE retriever using query expansion with existing Lucene index."""
    
    def __init__(
        self, 
        retriever_name: str = "splade", 
        corpus_name: str = "textbooks", 
        db_dir: str = "./corpus", 
        **kwargs
    ):
        """
        Initialize SPLADE retriever.
        
        Args:
            retriever_name: Name of the retriever
            corpus_name: Name of the corpus
            db_dir: Database directory path
            **kwargs: Additional arguments
        """
        # Initialize base BM25 retriever (creates/loads BM25 index)
        super().__init__(retriever_name, corpus_name, db_dir, **kwargs)
        
        # Initialize SPLADE model for query expansion
        self._initialize_splade_model()
    
    def _initialize_splade_model(self) -> None:
        """Initialize SPLADE model for query expansion only."""
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch
            
            print("Loading SPLADE model for query expansion...")
            
            config = MODEL_CONFIGS["splade"]
            model_name = config["model_name"]
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            
            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model.eval()
            
            # Store configuration
            self.max_expansion_terms = config["max_expansion_terms"]
            self.relevance_threshold = config["relevance_threshold"]
            
            print(f"SPLADE query expansion ready on {self.device}")
                
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with 'pip install transformers torch'"
            )
        except Exception as e:
            print(f"Error initializing SPLADE: {e}")
            raise
    
    def get_relevant_documents(
        self, 
        question: str, 
        k: int = DEFAULTS["retrieval_k"], 
        id_only: bool = False, 
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents using SPLADE query expansion on existing BM25 index.
        
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
            # Expand query with SPLADE
            expanded_query = self._expand_query(question)
            
            # Search using expanded query on existing BM25 index
            hits = self.index.search(expanded_query, k=k)
            
            if not hits:
                return [], []
            
            # Extract results (same as base BM25Retriever)
            scores = [hit.score for hit in hits]
            ids = [hit.docid for hit in hits]
            
            # Parse document indices
            indices = [self._parse_document_id(hit.docid) for hit in hits]
            
            if id_only:
                return [{"id": doc_id} for doc_id in ids], scores
            
            # Get full documents
            docs = self._get_documents(indices, ids)
            return docs, scores
            
        except Exception as e:
            print(f"Error during SPLADE search: {e}")
            return [], []
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query using SPLADE.
        
        Args:
            query: Original query string
            
        Returns:
            Expanded query string
        """
        try:
            import torch
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    [query], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=256
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                logits = outputs.logits[0]  # First (and only) query
                sparse_rep = torch.log(1 + torch.relu(logits)) * inputs.attention_mask[0].unsqueeze(-1)
                
                # Get top expansion terms
                values, indices = torch.topk(
                    sparse_rep.max(dim=0).values, 
                    k=self.max_expansion_terms * 2  # Get more to filter
                )
                
                expansion_tokens = self._extract_expansion_tokens(values, indices)
                
                # Combine original query with expansions
                if expansion_tokens:
                    expanded = f"{query} {' '.join(expansion_tokens[:self.max_expansion_terms])}"
                    print(f"Query expanded: '{query}' -> '{expanded}'")
                    return expanded
                else:
                    return query
                
        except Exception as e:
            print(f"Error expanding query: {e}")
            return query
    
    def _extract_expansion_tokens(
        self, 
        values: 'torch.Tensor', 
        indices: 'torch.Tensor'
    ) -> List[str]:
        """
        Extract valid expansion tokens from SPLADE output.
        
        Args:
            values: Token importance values
            indices: Token indices
            
        Returns:
            List of expansion tokens
        """
        expansion_tokens = []
        
        for idx, val in zip(indices, values):
            if val > self.relevance_threshold:  # Threshold for relevance
                token = self.tokenizer.decode([idx]).strip()
                if self._is_valid_expansion_token(token):
                    expansion_tokens.append(token)
        
        return expansion_tokens
    
    def _is_valid_expansion_token(self, token: str) -> bool:
        """
        Check if a token is valid for expansion.
        
        Args:
            token: Token to validate
            
        Returns:
            True if token is valid for expansion
        """
        return (
            token and 
            not token.startswith('[') and 
            len(token) > 2 and
            token.isalpha()  # Only alphabetic tokens
        )