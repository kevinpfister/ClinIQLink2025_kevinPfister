"""UniCOIL retriever using BERT-based term reweighting on existing Lucene index."""

from typing import List, Dict, Any, Tuple, Optional

from .bm25_retriever import BM25Retriever
from .config import MODEL_CONFIGS, DEFAULTS


class UniCOILRetriever(BM25Retriever):
    """UniCOIL retriever using BERT-based term reweighting on existing Lucene index."""
    
    def __init__(
        self, 
        retriever_name: str = "unicoil", 
        corpus_name: str = "textbooks", 
        db_dir: str = "./corpus", 
        **kwargs
    ):
        """
        Initialize UniCOIL retriever.
        
        Args:
            retriever_name: Name of the retriever
            corpus_name: Name of the corpus
            db_dir: Database directory path
            **kwargs: Additional arguments
        """
        # Initialize base BM25 retriever (uses existing BM25 index)
        super().__init__(retriever_name, corpus_name, db_dir, **kwargs)
        
        # Initialize UniCOIL model for term weighting
        self._initialize_unicoil_model()
    
    def _initialize_unicoil_model(self) -> None:
        """Initialize BERT model for UniCOIL-style term weighting."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            print("Loading BERT model for UniCOIL term weighting...")
            
            config = MODEL_CONFIGS["unicoil"]
            model_name = config["model_name"]
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model_name = model_name
            except Exception:
                print("BioBERT not available, using standard BERT...")
                fallback_model = config["fallback_model"]
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.model = AutoModel.from_pretrained(fallback_model)
                self.model_name = fallback_model
            
            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model.eval()
            
            # Store configuration
            self.max_length = config["max_length"]
            self.weight_min, self.weight_max = config["weight_range"]
            
            print(f"UniCOIL term weighting ready with {self.model_name} on {self.device}")
                
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with 'pip install transformers torch'"
            )
        except Exception as e:
            print(f"Error initializing UniCOIL: {e}")
            raise
    
    def get_relevant_documents(
        self, 
        question: str, 
        k: int = DEFAULTS["retrieval_k"], 
        id_only: bool = False, 
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents using UniCOIL term weighting on existing BM25 index.
        
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
            # Weight query terms with UniCOIL
            weighted_query = self._weight_query_terms(question)
            
            # Search using weighted query on existing BM25 index
            hits = self.index.search(weighted_query, k=k)
            
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
            print(f"Error during UniCOIL search: {e}")
            return [], []
    
    def _weight_query_terms(self, query: str) -> str:
        """
        Weight query terms using BERT contextual embeddings.
        
        Args:
            query: Original query string
            
        Returns:
            Weighted query string for Lucene
        """
        try:
            import torch
            
            # Tokenize query
            tokens = self.tokenizer.tokenize(query.lower())
            if not tokens:
                return query
            
            # Get BERT embeddings for context
            inputs = self.tokenizer(
                query, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
                
                # Get token positions (skip [CLS], [SEP])
                input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
                # Calculate term importance weights
                weighted_terms = self._calculate_term_weights(input_tokens, hidden_states)
                
                # Create weighted query
                if weighted_terms:
                    weighted_query = " ".join(weighted_terms)
                    print(f"UniCOIL weighted: '{query}' -> '{weighted_query}'")
                    return weighted_query
                else:
                    return query
                
        except Exception as e:
            print(f"Error weighting query terms: {e}")
            return query
    
    def _calculate_term_weights(
        self, 
        input_tokens: List[str], 
        hidden_states: 'torch.Tensor'
    ) -> List[str]:
        """
        Calculate term weights based on BERT hidden states.
        
        Args:
            input_tokens: List of tokenized input tokens
            hidden_states: BERT hidden states tensor
            
        Returns:
            List of weighted terms for Lucene query
        """
        import torch
        
        weighted_terms = []
        
        # Skip [CLS] and [SEP] tokens
        for i, token in enumerate(input_tokens[1:-1], 1):
            if token.startswith('##'):  # Skip sub-word tokens
                continue
            
            # Compute term weight based on hidden state norm
            hidden_norm = torch.norm(hidden_states[i]).item()
            
            # Normalize and scale weight
            weight = self._normalize_weight(hidden_norm)
            
            # Create weighted term for Lucene
            weighted_term = self._format_weighted_term(token, weight)
            weighted_terms.append(weighted_term)
        
        return weighted_terms
    
    def _normalize_weight(self, hidden_norm: float) -> float:
        """
        Normalize weight to acceptable range.
        
        Args:
            hidden_norm: Raw hidden state norm
            
        Returns:
            Normalized weight
        """
        # Simple weighting: normalize and scale
        weight = hidden_norm / 10.0  # Scale down
        
        # Clamp to acceptable range
        return max(min(weight, self.weight_max), self.weight_min)
    
    def _format_weighted_term(self, token: str, weight: float) -> str:
        """
        Format term with weight for Lucene query.
        
        Args:
            token: Token string
            weight: Normalized weight
            
        Returns:
            Formatted weighted term
        """
        if weight > 1.5:  # High importance
            return f"{token}^{weight:.1f}"
        elif weight < 0.5:  # Low importance  
            return f"{token}^{weight:.1f}"
        else:  # Normal importance
            return token