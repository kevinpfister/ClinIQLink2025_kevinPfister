"""MedCPT retriever that uses MedCPT models for biomedical information retrieval."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .bm25_retriever import BM25Retriever
from .config import MODEL_CONFIGS, DEFAULTS
from .utils import concat


class MedCPTRetriever(BM25Retriever):
    """MedCPT retriever that uses MedCPT models for biomedical information retrieval."""
    
    def __init__(
        self, 
        retriever_name: str = "medcpt", 
        corpus_name: str = "textbooks", 
        db_dir: str = "./corpus", 
        **kwargs
    ):
        """
        Initialize MedCPT retriever.
        
        Args:
            retriever_name: Name of the retriever
            corpus_name: Name of the corpus
            db_dir: Database directory path
            **kwargs: Additional arguments
        """
        super().__init__(retriever_name, corpus_name, db_dir, **kwargs)
        
        # Initialize MedCPT models
        self._initialize_medcpt_models()
    
    def _initialize_medcpt_models(self) -> None:
        """Initialize MedCPT query and article encoders."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            print("Loading MedCPT models...")
            
            config = MODEL_CONFIGS["medcpt"]
            
            # Load query encoder
            self.query_tokenizer = AutoTokenizer.from_pretrained(config["query_encoder"])
            self.query_encoder = AutoModel.from_pretrained(config["query_encoder"])
            
            # Load article encoder
            self.article_tokenizer = AutoTokenizer.from_pretrained(config["article_encoder"])
            self.article_encoder = AutoModel.from_pretrained(config["article_encoder"])
            
            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.query_encoder.to(self.device)
            self.article_encoder.to(self.device)
            
            # Set to evaluation mode
            self.query_encoder.eval()
            self.article_encoder.eval()
            
            # Store configuration
            self.max_length = config["max_length"]
            self.batch_size = config["batch_size"]
            self.medcpt_weight = config["medcpt_weight"]
            self.bm25_weight = config["bm25_weight"]
            
            print(f"MedCPT retriever initialized on {self.device}")
            
        except ImportError:
            raise ImportError(
                "transformers package required. Install with 'pip install transformers'"
            )
        except Exception as e:
            print(f"Error initializing MedCPT models: {e}")
            raise
    
    def get_relevant_documents(
        self, 
        question: str, 
        k: int = DEFAULTS["retrieval_k"], 
        id_only: bool = False, 
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents using BM25 + MedCPT re-ranking.
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            id_only: Whether to return only document IDs
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (documents, scores)
        """
        self._validate_question(question)
        
        # Step 1: Get more candidates from BM25 than needed
        candidate_k = min(k * DEFAULTS["candidate_multiplier"], DEFAULTS["max_candidates"])
        
        try:
            hits = self.index.search(question, k=candidate_k)
            
            if not hits:
                return [], []
            
            # Extract document IDs and BM25 scores
            ids = [hit.docid for hit in hits]
            bm25_scores = [hit.score for hit in hits]
            
            # Parse document indices
            indices = [self._parse_document_id(hit.docid) for hit in hits]
            
            # Get full documents for re-ranking
            docs = self._get_documents(indices, ids)
            
            if not docs:
                return [], []
            
            # Step 2: Re-rank using MedCPT
            reranked_docs, reranked_scores = self._medcpt_rerank(
                question, docs, bm25_scores
            )
            
            # Return top k
            top_k_docs = reranked_docs[:k]
            top_k_scores = reranked_scores[:k]
            
            if id_only:
                return [
                    {"id": doc.get("id", f"doc_{i}")} 
                    for i, doc in enumerate(top_k_docs)
                ], top_k_scores
            else:
                return top_k_docs, top_k_scores
            
        except Exception as e:
            print(f"Error during MedCPT search: {e}")
            return [], []
    
    def _medcpt_rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        bm25_scores: List[float]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Re-rank documents using MedCPT similarity.
        
        Args:
            query: Query string
            documents: List of documents to re-rank
            bm25_scores: Original BM25 scores
            
        Returns:
            Tuple of (reranked_documents, reranked_scores)
        """
        if not documents:
            return [], []
        
        try:
            import torch
            
            # Encode query
            query_embedding = self._encode_query(query)
            
            # Encode documents in batches
            doc_embeddings = self._encode_documents(documents)
            
            # Compute similarities
            similarities = self._compute_similarities(query_embedding, doc_embeddings)
            
            # Combine BM25 and MedCPT scores
            combined_scores = self._combine_scores(bm25_scores, similarities)
            
            # Sort by combined score (descending)
            sorted_indices = np.argsort(combined_scores)[::-1]
            
            reranked_docs = [documents[i] for i in sorted_indices]
            reranked_scores = [float(combined_scores[i]) for i in sorted_indices]
            
            return reranked_docs, reranked_scores
            
        except Exception as e:
            print(f"Error during MedCPT re-ranking: {e}")
            # Fallback to original order
            return documents, bm25_scores or [1.0] * len(documents)
    
    def _encode_query(self, query: str) -> 'torch.Tensor':
        """
        Encode query using MedCPT query encoder.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding tensor
        """
        import torch
        
        query_inputs = self.query_tokenizer(
            query, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            query_outputs = self.query_encoder(**query_inputs)
            query_embedding = query_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return query_embedding
    
    def _encode_documents(self, documents: List[Dict[str, Any]]) -> 'torch.Tensor':
        """
        Encode documents using MedCPT article encoder in batches.
        
        Args:
            documents: List of documents to encode
            
        Returns:
            Document embeddings tensor
        """
        import torch
        
        # Prepare document texts
        doc_texts = self._prepare_document_texts(documents)
        
        # Encode documents in batches
        doc_embeddings = []
        
        for i in range(0, len(doc_texts), self.batch_size):
            batch = doc_texts[i:i + self.batch_size]
            batch_embeddings = self._encode_document_batch(batch)
            doc_embeddings.append(batch_embeddings)
        
        # Concatenate all document embeddings
        if doc_embeddings:
            return torch.cat(doc_embeddings, dim=0)
        else:
            return torch.empty(0, 768)  # Empty tensor with BERT embedding dimension
    
    def _prepare_document_texts(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Prepare document texts for encoding.
        
        Args:
            documents: List of documents
            
        Returns:
            List of prepared text strings
        """
        doc_texts = []
        
        for doc in documents:
            title = doc.get('title', '')
            content = doc.get('content', '')
            
            # MedCPT expects [title, abstract] format for articles
            if title and content:
                # Truncate content to avoid memory issues
                truncated_content = content[:1024]
                combined_text = concat(title, truncated_content)
            else:
                combined_text = title or content or 'No content'
            
            doc_texts.append(combined_text)
        
        return doc_texts
    
    def _encode_document_batch(self, batch_texts: List[str]) -> 'torch.Tensor':
        """
        Encode a batch of document texts.
        
        Args:
            batch_texts: List of document texts
            
        Returns:
            Batch embeddings tensor
        """
        import torch
        
        # Tokenize batch
        inputs = self.article_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.article_encoder(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return batch_embeddings
    
    def _compute_similarities(
        self, 
        query_embedding: 'torch.Tensor', 
        doc_embeddings: 'torch.Tensor'
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and documents.
        
        Args:
            query_embedding: Query embedding tensor
            doc_embeddings: Document embeddings tensor
            
        Returns:
            Array of similarity scores
        """
        import torch
        
        if doc_embeddings.numel() == 0:
            return np.array([])
        
        # Compute cosine similarities
        query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        similarities = torch.mm(query_norm, doc_norm.transpose(0, 1))[0].cpu().numpy()
        
        return similarities
    
    def _combine_scores(
        self, 
        bm25_scores: List[float], 
        medcpt_scores: np.ndarray
    ) -> np.ndarray:
        """
        Combine BM25 and MedCPT scores.
        
        Args:
            bm25_scores: BM25 scores
            medcpt_scores: MedCPT similarity scores
            
        Returns:
            Combined scores array
        """
        # Normalize BM25 scores
        if bm25_scores:
            bm25_array = np.array(bm25_scores)
            if bm25_array.std() > 0:
                bm25_normalized = (bm25_array - bm25_array.mean()) / bm25_array.std()
            else:
                bm25_normalized = bm25_array
        else:
            bm25_normalized = np.zeros(len(medcpt_scores))
        
        # Normalize MedCPT scores
        if medcpt_scores.std() > 0:
            medcpt_normalized = (medcpt_scores - medcpt_scores.mean()) / medcpt_scores.std()
        else:
            medcpt_normalized = medcpt_scores
        
        # Combine scores (weighted combination for biomedical domain)
        combined_scores = (
            self.medcpt_weight * medcpt_normalized + 
            self.bm25_weight * bm25_normalized
        )
        
        return combined_scores