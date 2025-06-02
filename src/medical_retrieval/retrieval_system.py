"""Main retrieval system that orchestrates multiple retrievers and corpora."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

from .config import CORPUS_NAMES, RETRIEVER_NAMES, DEFAULTS
from .bm25_retriever import BM25Retriever
from .medcpt_retriever import MedCPTRetriever
from .splade_retriever import SPLADERetriever
from .unicoil_retriever import UniCOILRetriever
from .doc_extracter import DocExtracter


class RetrievalSystem:
    """Main retrieval system that orchestrates multiple retrievers and corpora."""
    
    def __init__(
        self, 
        retriever_name: str = "BM25", 
        corpus_name: str = "Textbooks", 
        db_dir: str = "./corpus", 
        HNSW: bool = False, 
        cache: bool = False
    ):
        """
        Initialize retrieval system.
        
        Args:
            retriever_name: Name of retriever configuration
            corpus_name: Name of corpus configuration
            db_dir: Database directory path
            HNSW: Whether to use HNSW (currently unused)
            cache: Whether to enable document caching
        """
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache = cache
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize retrievers
        self.retrievers = self._initialize_retrievers()
        
        # Set up document extraction if caching is enabled
        self.docExt = None
        if self.cache:
            self.docExt = DocExtracter(
                cache=cache, 
                corpus_name=self.corpus_name, 
                db_dir=db_dir
            )
    
    def _validate_configuration(self) -> None:
        """Validate retriever and corpus configuration."""
        if self.corpus_name not in CORPUS_NAMES:
            raise ValueError(f"Invalid corpus name: {self.corpus_name}")
        
        if self.retriever_name not in RETRIEVER_NAMES:
            raise ValueError(f"Invalid retriever name: {self.retriever_name}")
    
    def _initialize_retrievers(self) -> List[List[Optional['BaseRetriever']]]:
        """
        Initialize all retrievers for each corpus.
        
        Returns:
            Nested list of retrievers [retriever_type][corpus_index]
        """
        retrievers = []
        
        for retriever_type in RETRIEVER_NAMES[self.retriever_name]:
            retriever_list = []
            
            for corpus in CORPUS_NAMES[self.corpus_name]:
                print(f"Initializing {retriever_type} retriever for {corpus}...")
                
                try:
                    retriever = self._create_retriever(retriever_type, corpus)
                    retriever_list.append(retriever)
                except Exception as e:
                    print(f"Error initializing retriever for {corpus}: {e}")
                    # Add None as placeholder for failed retrievers
                    retriever_list.append(None)
            
            retrievers.append(retriever_list)
        
        return retrievers
    
    def _create_retriever(self, retriever_type: str, corpus: str) -> 'BaseRetriever':
        """
        Create a specific retriever instance.
        
        Args:
            retriever_type: Type of retriever to create
            corpus: Corpus name
            
        Returns:
            Retriever instance
        """
        retriever_classes = {
            "bm25": BM25Retriever,
            "medcpt": MedCPTRetriever,
            "splade": SPLADERetriever,
            "unicoil": UniCOILRetriever
        }
        
        if retriever_type not in retriever_classes:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        retriever_class = retriever_classes[retriever_type]
        return retriever_class(retriever_type, corpus, self.db_dir)
    
    def retrieve(
        self, 
        question: str, 
        k: int = DEFAULTS["retrieval_k"], 
        rrf_k: int = DEFAULTS["rrf_k"], 
        id_only: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents for a question across all corpora.
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            rrf_k: RRF parameter for score fusion
            id_only: Whether to return only document IDs
            
        Returns:
            Tuple of (documents, scores)
        """
        if not isinstance(question, str):
            raise TypeError("Question must be a string")
        
        # Adjust id_only if caching is enabled
        output_id_only = id_only
        if self.cache:
            id_only = True
        
        # Collect results from all retrievers
        texts, scores = self._collect_retriever_results(question, k, id_only)
        
        # Check if we have any results
        if self._no_results_found(texts):
            print("⚠️ No results found in any retriever/corpus combination")
            return [], []
        
        # Merge results using appropriate strategy
        try:
            merged_texts, merged_scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)
        except Exception as e:
            print(f"⚠️ Error merging results: {e}")
            return [], []
        
        # Apply document extraction if caching is enabled
        if self.cache and merged_texts and not output_id_only:
            try:
                merged_texts = self.docExt.extract(merged_texts)
            except Exception as e:
                print(f"⚠️ Error extracting documents: {e}")
                return [], []
        
        return merged_texts, merged_scores
    
    def _collect_retriever_results(
        self, 
        question: str, 
        k: int, 
        id_only: bool
    ) -> Tuple[List[List[List[Dict[str, Any]]]], List[List[List[float]]]]:
        """
        Collect results from all retrievers and corpora.
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            id_only: Whether to return only document IDs
            
        Returns:
            Tuple of (texts, scores) with structure [retriever][corpus][results]
        """
        texts = []
        scores = []
        
        for i, retriever_group in enumerate(self.retrievers):
            texts.append([])
            scores.append([])
            
            for j, retriever in enumerate(retriever_group):
                if retriever is None:
                    texts[-1].append([])
                    scores[-1].append([])
                    continue
                
                try:
                    t, s = retriever.get_relevant_documents(
                        question, k=k, id_only=id_only
                    )
                    texts[-1].append(t)
                    scores[-1].append(s)
                except Exception as e:
                    retriever_name = RETRIEVER_NAMES[self.retriever_name][i]
                    corpus_name = CORPUS_NAMES[self.corpus_name][j]
                    print(f"⚠️ Error in retriever {retriever_name}, corpus {corpus_name}: {e}")
                    texts[-1].append([])
                    scores[-1].append([])
        
        return texts, scores
    
    def _no_results_found(self, texts: List[List[List[Dict[str, Any]]]]) -> bool:
        """
        Check if no results were found in any retriever/corpus combination.
        
        Args:
            texts: Nested list of results
            
        Returns:
            True if no results found
        """
        return all(all(len(x) == 0 for x in row) for row in texts)
    
    def merge(
        self, 
        texts: List[List[List[Dict[str, Any]]]], 
        scores: List[List[List[float]]], 
        k: int = DEFAULTS["retrieval_k"], 
        rrf_k: int = DEFAULTS["rrf_k"]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Merge results from different corpora and retrievers.
        
        Args:
            texts: Nested list of document results
            scores: Nested list of scores
            k: Number of final results to return
            rrf_k: RRF parameter for score fusion
            
        Returns:
            Tuple of (merged_documents, merged_scores)
        """
        # For single retriever cases
        if len(texts) == 1:
            return self._merge_single_retriever(texts[0], scores[0], k)
        
        # For hybrid retrievers, use RRF (Reciprocal Rank Fusion)
        return self._merge_with_rrf(texts, scores, k, rrf_k)
    
    def _merge_single_retriever(
        self, 
        corpus_texts: List[List[Dict[str, Any]]], 
        corpus_scores: List[List[float]], 
        k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Merge results from multiple corpora for a single retriever.
        
        Args:
            corpus_texts: List of results from each corpus
            corpus_scores: List of scores from each corpus
            k: Number of final results
            
        Returns:
            Tuple of (merged_documents, merged_scores)
        """
        all_texts = []
        all_scores = []
        
        for texts, scores in zip(corpus_texts, corpus_scores):
            if not texts:  # Skip empty results
                continue
            
            all_texts.extend(texts)
            all_scores.extend(scores)
        
        if not all_texts:
            return [], []
        
        # Sort by score (descending)
        sorted_indices = np.argsort(all_scores)[::-1]
        sorted_texts = [all_texts[i] for i in sorted_indices]
        sorted_scores = [all_scores[i] for i in sorted_indices]
        
        # Return top k results
        return sorted_texts[:k], sorted_scores[:k]
    
    def _merge_with_rrf(
        self, 
        texts: List[List[List[Dict[str, Any]]]], 
        scores: List[List[List[float]]], 
        k: int, 
        rrf_k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        Args:
            texts: Nested list of document results
            scores: Nested list of scores
            k: Number of final results
            rrf_k: RRF parameter
            
        Returns:
            Tuple of (merged_documents, merged_scores)
        """
        rrf_dict = {}
        
        for i in range(len(RETRIEVER_NAMES[self.retriever_name])):
            # Collect all results for this retriever type
            texts_all, scores_all = self._collect_retriever_type_results(
                texts[i], scores[i]
            )
            
            if not texts_all:
                continue
            
            # Sort by scores (highest first)
            sorted_results = self._sort_by_scores(texts_all, scores_all)
            
            # Add to RRF dictionary
            self._add_to_rrf_dict(sorted_results, rrf_dict, rrf_k)
        
        if not rrf_dict:
            return [], []
        
        # Sort RRF results and return top k
        return self._finalize_rrf_results(rrf_dict, k)
    
    def _collect_retriever_type_results(
        self, 
        retriever_texts: List[List[Dict[str, Any]]], 
        retriever_scores: List[List[float]]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Collect all results for a specific retriever type across corpora.
        
        Args:
            retriever_texts: Texts from all corpora for this retriever
            retriever_scores: Scores from all corpora for this retriever
            
        Returns:
            Tuple of (all_texts, all_scores)
        """
        texts_all = []
        scores_all = []
        
        for corpus_texts, corpus_scores in zip(retriever_texts, retriever_scores):
            if corpus_texts:
                texts_all.extend(corpus_texts)
                scores_all.extend(corpus_scores)
        
        return texts_all, scores_all
    
    def _sort_by_scores(
        self, 
        texts: List[Dict[str, Any]], 
        scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Sort texts by their scores in descending order.
        
        Args:
            texts: List of documents
            scores: List of scores
            
        Returns:
            List of sorted documents
        """
        if not texts:
            return []
        
        sorted_indices = np.argsort(scores)[::-1]
        return [texts[i] for i in sorted_indices]
    
    def _add_to_rrf_dict(
        self, 
        sorted_texts: List[Dict[str, Any]], 
        rrf_dict: Dict[str, Dict[str, Any]], 
        rrf_k: int
    ) -> None:
        """
        Add sorted results to RRF dictionary.
        
        Args:
            sorted_texts: Sorted list of documents
            rrf_dict: RRF accumulation dictionary
            rrf_k: RRF parameter
        """
        for j, item in enumerate(sorted_texts):
            if not isinstance(item, dict) or "id" not in item:
                continue
            
            item_id = item["id"]
            rrf_score = 1 / (rrf_k + j + 1)
            
            if item_id in rrf_dict:
                rrf_dict[item_id]["score"] += rrf_score
                rrf_dict[item_id]["count"] += 1
            else:
                rrf_dict[item_id] = {
                    "id": item_id,
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "score": rrf_score,
                    "count": 1
                }
    
    def _finalize_rrf_results(
        self, 
        rrf_dict: Dict[str, Dict[str, Any]], 
        k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Finalize RRF results by sorting and extracting top k.
        
        Args:
            rrf_dict: RRF accumulation dictionary
            k: Number of results to return
            
        Returns:
            Tuple of (final_documents, final_scores)
        """
        rrf_list = sorted(rrf_dict.items(), key=lambda x: x[1]["score"], reverse=True)
        
        result_texts = [
            {key: item[1][key] for key in ("id", "title", "content") if key in item[1]} 
            for item in rrf_list[:k]
        ]
        result_scores = [item[1]["score"] for item in rrf_list[:k]]
        
        return result_texts, result_scores
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the retrieval system configuration.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "retriever_name": self.retriever_name,
            "corpus_name": self.corpus_name,
            "db_directory": self.db_dir,
            "cache_enabled": self.cache,
            "retriever_types": RETRIEVER_NAMES[self.retriever_name],
            "corpus_types": CORPUS_NAMES[self.corpus_name],
            "num_retrievers": len(self.retrievers),
            "retriever_status": []
        }
        
        # Add status of each retriever
        for i, retriever_group in enumerate(self.retrievers):
            retriever_type = RETRIEVER_NAMES[self.retriever_name][i]
            status = {
                "retriever_type": retriever_type,
                "corpora_status": []
            }
            
            for j, retriever in enumerate(retriever_group):
                corpus_name = CORPUS_NAMES[self.corpus_name][j]
                status["corpora_status"].append({
                    "corpus": corpus_name,
                    "status": "active" if retriever is not None else "failed"
                })
            
            info["retriever_status"].append(status)
        
        if self.cache and self.docExt:
            info["cache_stats"] = self.docExt.get_cache_stats()
        
        return info