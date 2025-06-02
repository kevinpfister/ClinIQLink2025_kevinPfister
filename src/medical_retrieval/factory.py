"""Factory pattern for creating retrievers and related components."""

from typing import Optional, Dict, Any, Union

from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from .splade_retriever import SPLADERetriever
from .unicoil_retriever import UniCOILRetriever
from .medcpt_retriever import MedCPTRetriever
from .retrieval_system import RetrievalSystem
from .doc_extracter import DocExtracter
from .config import RETRIEVER_NAMES, CORPUS_NAMES


class RetrieverFactory:
    """Factory class for creating retriever instances."""
    
    _retriever_classes = {
        "bm25": BM25Retriever,
        "splade": SPLADERetriever,
        "unicoil": UniCOILRetriever,
        "medcpt": MedCPTRetriever
    }
    
    @classmethod
    def create_retriever(
        cls,
        retriever_type: str,
        corpus_name: str,
        db_dir: str = "./corpus",
        **kwargs
    ) -> BaseRetriever:
        """
        Create a retriever instance of the specified type.
        
        Args:
            retriever_type: Type of retriever ("bm25", "splade", "unicoil", "medcpt")
            corpus_name: Name of the corpus
            db_dir: Database directory path
            **kwargs: Additional arguments for the retriever
            
        Returns:
            Retriever instance
            
        Raises:
            ValueError: If retriever_type is not supported
            
        Example:
            >>> factory = RetrieverFactory()
            >>> retriever = factory.create_retriever("bm25", "textbooks")
            >>> docs, scores = retriever.get_relevant_documents("diabetes treatment")
        """
        if retriever_type not in cls._retriever_classes:
            available = list(cls._retriever_classes.keys())
            raise ValueError(f"Unsupported retriever type '{retriever_type}'. Available: {available}")
        
        retriever_class = cls._retriever_classes[retriever_type]
        return retriever_class(
            retriever_name=retriever_type,
            corpus_name=corpus_name,
            db_dir=db_dir,
            **kwargs
        )
    
    @classmethod
    def get_available_retrievers(cls) -> list:
        """
        Get list of available retriever types.
        
        Returns:
            List of available retriever type names
        """
        return list(cls._retriever_classes.keys())
    
    @classmethod
    def get_retriever_info(cls, retriever_type: str) -> Dict[str, Any]:
        """
        Get information about a specific retriever type.
        
        Args:
            retriever_type: Type of retriever
            
        Returns:
            Dictionary with retriever information
            
        Raises:
            ValueError: If retriever_type is not supported
        """
        if retriever_type not in cls._retriever_classes:
            raise ValueError(f"Unsupported retriever type '{retriever_type}'")
        
        retriever_class = cls._retriever_classes[retriever_type]
        
        return {
            "type": retriever_type,
            "class": retriever_class.__name__,
            "description": retriever_class.__doc__ or "No description available",
            "module": retriever_class.__module__
        }


class SystemFactory:
    """Factory class for creating complete retrieval systems."""
    
    @staticmethod
    def create_system(
        retriever_name: str = "BM25",
        corpus_name: str = "Textbooks",
        db_dir: str = "./corpus",
        cache: bool = False,
        validate: bool = True,
        **kwargs
    ) -> RetrievalSystem:
        """
        Create a complete retrieval system.
        
        Args:
            retriever_name: Name of retriever configuration
            corpus_name: Name of corpus configuration
            db_dir: Database directory path
            cache: Whether to enable document caching
            validate: Whether to validate configuration before creation
            **kwargs: Additional arguments
            
        Returns:
            Configured RetrievalSystem instance
            
        Raises:
            ValueError: If configuration is invalid and validate=True
            
        Example:
            >>> system = SystemFactory.create_system("MedicalHybrid", "MedCorp", cache=True)
            >>> docs, scores = system.retrieve("symptoms of hypertension")
        """
        if validate:
            SystemFactory.validate_system_config(retriever_name, corpus_name)
        
        return RetrievalSystem(
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir,
            cache=cache,
            **kwargs
        )
    
    @staticmethod
    def create_simple_system(
        corpus: str = "textbooks",
        retriever: str = "bm25",
        db_dir: str = "./corpus",
        **kwargs
    ) -> RetrievalSystem:
        """
        Create a simple single-corpus, single-retriever system.
        
        Args:
            corpus: Single corpus name
            retriever: Single retriever type
            db_dir: Database directory path
            **kwargs: Additional arguments
            
        Returns:
            Simple RetrievalSystem instance
            
        Example:
            >>> system = SystemFactory.create_simple_system("pubmed", "medcpt")
            >>> docs, scores = system.retrieve("cancer treatment")
        """
        # Create temporary configuration for single corpus/retriever
        corpus_config = {f"Custom_{corpus}": [corpus]}
        retriever_config = {f"Custom_{retriever}": [retriever]}
        
        # Temporarily add to global configs (this is a bit hacky but works)
        corpus_key = f"Custom_{corpus}"
        retriever_key = f"Custom_{retriever}"
        
        if corpus_key not in CORPUS_NAMES:
            CORPUS_NAMES[corpus_key] = [corpus]
        
        if retriever_key not in RETRIEVER_NAMES:
            RETRIEVER_NAMES[retriever_key] = [retriever]
        
        return RetrievalSystem(
            retriever_name=retriever_key,
            corpus_name=corpus_key,
            db_dir=db_dir,
            **kwargs
        )
    
    @staticmethod
    def validate_system_config(retriever_name: str, corpus_name: str) -> None:
        """
        Validate system configuration.
        
        Args:
            retriever_name: Name of retriever configuration
            corpus_name: Name of corpus configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if retriever_name not in RETRIEVER_NAMES:
            available = list(RETRIEVER_NAMES.keys())
            raise ValueError(f"Invalid retriever_name '{retriever_name}'. Available: {available}")
        
        if corpus_name not in CORPUS_NAMES:
            available = list(CORPUS_NAMES.keys())
            raise ValueError(f"Invalid corpus_name '{corpus_name}'. Available: {available}")
    
    @staticmethod
    def get_system_configurations() -> Dict[str, Any]:
        """
        Get all available system configurations.
        
        Returns:
            Dictionary with available configurations
        """
        return {
            "retrievers": RETRIEVER_NAMES,
            "corpora": CORPUS_NAMES,
            "recommended_combinations": {
                "medical_research": {
                    "retriever": "TBD",
                    "corpus": "TBD",
                    "description": "Best for medical research queries"
                },
                "TBD": {
                    "retriever": "TBD",
                    "corpus": "TBD", 
                    "description": "TBD"
                },
                "TBD2": {
                    "retriever": "TBD",
                    "corpus": "TBD",
                    "description": "TBD"
                }
            }
        }


class ComponentFactory:
    """Factory for creating individual components."""
    
    @staticmethod
    def create_doc_extracter(
        corpus_name: str = "MedCorp",
        db_dir: str = "./corpus",
        cache: bool = False
    ) -> DocExtracter:
        """
        Create a document extracter.
        
        Args:
            corpus_name: Name of corpus configuration
            db_dir: Database directory path
            cache: Whether to enable full document caching
            
        Returns:
            DocExtracter instance
            
        Example:
            >>> extracter = ComponentFactory.create_doc_extracter("Textbooks", cache=True)
            >>> docs = extracter.extract(["doc_id_1", "doc_id_2"])
        """
        return DocExtracter(
            db_dir=db_dir,
            cache=cache,
            corpus_name=corpus_name
        )
    
    @staticmethod
    def create_batch_retriever(
        retriever_types: list,
        corpus_names: list,
        db_dir: str = "./corpus",
        **kwargs
    ) -> Dict[str, Dict[str, BaseRetriever]]:
        """
        Create multiple retrievers in batch.
        
        Args:
            retriever_types: List of retriever types to create
            corpus_names: List of corpus names to use
            db_dir: Database directory path
            **kwargs: Additional arguments for retrievers
            
        Returns:
            Nested dictionary of retrievers [retriever_type][corpus_name]
            
        Example:
            >>> retrievers = ComponentFactory.create_batch_retriever(
            ...     ["bm25", "medcpt"], 
            ...     ["textbooks", "pubmed"]
            ... )
            >>> bm25_textbooks = retrievers["bm25"]["textbooks"]
        """
        retrievers = {}
        
        for retriever_type in retriever_types:
            retrievers[retriever_type] = {}
            
            for corpus_name in corpus_names:
                try:
                    retriever = RetrieverFactory.create_retriever(
                        retriever_type=retriever_type,
                        corpus_name=corpus_name,
                        db_dir=db_dir,
                        **kwargs
                    )
                    retrievers[retriever_type][corpus_name] = retriever
                except Exception as e:
                    print(f"Failed to create {retriever_type} retriever for {corpus_name}: {e}")
                    retrievers[retriever_type][corpus_name] = None
        
        return retrievers