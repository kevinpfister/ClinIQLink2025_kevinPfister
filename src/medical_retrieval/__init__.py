"""
Medical Retrieval System Package

A modular retrieval system for medical documents supporting multiple 
retrieval methods (BM25, SPLADE, UniCOIL, MedCPT) and corpus configurations.
"""

__version__ = "1.0.0"
__author__ = "Zakaria Omarar"

# Import main classes and functions
from .retrieval_system import RetrievalSystem
from .doc_extracter import DocExtracter

# Import individual retrievers
from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from .splade_retriever import SPLADERetriever
from .unicoil_retriever import UniCOILRetriever
from .medcpt_retriever import MedCPTRetriever

# Import factory classes
from .factory import RetrieverFactory, SystemFactory, ComponentFactory

# Import configuration and utilities
from .config import CORPUS_NAMES, RETRIEVER_NAMES, MODEL_CONFIGS, DEFAULTS
from .utils import concat, batch_index_files, create_minimal_index

# Define what gets imported with "from medical_retrieval import *"
__all__ = [
    # Main classes
    "RetrievalSystem",
    "DocExtracter",
    
    # Retriever classes
    "BaseRetriever",
    "BM25Retriever", 
    "SPLADERetriever",
    "UniCOILRetriever",
    "MedCPTRetriever",
    
    # Factory classes
    "RetrieverFactory",
    "SystemFactory", 
    "ComponentFactory",
    
    # Configuration
    "CORPUS_NAMES",
    "RETRIEVER_NAMES", 
    "MODEL_CONFIGS",
    "DEFAULTS",
    
    # Utility functions
    "concat",
    "batch_index_files",
    "create_minimal_index",
    
    # Factory functions
    "create_retrieval_system",
    "list_available_configurations",
    "validate_dependencies",
]


def create_retrieval_system(
    retriever_name: str = "BM25",
    corpus_name: str = "Textbooks", 
    db_dir: str = "./corpus",
    cache: bool = False,
    **kwargs
) -> RetrievalSystem:
    """
    Factory function to create a retrieval system with validated parameters.
    
    Args:
        retriever_name: Name of retriever configuration 
        corpus_name: Name of corpus configuration
        db_dir: Database directory path
        cache: Whether to enable document caching
        **kwargs: Additional arguments
        
    Returns:
        Configured RetrievalSystem instance
        
    Raises:
        ValueError: If retriever_name or corpus_name is invalid
        
    Example:
        >>> system = create_retrieval_system("BM25", "Textbooks")
        >>> docs, scores = system.retrieve("What is diabetes?")
    """
    if retriever_name not in RETRIEVER_NAMES:
        available = list(RETRIEVER_NAMES.keys())
        raise ValueError(f"Invalid retriever_name '{retriever_name}'. Available: {available}")
    
    if corpus_name not in CORPUS_NAMES:
        available = list(CORPUS_NAMES.keys())
        raise ValueError(f"Invalid corpus_name '{corpus_name}'. Available: {available}")
    
    return RetrievalSystem(
        retriever_name=retriever_name,
        corpus_name=corpus_name,
        db_dir=db_dir,
        cache=cache,
        **kwargs
    )


def list_available_configurations() -> dict:
    """
    Get all available retriever and corpus configurations.
    
    Returns:
        Dictionary with available configurations
        
    Example:
        >>> configs = list_available_configurations()
        >>> print("Available retrievers:", list(configs['retrievers'].keys()))
        >>> print("Available corpora:", list(configs['corpora'].keys()))
    """
    return {
        "retrievers": RETRIEVER_NAMES,
        "corpora": CORPUS_NAMES,
        "model_configs": MODEL_CONFIGS,
        "defaults": DEFAULTS
    }


def validate_dependencies() -> dict:
    """
    Check if required dependencies are available.
    
    Returns:
        Dictionary with dependency status
        
    Example:
        >>> deps = validate_dependencies()
        >>> if not deps['all_available']:
        ...     print("Missing dependencies:", deps['missing'])
    """
    dependencies = {
        'pyserini': False,
        'transformers': False, 
        'torch': False,
        'numpy': False,
        'tqdm': False
    }
    
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            missing.append(dep)
    
    return {
        'dependencies': dependencies,
        'missing': missing,
        'all_available': len(missing) == 0
    }