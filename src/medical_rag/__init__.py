"""
Medical RAG System Package

A modular RAG (Retrieval-Augmented Generation) system for medical question answering
that supports multiple LLM providers and retrieval methods.
"""

__version__ = "1.0.0"
__author__ = "Zakaria Omarar"

# Import main classes
from .rag_system import RAGSystem
from .rag_factory import RAGFactory, create_rag_system

# Import LLM components
from .base_llm import BaseLLM, LLMError, APIError, ConfigurationError, GenerationError
from .llm_factory import LLMFactory, create_llm
from .openai_llm import OpenAILLM
from .gemini_llm import GeminiLLM, Gemma3LLM
from .local_llm import LocalLLM

# Import utilities
from .rag_utils import (
    extract_valid_json,
    safe_json_parse,
    process_llm_response,
    format_context_snippets,
    create_direct_prompt
)

# Import configuration
from .rag_config import (
    LLM_PROVIDERS,
    RAG_DEFAULTS,
    PROMPT_SETTINGS,
    JSON_PARSING,
    GENERATION_PARAMS
)

# Define public API
__all__ = [
    # Main classes
    "RAGSystem",
    "RAGFactory",
    
    # LLM classes
    "BaseLLM",
    "OpenAILLM",
    "GeminiLLM", 
    "Gemma3LLM",
    "LocalLLM",
    "LLMFactory",
    
    # Exceptions
    "LLMError",
    "APIError",
    "ConfigurationError",
    "GenerationError",
    
    # Utility functions
    "extract_valid_json",
    "safe_json_parse", 
    "process_llm_response",
    "format_context_snippets",
    "create_direct_prompt",
    
    # Factory functions
    "create_rag_system",
    "create_llm",
    
    # Configuration
    "LLM_PROVIDERS",
    "RAG_DEFAULTS",
    "PROMPT_SETTINGS",
    "JSON_PARSING",
    "GENERATION_PARAMS"
]


def create_medical_rag(
    llm_provider: str = "local",
    model_name: str = None,
    use_rag: bool = True,
    **kwargs
) -> RAGSystem:
    """
    Factory function to create a medical RAG system with sensible defaults.
    
    Args:
        llm_provider: LLM provider ("openai", "gemini", "gemma3", "local")
        model_name: Model name (uses provider default if None)
        use_rag: Whether to enable RAG functionality
        **kwargs: Additional configuration arguments
        
    Returns:
        Configured RAG system for medical applications
        
    Example:
        >>> rag = create_medical_rag("openai", use_rag=True)
        >>> answer, docs, scores = rag.answer({
        ...     "question": "What are the symptoms of diabetes?",
        ...     "type": "multiple_choice",
        ...     "options": {"A": "Thirst", "B": "Fatigue", "C": "Both", "D": "Neither"}
        ... })
    """
    # Set medical-specific defaults
    defaults = {
        "retriever_name": "BM25",
        "corpus_name": "MedCorp",
        "corpus_cache": False,
        "use_rag": use_rag
    }
    
    # Override with user-provided kwargs
    config = {**defaults, **kwargs}
    
    return RAGFactory.create_rag_system(
        llm_provider=llm_provider,
        model_name=model_name,
        **config
    )


def list_available_llm_providers() -> list:
    """
    Get list of available LLM providers.
    
    Returns:
        List of provider names
    """
    return LLMFactory.get_available_providers()


def get_recommended_configurations() -> dict:
    """
    Get recommended RAG system configurations for different use cases.
    
    Returns:
        Dictionary of recommended configurations
    """
    return RAGFactory.get_recommended_configs()


def validate_environment() -> dict:
    """
    Check environment for required dependencies and API keys.
    
    Returns:
        Dictionary with validation results
    """
    import os
    
    results = {
        "dependencies": {},
        "api_keys": {},
        "warnings": [],
        "all_good": True
    }
    
    # Check dependencies
    deps_to_check = [
        ("openai", "OpenAI API support"),
        ("google.generativeai", "Google Gemini API support"),
        ("llama_cpp", "Local model support"),
        ("transformers", "HuggingFace transformers"),
        ("medical_retrieval", "Medical retrieval system")
    ]
    
    for dep, description in deps_to_check:
        try:
            __import__(dep)
            results["dependencies"][dep] = {"available": True, "description": description}
        except ImportError:
            results["dependencies"][dep] = {"available": False, "description": description}
            results["warnings"].append(f"Optional dependency '{dep}' not available: {description}")
    
    # Check API keys
    api_keys_to_check = [
        ("OPENAI_API_KEY", "OpenAI API access"),
        ("GOOGLE_API_KEY", "Google Gemini/Gemma API access"),
        ("HUGGINGFACE_TOKEN", "HuggingFace model access")
    ]
    
    for env_var, description in api_keys_to_check:
        value = os.getenv(env_var)
        results["api_keys"][env_var] = {
            "set": bool(value),
            "description": description
        }
        
        if not value:
            results["warnings"].append(f"API key '{env_var}' not set: {description}")
    
    # Overall status
    core_deps_available = all(
        results["dependencies"].get(dep, {}).get("available", False)
        for dep in ["medical_retrieval"]
    )
    
    results["all_good"] = core_deps_available and len(results["warnings"]) == 0
    
    return results


# Convenience aliases for backward compatibility
RAG = RAGSystem  # For compatibility with original RAG class