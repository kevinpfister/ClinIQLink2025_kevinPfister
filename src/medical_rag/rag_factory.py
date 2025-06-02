"""Factory for creating RAG systems with different configurations."""

import os
from typing import Dict, Any, Optional

from .rag_system import RAGSystem
from .rag_config import LLM_PROVIDERS, RAG_DEFAULTS
from .base_llm import ConfigurationError


class RAGFactory:
    """Factory class for creating RAG systems."""
    
    @classmethod
    def create_rag_system(
        cls,
        llm_provider: str = "local",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        use_rag: bool = True,
        retriever_name: str = "SPLADE",
        corpus_name: str = "MedCorp",
        **kwargs
    ) -> RAGSystem:
        """
        Create a RAG system with the specified configuration.
        
        Args:
            llm_provider: LLM provider ("openai", "gemini", "gemma3", "local")
            model_name: Name of the model to use
            api_key: API key for the service
            use_rag: Whether to enable RAG functionality
            retriever_name: Name of retriever configuration
            corpus_name: Name of corpus configuration
            **kwargs: Additional arguments
            
        Returns:
            Configured RAG system
            
        Example:
            >>> rag = RAGFactory.create_rag_system("openai", "gpt-3.5-turbo")
            >>> answer, docs, scores = rag.answer({"question": "What is diabetes?"})
        """
        # Prepare LLM configuration
        llm_config = {
            "model_name": model_name,
            "api_key": api_key
        }
        
        # Add provider-specific configurations
        if llm_provider in LLM_PROVIDERS:
            provider_config = LLM_PROVIDERS[llm_provider]
            if not model_name:
                llm_config["model_name"] = provider_config.get("model_name")
        
        return RAGSystem(
            llm_provider=llm_provider,
            llm_config=llm_config,
            use_rag=use_rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            **kwargs
        )
    
    @classmethod
    def create_openai_rag(
        cls,
        model_name: str = "gpt-3.5-turbo-16k",
        api_key: Optional[str] = None,
        retriever_name: str = "MedicalHybrid",
        corpus_name: str = "MedCorp",
        **kwargs
    ) -> RAGSystem:
        """
        Create a RAG system with OpenAI LLM.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            retriever_name: Retriever configuration
            corpus_name: Corpus configuration
            **kwargs: Additional arguments
            
        Returns:
            RAG system with OpenAI LLM
        """
        return cls.create_rag_system(
            llm_provider="openai",
            model_name=model_name,
            api_key=api_key,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            **kwargs
        )
    
    @classmethod
    def create_gemini_rag(
        cls,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        retriever_name: str = "MedicalHybrid",
        corpus_name: str = "MedCorp",
        **kwargs
    ) -> RAGSystem:
        """
        Create a RAG system with Google Gemini LLM.
        
        Args:
            model_name: Gemini model name
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            retriever_name: Retriever configuration
            corpus_name: Corpus configuration
            **kwargs: Additional arguments
            
        Returns:
            RAG system with Gemini LLM
        """
        return cls.create_rag_system(
            llm_provider="gemini",
            model_name=model_name,
            api_key=api_key,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            **kwargs
        )
    
    @classmethod
    def create_local_rag(
        cls,
        model_name: str = "google/gemma-3-12b-it-qat-q4_0-gguf",
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        retriever_name: str = "SPLADE",
        corpus_name: str = "MedCorp",
        **kwargs
    ) -> RAGSystem:
        """
        Create a RAG system with local LLM.
        
        Args:
            model_name: Local model name/path
            hf_token: HuggingFace token (or set HUGGINGFACE_TOKEN env var)
            cache_dir: Directory for caching models
            retriever_name: Retriever configuration
            corpus_name: Corpus configuration
            **kwargs: Additional arguments
            
        Returns:
            RAG system with local LLM
        """
        llm_config = {
            "model_name": model_name,
            "hf_token": hf_token,
            "cache_dir": cache_dir
        }
        
        return RAGSystem(
            llm_provider="local",
            llm_config=llm_config,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            **kwargs
        )
    
    @classmethod
    def create_no_rag_system(
        cls,
        llm_provider: str = "local",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> RAGSystem:
        """
        Create a system without RAG (direct LLM only).
        
        Args:
            llm_provider: LLM provider
            model_name: Model name
            api_key: API key if needed
            **kwargs: Additional arguments
            
        Returns:
            RAG system with RAG disabled
        """
        return cls.create_rag_system(
            llm_provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            use_rag=False,
            **kwargs
        )
    
    @classmethod
    def create_from_env(
        cls,
        env_prefix: str = "RAG_",
        **kwargs
    ) -> RAGSystem:
        """
        Create RAG system from environment variables.
        
        Args:
            env_prefix: Prefix for environment variables
            **kwargs: Override arguments
            
        Returns:
            RAG system configured from environment
            
        Environment Variables:
            RAG_LLM_PROVIDER: LLM provider type
            RAG_MODEL_NAME: Model name
            RAG_API_KEY: API key
            RAG_USE_RAG: Whether to enable RAG (true/false)
            RAG_RETRIEVER_NAME: Retriever configuration
            RAG_CORPUS_NAME: Corpus configuration
        """
        config = {}
        
        # Map environment variables to config
        env_mappings = {
            f"{env_prefix}LLM_PROVIDER": "llm_provider",
            f"{env_prefix}MODEL_NAME": "model_name", 
            f"{env_prefix}API_KEY": "api_key",
            f"{env_prefix}USE_RAG": "use_rag",
            f"{env_prefix}RETRIEVER_NAME": "retriever_name",
            f"{env_prefix}CORPUS_NAME": "corpus_name",
            f"{env_prefix}DB_DIR": "db_dir",
            f"{env_prefix}CACHE_DIR": "cache_dir"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle boolean values
                if config_key == "use_rag":
                    config[config_key] = value.lower() in ("true", "1", "yes")
                else:
                    config[config_key] = value
        
        # Override with provided kwargs
        config.update(kwargs)
        
        return cls.create_rag_system(**config)
    
    @classmethod
    def get_recommended_configs(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get recommended RAG system configurations.
        
        Returns:
            Dictionary of recommended configurations
        """
        return {
            "medical_research": {
                "description": "Best for medical research questions",
                "llm_provider": "TBD",
                "model_name": "TBD",
                "retriever_name": "TBD",
                "corpus_name": "TBD",
                "use_rag": True
            },
            "TBD1": {
                "description": "TBD",
                "llm_provider": "TBD",
                "model_name": "TBD",
                "retriever_name": "TBD",
                "corpus_name": "TBD",
                "use_rag": True
            },
            "TBD2": {
                "description": "TBD",
                "llm_provider": "local",
                "model_name": "TBD",
                "retriever_name": "TBD",
                "corpus_name": "TBD",
                "use_rag": True
            },
            "TBD3": {
                "description": "TBD",
                "llm_provider": "TBD",
                "model_name": "TBD",
                "retriever_name": "TBD",
                "corpus_name": "TBD",
                "use_rag": True
            },
            "TBD4": {
                "description": "TBD",
                "llm_provider": "TBD",
                "model_name": "TBD",
                "use_rag": False
            }
        }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate RAG system configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Check required fields
        if "llm_provider" not in config:
            raise ConfigurationError("llm_provider is required")
        
        provider = config["llm_provider"]
        if provider not in LLM_PROVIDERS:
            available = list(LLM_PROVIDERS.keys())
            raise ConfigurationError(f"Invalid llm_provider '{provider}'. Available: {available}")
        
        # Provider-specific validation
        provider_config = LLM_PROVIDERS[provider]
        api_key_env = provider_config.get("api_key_env")
        
        if api_key_env and not config.get("api_key") and not os.getenv(api_key_env):
            raise ConfigurationError(f"API key required for {provider}. Set {api_key_env} or provide api_key.")
        
        return True


def create_rag_system(**kwargs) -> RAGSystem:
    """
    Convenience function to create a RAG system.
    
    Args:
        **kwargs: Configuration arguments
        
    Returns:
        RAG system instance
    """
    return RAGFactory.create_rag_system(**kwargs)