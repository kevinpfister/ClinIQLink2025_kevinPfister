"""Factory for creating LLM instances."""

from typing import Dict, Type, Any, Optional

from .base_llm import BaseLLM, ConfigurationError
from .openai_llm import OpenAILLM
from .gemini_llm import GeminiLLM, Gemma3LLM
from .local_llm import LocalLLM
from .rag_config import LLM_PROVIDERS


class LLMFactory:
    """Factory class for creating LLM instances."""
    
    _llm_classes: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAILLM,
        "gemini": GeminiLLM,
        "gemma3": Gemma3LLM,
        "local": LocalLLM
    }
    
    @classmethod
    def create_llm(
        cls,
        provider: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance of the specified provider.
        
        Args:
            provider: LLM provider type ("openai", "gemini", "gemma3", "local")
            model_name: Name of the model to use
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLM instance
            
        Raises:
            ConfigurationError: If provider is not supported
            
        Example:
            >>> factory = LLMFactory()
            >>> llm = factory.create_llm("openai", model_name="gpt-3.5-turbo")
            >>> response = llm.generate([{"role": "user", "content": "Hello!"}])
        """
        provider = provider.lower()
        
        if provider not in cls._llm_classes:
            available = list(cls._llm_classes.keys())
            raise ConfigurationError(
                f"Unsupported LLM provider '{provider}'. Available: {available}"
            )
        
        llm_class = cls._llm_classes[provider]
        return llm_class(model_name=model_name, **kwargs)
    
    @classmethod
    def create_from_config(
        cls,
        provider: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance from configuration.
        
        Args:
            provider: LLM provider type
            config: Configuration dictionary (uses defaults if None)
            **kwargs: Additional arguments that override config
            
        Returns:
            LLM instance
        """
        provider = provider.lower()
        
        # Get default configuration
        default_config = LLM_PROVIDERS.get(provider, {})
        
        # Merge configurations
        final_config = default_config.copy()
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        return cls.create_llm(provider, **final_config)
    
    @classmethod 
    def create_auto(
        cls,
        model_name: str,
        **kwargs
    ) -> BaseLLM:
        """
        Automatically detect provider from model name and create LLM.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional arguments
            
        Returns:
            LLM instance
            
        Raises:
            ConfigurationError: If provider cannot be detected
        """
        provider = cls._detect_provider(model_name)
        return cls.create_llm(provider, model_name=model_name, **kwargs)
    
    @classmethod
    def _detect_provider(cls, model_name: str) -> str:
        """
        Detect provider from model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Provider type
        """
        model_lower = model_name.lower()
        
        # Check for specific patterns
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "gemini" in model_lower:
            return "gemini"
        elif "gemma-3" in model_lower or "gemma3" in model_lower:
            return "gemma3"
        elif "gguf" in model_lower or model_name.startswith("/") or model_name.endswith(".gguf"):
            return "local"
        elif any(x in model_lower for x in ["huggingface", "hf", "transformers"]):
            return "local"
        else:
            # Default to local for unknown models
            return "local"
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available LLM providers.
        
        Returns:
            List of provider names
        """
        return list(cls._llm_classes.keys())
    
    @classmethod
    def get_provider_info(cls, provider: str) -> Dict[str, Any]:
        """
        Get information about a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with provider information
            
        Raises:
            ConfigurationError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._llm_classes:
            raise ConfigurationError(f"Unsupported provider '{provider}'")
        
        llm_class = cls._llm_classes[provider]
        config = LLM_PROVIDERS.get(provider, {})
        
        return {
            "provider": provider,
            "class": llm_class.__name__,
            "description": llm_class.__doc__ or "No description available",
            "default_config": config,
            "module": llm_class.__module__
        }
    
    @classmethod
    def validate_provider_config(cls, provider: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a provider.
        
        Args:
            provider: Provider name
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        provider = provider.lower()
        
        if provider not in cls._llm_classes:
            raise ConfigurationError(f"Unsupported provider '{provider}'")
        
        # Basic validation - could be extended
        required_fields = {
            "openai": ["api_key"],
            "gemini": ["api_key"],
            "gemma3": ["api_key"],
            "local": []  # Local models have fewer requirements
        }
        
        required = required_fields.get(provider, [])
        
        for field in required:
            if field not in config or not config[field]:
                raise ConfigurationError(f"Missing required field '{field}' for provider '{provider}'")
        
        return True


def create_llm(provider: str, **kwargs) -> BaseLLM:
    """
    Convenience function to create an LLM instance.
    
    Args:
        provider: LLM provider type
        **kwargs: Additional arguments
        
    Returns:
        LLM instance
    """
    return LLMFactory.create_llm(provider, **kwargs)