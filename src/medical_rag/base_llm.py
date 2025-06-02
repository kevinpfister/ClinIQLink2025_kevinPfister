"""Base LLM interface and common functionality."""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

from .rag_config import LLM_PROVIDERS, GENERATION_PARAMS


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = GENERATION_PARAMS["temperature"],
        max_tokens: int = GENERATION_PARAMS["max_tokens"],
        **kwargs
    ):
        """
        Initialize base LLM.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the service
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_type = self._get_provider_type()
        
        # Get provider configuration
        self.config = LLM_PROVIDERS.get(self.provider_type, {})
        
        # Set defaults from config if not provided
        if not self.model_name:
            self.model_name = self.config.get("model_name")
            
        self.max_length = self.config.get("max_length", 2048)
        self.context_length = self.config.get("context_length", 1024)
        
        # Initialize the provider-specific client
        self._initialize_client(**kwargs)
    
    @abstractmethod
    def _get_provider_type(self) -> str:
        """Return the provider type identifier."""
        pass
    
    @abstractmethod
    def _initialize_client(self, **kwargs) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate response from messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        pass
    
    def _get_api_key(self, env_var: str) -> str:
        """
        Get API key from environment or instance variable.
        
        Args:
            env_var: Environment variable name
            
        Returns:
            API key string
            
        Raises:
            ValueError: If API key is not found
        """
        api_key = self.api_key or os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key not provided. Set {env_var} environment variable "
                f"or pass api_key parameter."
            )
        return api_key
    
    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Validate message format.
        
        Args:
            messages: List of message dictionaries
            
        Raises:
            ValueError: If messages format is invalid
        """
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")
        
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' keys")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get LLM information.
        
        Returns:
            Dictionary with LLM information
        """
        return {
            "provider": self.provider_type,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "context_length": self.context_length,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def __repr__(self) -> str:
        """String representation of the LLM."""
        return f"{self.__class__.__name__}(model={self.model_name})"


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class APIError(LLMError):
    """Exception for API-related errors."""
    pass


class ConfigurationError(LLMError):
    """Exception for configuration-related errors."""
    pass


class GenerationError(LLMError):
    """Exception for generation-related errors."""
    pass