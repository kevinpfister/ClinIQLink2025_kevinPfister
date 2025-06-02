"""OpenAI LLM implementation."""

from typing import List, Dict, Any, Optional

from .base_llm import BaseLLM, APIError, ConfigurationError


class OpenAILLM(BaseLLM):
    """OpenAI GPT model implementation."""
    
    def _get_provider_type(self) -> str:
        """Return the provider type identifier."""
        return "openai"
    
    def _initialize_client(self, **kwargs) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ConfigurationError(
                "OpenAI package not installed. Install with 'pip install openai'"
            )
        
        # Get API key
        api_key = self._get_api_key(self.config["api_key_env"])
        
        # Initialize client
        try:
            self.client = OpenAI(api_key=api_key)
            print(f"Initialized OpenAI client for {self.model_name}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            temperature: Generation temperature (overrides instance default)
            max_tokens: Maximum tokens to generate (overrides instance default)
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Generated response text
            
        Raises:
            APIError: If API call fails
        """
        self._validate_messages(messages)
        
        # Use provided parameters or instance defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs
            )
            
            if (hasattr(response, 'choices') and 
                response.choices and 
                len(response.choices) > 0):
                return response.choices[0].message.content.strip()
            else:
                raise APIError(f"Unexpected response format: {response}")
                
        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"OpenAI API call failed: {e}")
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get usage information (if available from last request).
        
        Returns:
            Dictionary with usage information
        """
        # This could be extended to track usage across requests
        return {
            "provider": "openai",
            "model": self.model_name,
            "pricing_info": "Check OpenAI pricing page for current rates"
        }