"""Google Gemini LLM implementations."""

from typing import List, Dict, Any, Optional

from .base_llm import BaseLLM, APIError, ConfigurationError


class GeminiLLM(BaseLLM):
    """Google Gemini model implementation."""
    
    def _get_provider_type(self) -> str:
        """Return the provider type identifier."""
        return "gemini"
    
    def _initialize_client(self, **kwargs) -> None:
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ConfigurationError(
                "Google Generative AI package not installed. "
                "Install with 'pip install google-generativeai'"
            )
        
        # Get API key
        api_key = self._get_api_key(self.config["api_key_env"])
        
        # Configure client
        try:
            genai.configure(api_key=api_key)
            self.genai = genai
            print(f"Initialized Google Generative AI client for {self.model_name}")
        except Exception as e:
            raise ConfigurationError(f"Failed to configure Google Generative AI: {e}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Gemini API.
        
        Args:
            messages: List of message dictionaries
            temperature: Generation temperature (overrides instance default)
            max_tokens: Maximum tokens to generate (overrides instance default)
            **kwargs: Additional Gemini-specific parameters
            
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
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            # Create model and generate response
            model = self.genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                gemini_messages,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": max_tok,
                    **kwargs
                }
            )
            
            if response and hasattr(response, "text"):
                return response.text.strip()
            else:
                raise APIError(f"Unexpected response format: {response}")
                
        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"Gemini API call failed: {e}")
    
    def _convert_messages_to_gemini_format(
        self, 
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Convert standard message format to Gemini format.
        
        Args:
            messages: Standard message format
            
        Returns:
            Gemini-formatted messages
        """
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        return gemini_messages


class Gemma3LLM(GeminiLLM):
    """Google Gemma 3 model implementation (via Gemini API)."""
    
    def _get_provider_type(self) -> str:
        """Return the provider type identifier."""
        return "gemma3"
    
    def _initialize_client(self, **kwargs) -> None:
        """Initialize Google Generative AI client for Gemma 3."""
        super()._initialize_client(**kwargs)
        print(f"Initialized Google Generative AI client for Gemma 3 ({self.model_name})")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Gemma 3 specific model information.
        
        Returns:
            Dictionary with model information
        """
        info = self.get_info()
        info.update({
            "architecture": "Gemma 3",
            "size": "12B parameters",
            "capabilities": ["instruction-following", "medical-qa", "reasoning"]
        })
        return info