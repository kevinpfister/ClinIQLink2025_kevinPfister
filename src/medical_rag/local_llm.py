"""Local LLM implementation using llama-cpp and HuggingFace."""

import os
from typing import List, Dict, Any, Optional, Union

from .base_llm import BaseLLM, APIError, ConfigurationError
from .rag_config import LOCAL_MODEL_CONFIGS


class LocalLLM(BaseLLM):
    """Local LLM implementation using llama-cpp and HuggingFace."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize local LLM.
        
        Args:
            model_name: Name/path of the model
            cache_dir: Directory for caching models
            hf_token: HuggingFace token for model access
            n_gpu_layers: Number of GPU layers (-1 for all)
            n_batch: Batch size for GPU decoding
            verbose: Whether to print verbose output
            **kwargs: Additional arguments
        """
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.verbose = verbose
        
        super().__init__(model_name=model_name, **kwargs)
    
    def _get_provider_type(self) -> str:
        """Return the provider type identifier."""
        return "local"
    
    def _initialize_client(self, **kwargs) -> None:
        """Initialize local model client."""
        # Check for required packages
        self._check_dependencies()
        
        # Get HuggingFace token if needed
        if not self.hf_token:
            self.hf_token = os.getenv(self.config["hf_token_env"])
        
        # Set up HuggingFace authentication
        if self.hf_token:
            self._setup_hf_auth()
        
        # Download and load model
        model_path = self._get_or_download_model()
        self._load_model(model_path)
        self._load_tokenizer()
        
        print(f"Initialized local LLM: {self.model_name}")
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ConfigurationError(
                "llama-cpp-python not installed. "
                "Install with 'pip install llama-cpp-python'"
            )
        
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ConfigurationError(
                "transformers not installed. "
                "Install with 'pip install transformers'"
            )
        
        try:
            from huggingface_hub import (
                HfFolder, hf_hub_download, login, try_to_load_from_cache
            )
        except ImportError:
            raise ConfigurationError(
                "huggingface_hub not installed. "
                "Install with 'pip install huggingface_hub'"
            )
    
    def _setup_hf_auth(self) -> None:
        """Set up HuggingFace authentication."""
        from huggingface_hub import HfFolder, login
        
        try:
            login(self.hf_token)
            HfFolder.save_token(self.hf_token)
        except Exception as e:
            print(f"Warning: HuggingFace authentication failed: {e}")
    
    def _get_or_download_model(self) -> str:
        """Get model path, downloading if necessary."""
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        
        # Handle different model specifications
        if os.path.exists(self.model_name):
            print(f"Using local model file: {self.model_name}")
            return self.model_name
        
        # Get model configuration
        model_config = self._get_model_config()
        repo_id = model_config.get("repo_id", self.model_name)
        filename = model_config.get("filename")
        
        if not filename:
            # Try to infer filename
            filename = self._infer_filename(repo_id)
        
        print(f"Attempting to download {filename} from {repo_id}")
        
        try:
            # Check if model exists in cache
            local_path = try_to_load_from_cache(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir
            )
            
            # If not found in cache, download it
            if local_path is None:
                print(f"Model {filename} not found in cache. Downloading...")
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=self.hf_token,
                    cache_dir=self.cache_dir
                )
                print(f"Model downloaded to {local_path}")
            else:
                print(f"Using cached model from {local_path}")
            
            return local_path
            
        except Exception as e:
            raise ConfigurationError(f"Failed to download model: {e}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get configuration for the model."""
        # Check if model name matches known configurations
        for key, config in LOCAL_MODEL_CONFIGS.items():
            if key in self.model_name.lower():
                return config
        
        # Default configuration
        return {
            "repo_id": self.model_name,
            "filename": None
        }
    
    def _infer_filename(self, repo_id: str) -> str:
        """Infer GGUF filename from repo ID."""
        # Try common patterns
        base_name = repo_id.split("/")[-1]
        if not base_name.endswith(".gguf"):
            base_name = f"{base_name}.gguf"
        return base_name
    
    def _load_model(self, model_path: str) -> None:
        """Load the llama-cpp model."""
        from llama_cpp import Llama
        
        try:
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=self.n_batch,
                n_ctx=self.max_length,
                verbose=self.verbose,
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to load model: {e}")
    
    def _load_tokenizer(self) -> None:
        """Load the HuggingFace tokenizer."""
        from transformers import AutoTokenizer
        
        # Get tokenizer name from model config
        model_config = self._get_model_config()
        tokenizer_name = model_config.get("tokenizer")
        
        if not tokenizer_name:
            # Try to infer tokenizer name
            if "gemma" in self.model_name.lower():
                tokenizer_name = "google/gemma-3-12b-pt"
            else:
                tokenizer_name = self.model_name.split("/")[0] + "/" + self.model_name.split("/")[1]
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, 
                cache_dir=self.cache_dir
            )
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response using local model.
        
        Args:
            messages: List of message dictionaries
            temperature: Generation temperature (overrides instance default)
            max_tokens: Maximum tokens to generate (overrides instance default)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
            
        Raises:
            APIError: If generation fails
        """
        self._validate_messages(messages)
        
        # Use provided parameters or instance defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Use llama-cpp chat completion
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs
            )
            
            # Parse response
            return self._parse_llama_response(response)
            
        except Exception as e:
            raise APIError(f"Local model generation failed: {e}")
    
    def _parse_llama_response(self, response: Any) -> str:
        """
        Parse llama-cpp response.
        
        Args:
            response: Raw response from llama-cpp
            
        Returns:
            Parsed response text
        """
        if not isinstance(response, dict):
            raise APIError(f"Unexpected response type: {type(response)}")
        
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise APIError(f"Empty or missing choices: {response}")
        
        first_choice = choices[0] or {}
        
        # Look for content in message or text field
        content = (
            (first_choice.get("message") or {}).get("content") or
            first_choice.get("text") or
            ""
        )
        
        return content.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.
        
        Returns:
            Dictionary with model information
        """
        info = self.get_info()
        info.update({
            "model_path": getattr(self.model, "model_path", "Unknown"),
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "tokenizer": getattr(self.tokenizer, "name_or_path", "Unknown") if self.tokenizer else None,
            "cache_dir": self.cache_dir
        })
        return info