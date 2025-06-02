"""Configuration for RAG system components."""

from typing import Dict, Any, List

# LLM Provider configurations
LLM_PROVIDERS = {
    "openai": {
        "model_name": "gpt-3.5-turbo-16k",
        "max_length": 16384,
        "context_length": 12000,
        "default_temperature": 0.3,
        "default_max_tokens": 1024,
        "api_key_env": "OPENAI_API_KEY"
    },
    "gemini": {
        "model_name": "gemini-2.0-flash",
        "max_length": 32768,
        "context_length": 25000,
        "default_temperature": 0.3,
        "default_max_tokens": 1024,
        "api_key_env": "GOOGLE_API_KEY"
    },
    "gemma3": {
        "model_name": "gemma-3-12b-it",
        "max_length": 32768,
        "context_length": 25000,
        "default_temperature": 0.3,
        "default_max_tokens": 1024,
        "api_key_env": "GOOGLE_API_KEY"
    },
    "local": {
        "default_model": "google/gemma-3-12b-it-qat-q4_0-gguf",
        "max_length": 8192,
        "context_length": 5120,
        "default_temperature": 0.3,
        "default_max_tokens": 1024,
        "hf_token_env": "HUGGINGFACE_TOKEN"
    }
}

# Model-specific configurations for local models
LOCAL_MODEL_CONFIGS = {
    "gemma-3": {
        "repo_id": "google/gemma-3-12b-it-qat-q4_0-gguf",
        "filename": "gemma-3-12b-it-q4_0.gguf",
        "tokenizer": "google/gemma-3-12b-pt",
        "max_length": 8192,
        "context_length": 5120
    }
}

# RAG system defaults
RAG_DEFAULTS = {
    "retrieval_k": 3,
    "rrf_k": 60,
    "use_rag": True,
    "corpus_cache": False,
    "retriever_name": "SPLADE",
    "corpus_name": "MedCorp"
}

# Prompt generation settings
PROMPT_SETTINGS = {
    "system_prompt": "You are a medical expert. Answer the question strictly in JSON format with no additional text.",
    "stop_words": ["###", "User:", "\n\n\n"],
    "max_context_snippets": 5,
    "snippet_max_length": 1024
}

# JSON parsing settings
JSON_PARSING = {
    "valid_answer_keys": ["answer", "answer_choice"],
    "max_attempts": 3,
    "fallback_response": {"answer": "Unable to parse response"}
}

# Generation parameters
GENERATION_PARAMS = {
    "temperature": 0.3,
    "max_tokens": 1024,
    "do_sample": False,
    "truncation": True
}