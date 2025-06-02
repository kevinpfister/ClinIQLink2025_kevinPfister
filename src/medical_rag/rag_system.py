"""Main RAG system that orchestrates LLM and retrieval components."""

import os
from typing import Dict, Any, List, Optional, Tuple, Union

from medical_retrieval import create_retrieval_system
from .llm_factory import LLMFactory
from .base_llm import BaseLLM
from .rag_config import RAG_DEFAULTS, PROMPT_SETTINGS
from .rag_utils import (
    process_llm_response,
    format_context_snippets,
    build_messages,
    create_direct_prompt,
    prepare_question_data,
    save_rag_artifacts,
    build_retrieval_query,
    validate_question_data,
    truncate_context
)


class RAGSystem:
    """
    Main RAG system that combines retrieval and generation.
    
    This class orchestrates the interaction between document retrieval
    and language model generation to provide RAG-based question answering.
    """
    
    def __init__(
        self,
        llm_provider: str = "local",
        llm_config: Optional[Dict[str, Any]] = None,
        use_rag: bool = RAG_DEFAULTS["use_rag"],
        retriever_name: str = RAG_DEFAULTS["retriever_name"],
        corpus_name: str = RAG_DEFAULTS["corpus_name"],
        db_dir: str = "./corpus",
        corpus_cache: bool = RAG_DEFAULTS["corpus_cache"],
        templates: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize RAG system.
        
        Args:
            llm_provider: LLM provider type ("openai", "gemini", "gemma3", "local")
            llm_config: Configuration for LLM (model name, API keys, etc.)
            use_rag: Whether to use retrieval-augmented generation
            retriever_name: Name of retriever configuration
            corpus_name: Name of corpus configuration
            db_dir: Database directory for retrieval system
            corpus_cache: Whether to cache corpus documents
            templates: Prompt templates dictionary
            **kwargs: Additional arguments
        """
        self.use_rag = use_rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.templates = templates
        
        # Initialize LLM
        self.llm = self._initialize_llm(llm_provider, llm_config or {})
        
        # Initialize retrieval system if RAG is enabled
        self.retrieval_system = None
        if self.use_rag:
            self.retrieval_system = self._initialize_retrieval_system(
                retriever_name, corpus_name, db_dir, corpus_cache
            )
        
        print(f"RAG System initialized:")
        print(f"  LLM: {self.llm}")
        print(f"  RAG enabled: {self.use_rag}")
        if self.use_rag:
            print(f"  Retriever: {self.retriever_name}")
            print(f"  Corpus: {self.corpus_name}")
    
    def _initialize_llm(
        self, 
        provider: str, 
        config: Dict[str, Any]
    ) -> BaseLLM:
        """
        Initialize LLM with the specified provider and configuration.
        
        Args:
            provider: LLM provider type
            config: LLM configuration
            
        Returns:
            Initialized LLM instance
        """
        try:
            return LLMFactory.create_llm(provider, **config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM '{provider}': {e}")
    
    def _initialize_retrieval_system(
        self,
        retriever_name: str,
        corpus_name: str,
        db_dir: str,
        corpus_cache: bool
    ):
        """
        Initialize retrieval system.
        
        Args:
            retriever_name: Name of retriever configuration
            corpus_name: Name of corpus configuration
            db_dir: Database directory
            corpus_cache: Whether to cache corpus documents
            
        Returns:
            Initialized retrieval system
        """
        try:
            return create_retrieval_system(
                retriever_name=retriever_name,
                corpus_name=corpus_name,
                db_dir=db_dir,
                cache=corpus_cache
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize retrieval system: {e}")
    
    def answer(
        self,
        question_data: Dict[str, Any],
        k: int = RAG_DEFAULTS["retrieval_k"],
        rrf_k: int = RAG_DEFAULTS["rrf_k"],
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]], List[float]]:
        """
        Generate answer for a question using RAG or direct LLM.
        
        Args:
            question_data: Question data dictionary with 'question', 'type', 'options'
            k: Number of documents to retrieve
            rrf_k: RRF parameter for retrieval
            save_dir: Directory to save debugging artifacts
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (answer, retrieved_snippets, scores)
        """
        # Validate input
        validate_question_data(question_data)
        
        # Prepare question data
        prepared_data, original_options = prepare_question_data(question_data)
        
        # Retrieve context if RAG is enabled
        retrieved_snippets, scores = [], []
        context_str = ""
        
        if self.use_rag and self.retrieval_system:
            retrieved_snippets, scores = self._retrieve_context(
                prepared_data, k, rrf_k
            )
            context_str = format_context_snippets(retrieved_snippets)
        
        # Generate prompt
        prompt = self._generate_prompt(
            prepared_data, context_str, original_options
        )
        
        # Generate response
        response = self._generate_response(prompt, **kwargs)
        
        # Save artifacts if requested
        if save_dir:
            save_rag_artifacts(save_dir, prompt, response, retrieved_snippets)
        
        return response, retrieved_snippets, scores
    
    def rag_answer_textgrad(
        self,
        question_data: Dict[str, Any],
        custom_prompt: str,
        k: int = RAG_DEFAULTS["retrieval_k"],
        rrf_k: int = RAG_DEFAULTS["rrf_k"],
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]], List[float], str]:
        """
        Generate answer using a custom prompt (for TextGrad or custom experiments).
        
        Args:
            question_data: Question data dictionary
            custom_prompt: Custom prompt to use
            k: Number of documents to retrieve
            rrf_k: RRF parameter for retrieval
            save_dir: Directory to save debugging artifacts
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (answer, retrieved_snippets, scores, final_prompt)
        """
        # Validate input
        validate_question_data(question_data)
        
        # Retrieve context if RAG is enabled
        retrieved_snippets, scores = [], []
        context_str = ""
        
        if self.use_rag and self.retrieval_system:
            # Use just the question for retrieval
            query = question_data.get("question", "")
            retrieved_snippets, scores = self._retrieve_context_by_query(
                query, k, rrf_k
            )
            context_str = format_context_snippets(retrieved_snippets)
        
        # Build final prompt with context
        final_prompt = custom_prompt
        if self.use_rag and context_str:
            final_prompt = (
                f"With your own knowledge and the help of the following document:\n\n"
                f"{context_str}\n\n{custom_prompt}"
            )
        
        # Generate response
        response = self._generate_response(final_prompt, **kwargs)
        
        # Save artifacts if requested
        if save_dir:
            save_rag_artifacts(save_dir, final_prompt, response, retrieved_snippets)
        
        return response, retrieved_snippets, scores, final_prompt
    
    def _retrieve_context(
        self,
        question_data: Dict[str, Any],
        k: int,
        rrf_k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve context documents for the question.
        
        Args:
            question_data: Prepared question data
            k: Number of documents to retrieve
            rrf_k: RRF parameter
            
        Returns:
            Tuple of (retrieved_snippets, scores)
        """
        # Build retrieval query
        query = build_retrieval_query(
            question_data.get("question", ""),
            question_data.get("options", "")
        )
        
        return self._retrieve_context_by_query(query, k, rrf_k)
    
    def _retrieve_context_by_query(
        self,
        query: str,
        k: int,
        rrf_k: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve context documents by query string.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            rrf_k: RRF parameter
            
        Returns:
            Tuple of (retrieved_snippets, scores)
        """
        try:
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                query, k=k, rrf_k=rrf_k
            )
            
            # Validate return values
            if not isinstance(retrieved_snippets, list) or not isinstance(scores, list):
                print("⚠️ Warning: invalid return values from retrieval")
                return [], []
            
            return retrieved_snippets, scores
            
        except Exception as e:
            print(f"⚠️ Warning: retrieval failed: {e}")
            return [], []
    
    def _generate_prompt(
        self,
        question_data: Dict[str, Any],
        context_str: str,
        original_options: Any
    ) -> str:
        """
        Generate prompt for the question.
        
        Args:
            question_data: Prepared question data
            context_str: Context string from retrieval
            original_options: Original options format
            
        Returns:
            Generated prompt
        """
        # Add context to question data
        question_data["context"] = context_str
        
        # Try to use templates first
        if self.templates:
            prompt = self._generate_prompt_from_template(question_data)
            if prompt:
                return self._add_context_to_prompt(prompt, context_str)
        
        # Fallback to direct prompt creation
        fallback_data = question_data.copy()
        if original_options is not None:
            fallback_data["options"] = original_options
        
        prompt = create_direct_prompt(fallback_data)
        return self._add_context_to_prompt(prompt, context_str)
    
    def _generate_prompt_from_template(
        self,
        question_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate prompt using templates.
        
        Args:
            question_data: Question data
            
        Returns:
            Generated prompt or None if template rendering fails
        """
        try:
            qtype = question_data.get("type", "multiple_choice")
            
            if qtype in self.templates:
                template = self.templates[qtype]
            else:
                template = self.templates.get("multiple_choice")
            
            if template and hasattr(template, "render"):
                return template.render(**question_data)
            
        except Exception as e:
            print(f"Template rendering error: {e}")
        
        return None
    
    def _add_context_to_prompt(self, prompt: str, context_str: str) -> str:
        """
        Add context to prompt if RAG is enabled.
        
        Args:
            prompt: Base prompt
            context_str: Context string
            
        Returns:
            Prompt with context added
        """
        if self.use_rag and context_str:
            # Check if context fits within limits
            max_context_tokens = self.llm.context_length - len(prompt) // 4
            context_str = truncate_context(context_str, max_context_tokens)
            
            return (
                f"With your own knowledge and the help of the following document:\n\n"
                f"{context_str}\n\n{prompt}"
            )
        
        return prompt
    
    def _generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using LLM.
        
        Args:
            prompt: Prompt to generate from
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Build messages
        system_content = PROMPT_SETTINGS["system_prompt"]
        messages = build_messages(system_content, prompt)
        
        # Generate response
        try:
            response = self.llm.generate(messages, **kwargs)
            return response
        except Exception as e:
            print(f"⚠️ Generation failed: {e}")
            return ""
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        Process LLM response to extract structured answer.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Processed response dictionary
        """
        return process_llm_response(response)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "rag_enabled": self.use_rag,
            "llm_info": self.llm.get_info(),
            "retrieval_info": None
        }
        
        if self.use_rag and self.retrieval_system:
            info["retrieval_info"] = {
                "retriever_name": self.retriever_name,
                "corpus_name": self.corpus_name,
                "db_dir": self.db_dir
            }
        
        return info
    
    def update_llm_config(self, **kwargs) -> None:
        """
        Update LLM configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.llm, key):
                setattr(self.llm, key, value)
    
    def set_templates(self, templates: Dict[str, Any]) -> None:
        """
        Set prompt templates.
        
        Args:
            templates: Dictionary of templates
        """
        self.templates = templates
    
    def enable_rag(self, enable: bool = True) -> None:
        """
        Enable or disable RAG functionality.
        
        Args:
            enable: Whether to enable RAG
        """
        if enable and not self.retrieval_system:
            # Initialize retrieval system if not already done
            self.retrieval_system = self._initialize_retrieval_system(
                self.retriever_name, self.corpus_name, self.db_dir, False
            )
        
        self.use_rag = enable