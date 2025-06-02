"""Utility functions for RAG system."""

import json
import os
import re
from typing import Dict, Any, List, Optional, Union, Tuple

from .rag_config import JSON_PARSING, PROMPT_SETTINGS


def extract_valid_json(text: str) -> str:
    """
    Pull out the first {...} that actually contains an 'answer' or 'answer_choice'.
    
    Args:
        text: Text containing JSON
        
    Returns:
        First valid JSON string found, or empty dict if none found
    """
    # This simple regex will only handle one-level JSON
    matches = re.findall(r'\{[^{}]+\}', text)
    
    for match in matches:
        try:
            obj = json.loads(match)
            if any(key in obj for key in JSON_PARSING["valid_answer_keys"]):
                return match
        except json.JSONDecodeError:
            continue
    
    return "{}"


def safe_json_parse(response: str) -> Dict[str, Any]:
    """
    Parse the JSON we extracted and normalize it into standard format.
    
    Args:
        response: Response string containing JSON
        
    Returns:
        Parsed and normalized JSON dictionary
    """
    try:
        data = json.loads(response)
        
        # Check for direct answer keys
        if "answer" in data:
            return {"answer": data["answer"]}
        if "answer_choice" in data:
            return {"answer_choice": data["answer_choice"]}
        
        # Check for nested answer in data field
        if "data" in data and isinstance(data["data"], dict):
            if "answer" in data["data"]:
                return {"answer": data["data"]["answer"]}
            if "answer_choice" in data["data"]:
                return {"answer_choice": data["data"]["answer_choice"]}
                
    except json.JSONDecodeError:
        pass
    
    return JSON_PARSING["fallback_response"]


def process_llm_response(response: str) -> Dict[str, Any]:
    """
    Process LLM response to extract and validate JSON.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Processed JSON response
    """
    if not response:
        return JSON_PARSING["fallback_response"]
    
    # First, try to extract valid JSON
    json_str = extract_valid_json(response)
    
    # Then parse it safely
    return safe_json_parse(json_str)


def format_context_snippets(
    snippets: List[Dict[str, Any]], 
    max_snippets: int = PROMPT_SETTINGS["max_context_snippets"],
    max_length: int = PROMPT_SETTINGS["snippet_max_length"]
) -> str:
    """
    Format retrieved snippets into context string.
    
    Args:
        snippets: List of retrieved document snippets
        max_snippets: Maximum number of snippets to include
        max_length: Maximum length per snippet
        
    Returns:
        Formatted context string
    """
    if not snippets:
        return ""
    
    contexts = []
    for idx, snippet in enumerate(snippets[:max_snippets]):
        if not isinstance(snippet, dict):
            continue
        
        title = snippet.get("title", "Unknown")
        content = snippet.get("content", "")
        
        # Truncate content if too long
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        contexts.append(f"Document {idx+1} (Title: {title}): {content}")
    
    return "\n".join(contexts)


def build_messages(
    system_prompt: str,
    user_prompt: str
) -> List[Dict[str, str]]:
    """
    Build message list for LLM generation.
    
    Args:
        system_prompt: System prompt content
        user_prompt: User prompt content
        
    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def create_direct_prompt(question_data: Dict[str, Any]) -> str:
    """
    Create a prompt in the exact style of the baseline code.
    This serves as a fallback when template rendering fails.
    
    Args:
        question_data: Question data dictionary
        
    Returns:
        Formatted prompt string
    """
    qtype = question_data.get("type", "multiple_choice")
    question = question_data.get("question", "No question provided")
    options = question_data.get("options", {})
    
    if qtype == "true_false":
        prompt = (
            f"Question: {question}\n"
            "Options:\n"
            "A. True\n"
            "B. False\n\n"
            "Please respond **only** with a single valid JSON object in the following format:\n"
            '{"answer": "True"}  ← if the answer is true\n'
            '{"answer": "False"} ← if the answer is false\n'
            "Do not include any other text or comments. Output must be strictly JSON."
        )
    elif qtype == "multiple_choice":
        # Convert options dict to formatted string
        if isinstance(options, dict):
            options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        else:
            options_str = str(options) if options else ""
            
        prompt = (
            f"Question: {question}\n"
            f"Options:\n{options_str}\n\n"
            "Respond strictly in valid JSON format as follows:\n"
            '{"answer_choice": "A"} ← if A is the answer\n'
            "Output only the JSON object. Do not include any explanation, commentary, markdown, or extra text."
        )
    elif qtype == "list":
        # Convert options list to formatted string
        if isinstance(options, list):
            options_str = "\n".join([f"{idx+1}. {option}" for idx, option in enumerate(options)])
        else:
            options_str = str(options) if options else ""
            
        prompt = (
            f"Question: {question}\n"
            f"Options:\n{options_str}\n\n"
            "Respond strictly in valid JSON format as shown below:\n"
            '{"answer": ["1", "3"]} ← if options 1 and 3 are correct\n'
            "Only output the JSON object. Do not include explanations, labels, markdown, or any other text."
        )
    else:
        # Default to multiple choice format for unknown types
        prompt = (
            f"Question: {question}\n"
            "Respond in JSON format with your answer."
        )
    
    return prompt


def format_options_for_template(options: Union[Dict, List, None]) -> str:
    """
    Format options for use in templates.
    
    Args:
        options: Options in various formats
        
    Returns:
        Formatted options string
    """
    if options is None:
        return ""
    
    if isinstance(options, dict):
        return "\n".join([f"{key}. {value}" for key, value in options.items()])
    elif isinstance(options, list):
        return "\n".join([f"{idx+1}. {option}" for idx, option in enumerate(options)])
    else:
        return str(options)


def prepare_question_data(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """
    Prepare question data for processing, returning both template-ready and original formats.
    
    Args:
        question_data: Raw question data
        
    Returns:
        Tuple of (template_ready_data, original_options)
    """
    # Make a copy to avoid modifying original
    prepared_data = question_data.copy()
    
    # Store original options for fallback
    original_options = question_data.get("options")
    
    # Ensure essential fields exist
    if "question" not in prepared_data:
        prepared_data["question"] = "No question provided"
    
    # Format options for template
    if "options" in prepared_data:
        prepared_data["options"] = format_options_for_template(prepared_data["options"])
    else:
        prepared_data["options"] = ""
    
    return prepared_data, original_options


def save_rag_artifacts(
    save_dir: str,
    prompt: str,
    response: str,
    snippets: List[Dict[str, Any]]
) -> None:
    """
    Save RAG artifacts for debugging and analysis.
    
    Args:
        save_dir: Directory to save artifacts
        prompt: Generated prompt
        response: LLM response
        snippets: Retrieved snippets
    """
    if not save_dir:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save prompt
    with open(os.path.join(save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    
    # Save response
    with open(os.path.join(save_dir, "response.txt"), "w", encoding="utf-8") as f:
        f.write(response)
    
    # Save response as JSON
    with open(os.path.join(save_dir, "response.json"), "w", encoding="utf-8") as f:
        json.dump({"response": response}, f, indent=4, ensure_ascii=False)
    
    # Save snippets
    with open(os.path.join(save_dir, "snippets.json"), "w", encoding="utf-8") as f:
        json.dump(snippets, f, indent=4, ensure_ascii=False)


def build_retrieval_query(question: str, options: str = "") -> str:
    """
    Build retrieval query from question and options.
    
    Args:
        question: Base question
        options: Formatted options string
        
    Returns:
        Combined retrieval query
    """
    if options and options.strip():
        return f"{question} Options: {options}"
    return question


def validate_question_data(question_data: Dict[str, Any]) -> None:
    """
    Validate question data format.
    
    Args:
        question_data: Question data to validate
        
    Raises:
        ValueError: If question data is invalid
    """
    if not isinstance(question_data, dict):
        raise ValueError("Question data must be a dictionary")
    
    if "question" not in question_data:
        raise ValueError("Question data must contain 'question' field")
    
    # Validate question type if present
    valid_types = ["multiple_choice", "true_false", "list"]
    qtype = question_data.get("type")
    if qtype and qtype not in valid_types:
        print(f"Warning: Unknown question type '{qtype}', will use default handling")


def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count for text.
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    # Very rough approximation: ~4 characters per token
    return len(text) // 4


def truncate_context(
    context: str, 
    max_tokens: int, 
    preserve_docs: bool = True
) -> str:
    """
    Truncate context to fit within token limit.
    
    Args:
        context: Context string to truncate
        max_tokens: Maximum allowed tokens
        preserve_docs: Whether to preserve document boundaries
        
    Returns:
        Truncated context
    """
    if estimate_token_count(context) <= max_tokens:
        return context
    
    max_chars = max_tokens * 4  # Rough conversion
    
    if preserve_docs and "Document " in context:
        # Try to preserve complete documents
        docs = context.split("Document ")
        truncated_docs = []
        current_length = 0
        
        for i, doc in enumerate(docs):
            if i == 0:  # Skip empty first part
                continue
            
            doc_with_prefix = f"Document {doc}"
            if current_length + len(doc_with_prefix) <= max_chars:
                truncated_docs.append(doc_with_prefix)
                current_length += len(doc_with_prefix)
            else:
                break
        
        if truncated_docs:
            return "\n".join(truncated_docs)
    
    # Fallback: simple truncation
    return context[:max_chars] + "..." if len(context) > max_chars else context