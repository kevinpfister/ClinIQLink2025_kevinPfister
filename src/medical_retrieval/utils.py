"""Utility functions for the retrieval system."""

import os
import glob
import shutil
import subprocess
import tempfile
import time
from typing import Optional, List, Tuple
from pathlib import Path

from .config import INDEXING_CONFIG, DEFAULTS


def concat(title: Optional[str], content: Optional[str]) -> str:
    """
    Concatenate title and content with proper punctuation.
    
    Args:
        title: Document title
        content: Document content
        
    Returns:
        Properly formatted concatenated string
    """
    title = str(title).strip() if title else ""
    content = str(content).strip() if content else ""
    
    if not title:
        return content
    
    ending_punctuation = ('.', '?', '!')
    if any(title.endswith(char) for char in ending_punctuation):
        return f"{title} {content}"
    else:
        return f"{title}. {content}"


def ensure_directory(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def clean_directory(directory: str) -> None:
    """Remove directory and all its contents if it exists."""
    if os.path.exists(directory):
        shutil.rmtree(directory)


def get_jsonl_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """
    Get list of JSONL files in directory.
    
    Args:
        directory: Directory to search
        max_files: Maximum number of files to return
        
    Returns:
        List of JSONL file paths
    """
    pattern = os.path.join(directory, "*.jsonl")
    files = sorted(glob.glob(pattern))
    
    if max_files is not None:
        files = files[:max_files]
        
    return files


def determine_batch_size(file_count: int) -> Optional[int]:
    """
    Determine appropriate batch size based on file count.
    
    Args:
        file_count: Number of files to process
        
    Returns:
        Batch size or None for processing all at once
    """
    config = INDEXING_CONFIG
    
    if file_count <= config["small_corpus_threshold"]:
        return config["batch_sizes"]["small"]
    elif file_count <= config["medium_corpus_threshold"]:
        return config["batch_sizes"]["medium"]
    elif file_count <= config["large_corpus_threshold"]:
        return config["batch_sizes"]["large"]
    else:
        return config["batch_sizes"]["huge"]


def run_indexing_command(
    cmd: List[str], 
    env: dict, 
    batch_num: int, 
    is_retry: bool = False
) -> Tuple[bool, str]:
    """
    Run indexing command and handle errors.
    
    Args:
        cmd: Command to run
        env: Environment variables
        batch_num: Batch number for logging
        is_retry: Whether this is a retry attempt
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if process.returncode != 0:
            error_msg = f"Batch {batch_num} indexing failed: {process.stderr}"
            return False, error_msg
            
        return True, ""
        
    except Exception as e:
        error_msg = f"Error during batch {batch_num} indexing: {e}"
        return False, error_msg


def batch_index_files(
    input_dir: str, 
    output_dir: str, 
    batch_size: int = 1000, 
    max_files: Optional[int] = None
) -> bool:
    """
    Create a Lucene index in smaller batches to avoid memory issues.
    
    Args:
        input_dir: Directory containing JSONL files
        output_dir: Directory to store the index
        batch_size: Number of files to process at once
        max_files: Maximum number of files to process (for testing)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"Starting batch indexing process from {input_dir} to {output_dir}")
    
    # Clean up any existing index
    clean_directory(output_dir)
    ensure_directory(output_dir)
    
    # Find all JSONL files
    jsonl_files = get_jsonl_files(input_dir, max_files)
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return False
    
    total_files = len(jsonl_files)
    print(f"Found {total_files} JSONL files to process")
    
    # Process in batches
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = jsonl_files[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        success = _process_batch(
            batch_files, output_dir, batch_num, batch_num == 1
        )
        
        if not success and batch_num == 1:
            # First batch failed - critical error
            return False
        elif not success:
            # Later batch failed - continue with partial index
            print("Continuing with partial index...")
        
        # Sleep briefly between batches
        print(f"Batch {batch_num} complete. Taking a short break...")
        time.sleep(DEFAULTS["sleep_between_batches"])
    
    print(f"Indexing completed successfully. Index stored at {output_dir}")
    return True


def _process_batch(
    batch_files: List[str], 
    output_dir: str, 
    batch_num: int, 
    is_first_batch: bool
) -> bool:
    """Process a single batch of files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy files to temp directory
        for file_path in batch_files:
            shutil.copy2(file_path, temp_dir)
        
        # Set up environment
        env = os.environ.copy()
        env["JAVA_OPTS"] = INDEXING_CONFIG["memory_settings"]["default"]
        
        # Create command
        cmd = _build_indexing_command(temp_dir, output_dir, is_first_batch)
        
        print(f"Running: {' '.join(cmd)}")
        success, error_msg = run_indexing_command(cmd, env, batch_num)
        
        if not success:
            print(error_msg)
            
            if is_first_batch:
                # Retry first batch with minimal settings
                print("Retrying first batch with minimal settings...")
                return _retry_with_minimal_settings(temp_dir, output_dir, batch_num)
            
            return False
        
        return True


def _build_indexing_command(
    temp_dir: str, 
    output_dir: str, 
    is_first_batch: bool
) -> List[str]:
    """Build the indexing command."""
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", temp_dir,
        "--index", output_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(INDEXING_CONFIG["thread_counts"]["default"]),
        "--storeRaw"
    ]
    
    if not is_first_batch:
        cmd.append("--update")
    
    return cmd


def _retry_with_minimal_settings(
    temp_dir: str, 
    output_dir: str, 
    batch_num: int
) -> bool:
    """Retry indexing with minimal memory settings."""
    env = os.environ.copy()
    env["JAVA_OPTS"] = INDEXING_CONFIG["memory_settings"]["minimal"]
    
    cmd = _build_indexing_command(temp_dir, output_dir, True)
    cmd[8] = str(INDEXING_CONFIG["thread_counts"]["minimal"])  # threads
    
    success, error_msg = run_indexing_command(cmd, env, batch_num, is_retry=True)
    
    if not success:
        print("First batch failed even with minimal settings")
        print(error_msg)
    
    return success


def create_minimal_index(
    input_dir: str, 
    output_dir: str, 
    file_limit: int = DEFAULTS["minimal_index_files"]
) -> bool:
    """
    Create a minimal index with just a few files for basic functionality.
    
    Args:
        input_dir: Directory containing JSONL files
        output_dir: Directory to store the index
        file_limit: Maximum number of files to include
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Creating minimal index with {file_limit} files...")
    
    # Clean up any existing index
    clean_directory(output_dir)
    ensure_directory(output_dir)
    
    # Find limited number of JSONL files
    jsonl_files = get_jsonl_files(input_dir, file_limit)
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return False
    
    # Create temporary directory and copy files
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_path in jsonl_files:
            shutil.copy2(file_path, temp_dir)
        
        # Index with minimal settings
        env = os.environ.copy()
        env["JAVA_OPTS"] = INDEXING_CONFIG["memory_settings"]["minimal"]
        
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", temp_dir,
            "--index", output_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(INDEXING_CONFIG["thread_counts"]["minimal"]),
            "--storeRaw"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        success, error_msg = run_indexing_command(cmd, env, 1)
        
        if not success:
            print(f"Minimal indexing failed: {error_msg}")
            return False
    
    print(f"Minimal index created successfully at {output_dir}")
    return True


def clone_corpus(corpus_name: str, corpus_dir: str) -> bool:
    """
    Clone corpus from HuggingFace.
    
    Args:
        corpus_name: Name of the corpus
        corpus_dir: Directory to clone to
        
    Returns:
        True if successful, False otherwise
    """
    if corpus_name == "selfcorpus":
        print("Skipping cloning for selfcorpus")
        return True
    
    try:
        from .config import DATA_SOURCES
        
        print(f"Cloning the {corpus_name} corpus from Huggingface...")
        ensure_directory(os.path.dirname(corpus_dir))
        
        repo_url = f"{DATA_SOURCES['huggingface_base']}{corpus_name}"
        result = os.system(f"git clone {repo_url} {corpus_dir}")
        
        return result == 0
        
    except Exception as e:
        print(f"Error cloning corpus {corpus_name}: {e}")
        return False


def download_statpearls(corpus_dir: str) -> bool:
    """
    Download and extract StatPearls corpus.
    
    Args:
        corpus_dir: Directory to download to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from .config import DATA_SOURCES
        
        print("Downloading the statpearls corpus from NCBI bookshelf...")
        
        url = DATA_SOURCES["statpearls_url"]
        wget_result = os.system(f"wget {url} -P {corpus_dir}")
        
        if wget_result != 0:
            return False
        
        tar_file = os.path.join(corpus_dir, "statpearls_NBK430685.tar.gz")
        tar_result = os.system(f"tar -xzvf {tar_file} -C {corpus_dir}")
        
        if tar_result != 0:
            return False
        
        print("Chunking the statpearls corpus...")
        chunk_result = os.system("python src/data/statpearls.py")
        
        return chunk_result == 0
        
    except Exception as e:
        print(f"Error downloading StatPearls: {e}")
        return False