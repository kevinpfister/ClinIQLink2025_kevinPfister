"""Document extraction functionality for retrieving cached documents."""

import json
import os
from typing import List, Dict, Any, Union

import tqdm

from .config import CORPUS_NAMES, DATA_SOURCES
from .utils import ensure_directory


class DocExtracter:
    """Document extracter for retrieving cached documents by ID."""
    
    def __init__(
        self, 
        db_dir: str = "./corpus", 
        cache: bool = False, 
        corpus_name: str = "MedCorp"
    ):
        """
        Initialize document extracter.
        
        Args:
            db_dir: Database directory path
            cache: Whether to cache documents in memory
            corpus_name: Name of the corpus collection
        """
        self.db_dir = db_dir
        self.cache = cache
        self.corpus_name = corpus_name
        
        print("Initializing document extracter...")
        
        # Ensure all corpus files are available
        self._ensure_corpora_available()
        
        # Set up document cache
        self._initialize_cache()
        
        print(f"Document extracter initialization complete. Cached {len(self.dict)} documents.")
    
    def _ensure_corpora_available(self) -> None:
        """Ensure all required corpora are available locally."""
        for corpus in CORPUS_NAMES[self.corpus_name]:
            corpus_dir = os.path.join(self.db_dir, corpus, "chunk")
            if not os.path.exists(corpus_dir):
                self._download_corpus(corpus)
    
    def _download_corpus(self, corpus: str) -> None:
        """
        Download a specific corpus.
        
        Args:
            corpus: Name of the corpus to download
        """
        print(f"Cloning the {corpus} corpus from Huggingface...")
        
        corpus_root = os.path.join(self.db_dir, corpus)
        repo_url = f"{DATA_SOURCES['huggingface_base']}{corpus}"
        
        result = os.system(f"git clone {repo_url} {corpus_root}")
        
        if result != 0:
            raise RuntimeError(f"Failed to clone corpus {corpus}")
        
        # Handle special case for StatPearls
        if corpus == "statpearls":
            self._setup_statpearls(corpus_root)
    
    def _setup_statpearls(self, corpus_dir: str) -> None:
        """
        Set up StatPearls corpus with special handling.
        
        Args:
            corpus_dir: Directory containing the corpus
        """
        print("Downloading the statpearls corpus from NCBI bookshelf...")
        
        url = DATA_SOURCES["statpearls_url"]
        wget_result = os.system(f"wget {url} -P {corpus_dir}")
        
        if wget_result != 0:
            raise RuntimeError("Failed to download StatPearls corpus")
        
        tar_file = os.path.join(corpus_dir, "statpearls_NBK430685.tar.gz")
        tar_result = os.system(f"tar -xzvf {tar_file} -C {corpus_dir}")
        
        if tar_result != 0:
            raise RuntimeError("Failed to extract StatPearls corpus")
        
        print("Chunking the statpearls corpus...")
        chunk_result = os.system("python src/data/statpearls.py")
        
        if chunk_result != 0:
            print("Warning: StatPearls chunking script failed")
    
    def _initialize_cache(self) -> None:
        """Initialize the document cache (either full cache or path cache)."""
        if self.cache:
            self._initialize_full_cache()
        else:
            self._initialize_path_cache()
    
    def _initialize_full_cache(self) -> None:
        """Initialize full document cache (documents stored in memory)."""
        cache_file = os.path.join(self.db_dir, f"{self.corpus_name}_id2text.json")
        
        if os.path.exists(cache_file):
            print("Loading existing full document cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.dict = json.load(f)
        else:
            print("Building full document cache...")
            self.dict = {}
            self._build_full_cache()
            
            print("Saving full document cache...")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.dict, f)
    
    def _initialize_path_cache(self) -> None:
        """Initialize path cache (only file paths and indices stored)."""
        cache_file = os.path.join(self.db_dir, f"{self.corpus_name}_id2path.json")
        
        if os.path.exists(cache_file):
            print("Loading existing path cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.dict = json.load(f)
        else:
            print("Building path cache...")
            self.dict = {}
            self._build_path_cache()
            
            print("Saving path cache...")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.dict, f, indent=4)
    
    def _build_full_cache(self) -> None:
        """Build full document cache by loading all documents into memory."""
        for corpus in CORPUS_NAMES[self.corpus_name]:
            corpus_dir = os.path.join(self.db_dir, corpus, "chunk")
            
            if not os.path.exists(corpus_dir):
                print(f"Warning: Directory {corpus_dir} not found")
                continue
            
            self._process_corpus_for_full_cache(corpus_dir)
    
    def _build_path_cache(self) -> None:
        """Build path cache by storing file paths and indices."""
        for corpus in CORPUS_NAMES[self.corpus_name]:
            corpus_dir = os.path.join(self.db_dir, corpus, "chunk")
            
            if not os.path.exists(corpus_dir):
                print(f"Warning: Directory {corpus_dir} not found")
                continue
            
            self._process_corpus_for_path_cache(corpus, corpus_dir)
    
    def _process_corpus_for_full_cache(self, corpus_dir: str) -> None:
        """
        Process a corpus directory for full caching.
        
        Args:
            corpus_dir: Path to corpus chunk directory
        """
        files = [f for f in sorted(os.listdir(corpus_dir)) if f.endswith('.jsonl')]
        
        for fname in tqdm.tqdm(files, desc=f"Processing {corpus_dir}"):
            file_path = os.path.join(corpus_dir, fname)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                
                for line in content.split('\n'):
                    if not line.strip():
                        continue
                    
                    try:
                        item = json.loads(line)
                        # Remove 'contents' field to save space, keep 'content'
                        item.pop("contents", None)
                        self.dict[item["id"]] = item
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def _process_corpus_for_path_cache(self, corpus: str, corpus_dir: str) -> None:
        """
        Process a corpus directory for path caching.
        
        Args:
            corpus: Name of the corpus
            corpus_dir: Path to corpus chunk directory
        """
        files = [f for f in sorted(os.listdir(corpus_dir)) if f.endswith('.jsonl')]
        
        for fname in tqdm.tqdm(files, desc=f"Processing {corpus}"):
            file_path = os.path.join(corpus_dir, fname)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                
                for i, line in enumerate(content.split('\n')):
                    if not line.strip():
                        continue
                    
                    try:
                        item = json.loads(line)
                        self.dict[item["id"]] = {
                            "fpath": os.path.join(corpus, "chunk", fname),
                            "index": i
                        }
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def extract(self, ids: Union[List[str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Extract document content from IDs.
        
        Args:
            ids: List of document IDs (strings) or dictionaries with 'id' field
            
        Returns:
            List of extracted documents
        """
        if not ids:
            return []
        
        results = []
        
        for item in ids:
            try:
                # Handle both string IDs and dict items
                item_id = item if isinstance(item, str) else item.get("id")
                
                if not item_id:
                    print("Warning: Empty or missing ID")
                    continue
                
                if item_id not in self.dict:
                    print(f"Warning: ID {item_id} not found in cache")
                    continue
                
                if self.cache:
                    # Return cached document directly
                    results.append(self.dict[item_id])
                else:
                    # Load document from file
                    doc = self._load_document_from_path(item_id)
                    if doc:
                        results.append(doc)
                        
            except Exception as e:
                print(f"Error extracting document: {e}")
        
        return results
    
    def _load_document_from_path(self, item_id: str) -> Union[Dict[str, Any], None]:
        """
        Load document from file using path cache.
        
        Args:
            item_id: Document ID
            
        Returns:
            Document dictionary or None if loading fails
        """
        try:
            item_info = self.dict[item_id]
            file_path = os.path.join(self.db_dir, item_info["fpath"])
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
            
            index = item_info["index"]
            if 0 <= index < len(lines):
                doc = json.loads(lines[index])
                return doc
            else:
                print(f"Warning: Index {index} out of range for {file_path}")
                return None
                
        except Exception as e:
            print(f"Error loading document {item_id}: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "total_documents": len(self.dict),
            "cache_type": "full" if self.cache else "path",
            "corpus_name": self.corpus_name,
            "db_directory": self.db_dir
        }