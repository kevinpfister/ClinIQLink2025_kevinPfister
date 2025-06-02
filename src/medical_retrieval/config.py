"""Configuration constants for the retrieval system."""

from typing import Dict, List

# Corpus configuration
CORPUS_NAMES: Dict[str, List[str]] = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedText": ["textbooks", "statpearls"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
    "SelfCorpus": ["selfcorpus"],
    "All": ["selfcorpus", "pubmed", "textbooks", "statpearls", "wikipedia"]
}

# Retriever configuration
RETRIEVER_NAMES: Dict[str, List[str]] = {
    "BM25": ["bm25"],
    "MedCPT": ["medcpt"],
    "SPLADE": ["splade"],
    "UniCOIL": ["unicoil"],
    "MedicalHybrid": ["bm25", "medcpt"],                        # Lexical + biomedical semantics
    "PrecisionHybrid": ["bm25", "unicoil"],                     # Exact + weighted term matching
    "ExpansionHybrid": ["bm25", "splade"],                      # Traditional + learned expansion  
    "SparseHybrid": ["unicoil", "splade"],                      # Precision + expansion (complementary)
    "OptimalHybrid": ["bm25", "medcpt", "unicoil", "splade"]    # All paradigms
}

# Model configuration
MODEL_CONFIGS = {
    "splade": {
        "model_name": "naver/splade-cocondenser-ensembledistil",
        "max_expansion_terms": 5,
        "relevance_threshold": 0.5
    },
    "unicoil": {
        "model_name": "dmis-lab/biobert-base-cased-v1.1",
        "fallback_model": "bert-base-uncased",
        "max_length": 256,
        "weight_range": (0.1, 3.0)
    },
    "medcpt": {
        "query_encoder": "ncbi/MedCPT-Query-Encoder",
        "article_encoder": "ncbi/MedCPT-Article-Encoder",
        "max_length": 512,
        "batch_size": 8,
        "medcpt_weight": 0.7,
        "bm25_weight": 0.3
    }
}

# Indexing configuration
INDEXING_CONFIG = {
    "small_corpus_threshold": 100,
    "medium_corpus_threshold": 1000,
    "large_corpus_threshold": 5000,
    "batch_sizes": {
        "small": None,  # Process all at once
        "medium": 200,
        "large": 100,
        "huge": 50
    },
    "memory_settings": {
        "default": "-Xms2g -Xmx4g",
        "minimal": "-Xms1g -Xmx2g"
    },
    "thread_counts": {
        "default": 2,
        "minimal": 1
    }
}

# File paths and extensions
FILE_EXTENSIONS = {
    "jsonl": ".jsonl",
    "json": ".json"
}

# URLs for data sources
DATA_SOURCES = {
    "statpearls_url": "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz",
    "huggingface_base": "https://huggingface.co/datasets/MedRAG/"
}

# Default values
DEFAULTS = {
    "retrieval_k": 5,
    "rrf_k": 60,
    "candidate_multiplier": 3,
    "max_candidates": 100,
    "minimal_index_files": 50,
    "sleep_between_batches": 2
}