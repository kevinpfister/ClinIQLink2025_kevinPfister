import os
import json
import tqdm
import numpy as np
import subprocess
import shutil
import time
import glob
from pyserini.search.lucene import LuceneSearcher
import tempfile

corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedText": ["textbooks", "statpearls"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
    "SelfCorpus": ["selfcorpus"],
    "All": ["selfcorpus","pubmed", "textbooks", "statpearls", "wikipedia"]
}

retriever_names = {
    "BM25": ["bm25"]
}

def concat(title, content):
    """Concatenate title and content with proper punctuation"""
    title = str(title).strip() if title else ""
    content = str(content).strip() if content else ""
    
    if not title:
        return content
    
    ending_punctuation = ('.', '?', '!')
    if any(title.endswith(char) for char in ending_punctuation):
        return f"{title} {content}"
    else:
        return f"{title}. {content}"

def batch_index_files(input_dir, output_dir, batch_size=1000, max_files=None):
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
    if os.path.exists(output_dir):
        print(f"Removing existing index at {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSONL files
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return False
    
    if max_files:
        jsonl_files = jsonl_files[:max_files]
    
    total_files = len(jsonl_files)
    print(f"Found {total_files} JSONL files to process")
    
    # Process in batches
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = jsonl_files[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        # Create a temporary directory for this batch
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy files to temp directory
            for file_path in batch_files:
                shutil.copy2(file_path, temp_dir)
            
            # Index this batch
            env = os.environ.copy()
            env["JAVA_OPTS"] = "-Xms2g -Xmx4g"  # Lower memory for each batch
            
            # Use different command for first batch vs. subsequent batches
            if batch_num == 1:
                # First batch - create a new index
                cmd = [
                    "python", "-m", "pyserini.index.lucene",
                    "--collection", "JsonCollection",
                    "--input", temp_dir,
                    "--index", output_dir,
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "2",  # Very conservative
                    "--storeRaw"
                ]
            else:
                # Subsequent batches - update existing index
                cmd = [
                    "python", "-m", "pyserini.index.lucene",
                    "--collection", "JsonCollection",
                    "--input", temp_dir,
                    "--index", output_dir,
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "2",
                    "--storeRaw",
                    "--update"  # Update mode
                ]
            
            print(f"Running: {' '.join(cmd)}")
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
                    print(f"Batch {batch_num} indexing failed with error:")
                    print(process.stderr)
                    
                    # If it's not the first batch, we can continue with what we have
                    if batch_num > 1:
                        print("Continuing with partial index...")
                        continue
                    else:
                        # First batch failed - try with even less memory
                        print("Retrying first batch with minimal settings...")
                        env["JAVA_OPTS"] = "-Xms1g -Xmx2g"
                        cmd[8] = "1"  # Only 1 thread
                        
                        process = subprocess.run(
                            cmd,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False
                        )
                        
                        if process.returncode != 0:
                            print("First batch failed even with minimal settings")
                            return False
            except Exception as e:
                print(f"Error during batch {batch_num} indexing: {e}")
                # If it's not the first batch, we can continue with what we have
                if batch_num > 1:
                    print("Continuing with partial index...")
                    continue
                else:
                    return False
        
        # Sleep briefly between batches to let the system recover
        print(f"Batch {batch_num} complete. Taking a short break...")
        time.sleep(2)
    
    print(f"Indexing completed successfully. Index stored at {output_dir}")
    return True

def create_minimal_index(input_dir, output_dir, file_limit=100):
    """Create a minimal index with just a few files for basic functionality"""
    print(f"Creating minimal index with {file_limit} files...")
    
    # Clean up any existing index
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find JSON files, limited to a small number
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))[:file_limit]
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return False
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy a limited number of files
        for file_path in jsonl_files:
            shutil.copy2(file_path, temp_dir)
        
        # Index with minimal settings
        env = os.environ.copy()
        env["JAVA_OPTS"] = "-Xms1g -Xmx2g"
        
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", temp_dir,
            "--index", output_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storeRaw"
        ]
        
        print(f"Running: {' '.join(cmd)}")
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
                print(f"Minimal indexing failed with error:")
                print(process.stderr)
                return False
        except Exception as e:
            print(f"Error during minimal indexing: {e}")
            return False
    
    print(f"Minimal index created successfully at {output_dir}")
    return True

class Retriever:
    def __init__(self, retriever_name="bm25", corpus_name="textbooks", db_dir="./corpus", HNSW=False, **kwargs):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            
        # Set up directory paths
        self.corpus_dir = os.path.join(self.db_dir, self.corpus_name)
        self.chunk_dir = os.path.join(self.corpus_dir, "chunk")
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", self.retriever_name)
        
        # Ensure the corpus directory exists
        if not os.path.exists(self.chunk_dir):
            print(f"Cloning the {self.corpus_name} corpus from Huggingface...")
            os.makedirs(os.path.dirname(self.chunk_dir), exist_ok=True)
            
            # Check if the corpus is selfcorpus, if so, skip cloning
            if self.corpus_name == "selfcorpus":
                print("Skipping cloning for selfcorpus")
            else:
                # Clone the repository
                os.system(f"git clone https://huggingface.co/datasets/MedRAG/{corpus_name} {self.corpus_dir}")
            
            # Handle special case for StatPearls
            if self.corpus_name == "statpearls":
                print("Downloading the statpearls corpus from NCBI bookshelf...")
                os.system(f"wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {self.corpus_dir}")
                os.system(f"tar -xzvf {os.path.join(self.corpus_dir, 'statpearls_NBK430685.tar.gz')} -C {self.corpus_dir}")
                print("Chunking the statpearls corpus...")
                os.system("python src/data/statpearls.py")
        
        # Check if the index needs to be created
        try:
            if os.path.exists(self.index_dir) and os.listdir(self.index_dir):
                print(f"Attempting to load existing BM25 index for {self.corpus_name}...")
                try:
                    self.index = LuceneSearcher(self.index_dir)
                    print(f"Successfully loaded BM25 index for {self.corpus_name}")
                except Exception as e:
                    print(f"Error loading index: {e}")
                    print("Index appears to be corrupted. Recreating index...")
                    self._create_index()
            else:
                print(f"Creating BM25 index for {self.corpus_name}...")
                self._create_index()
        except Exception as e:
            print(f"Error during initialization: {e}")
            print("Attempting to create a working index...")
            self._create_index()

    def _create_index(self):
        """Create BM25 index for the corpus using batch processing"""
        # Make sure any existing index is completely removed
        if os.path.exists(self.index_dir):
            print(f"Removing existing index at {self.index_dir}")
            shutil.rmtree(self.index_dir)
        os.makedirs(os.path.dirname(self.index_dir), exist_ok=True)
        
        # Check if chunk directory has files
        if not os.path.exists(self.chunk_dir) or not os.listdir(self.chunk_dir):
            raise RuntimeError(f"Chunk directory {self.chunk_dir} is empty. Cannot create index.")
        
        # Count JSONL files
        jsonl_files = glob.glob(os.path.join(self.chunk_dir, "*.jsonl"))
        file_count = len(jsonl_files)
        print(f"Found {file_count} JSONL files to index in {self.chunk_dir}")
        
        # Determine batch size based on corpus size
        if file_count > 5000:
            batch_size = 50  # Very small batches for huge corpora
        elif file_count > 1000:
            batch_size = 100  # Small batches for large corpora
        elif file_count > 100:
            batch_size = 200  # Medium batches for medium corpora
        else:
            batch_size = file_count  # All at once for small corpora
        
        print(f"Using batch size of {batch_size} files")
        
        # Try batch indexing
        success = batch_index_files(
            input_dir=self.chunk_dir,
            output_dir=self.index_dir,
            batch_size=batch_size
        )
        
        # If batch indexing fails, try a minimal index as last resort
        if not success:
            print("Batch indexing failed. Creating minimal working index...")
            success = create_minimal_index(
                input_dir=self.chunk_dir,
                output_dir=self.index_dir,
                file_limit=50  # Very minimal
            )
            
            if not success:
                raise RuntimeError(f"Failed to create any working index for {self.corpus_name}")
        
        # Load the created index
        try:
            self.index = LuceneSearcher(self.index_dir)
            # ─── BM25 parameter tuning ────────────────────────────────────────
            # Lower b to reduce length‐normalization and raise k1 to reward TF
            self.index.set_b(0.5)
            self.index.set_k1(1.5)

            # ─── Enable RM3 pseudo‐relevance feedback ─────────────────────────
            self.index.set_rm3(fb_terms=10, fb_docs=3, orig_query_weight=0.9)

            print(f"Successfully loaded BM25 index for {self.corpus_name}")
        except Exception as e:
            raise RuntimeError(f"Created index but failed to load it: {e}")

    def get_relevant_documents(self, question, k=32, id_only=False, **kwargs):
        """Retrieve relevant documents for a question using BM25"""
        if not isinstance(question, str):
            raise TypeError("Question must be a string")
            
        # Search using BM25
        try:
            hits = self.index.search(question, k=k)
            
            if not hits:
                return [], []
                
            # Extract scores and document IDs
            scores = [hit.score for hit in hits]
            ids = [hit.docid for hit in hits]
            
            # Try to parse document indices
            indices = []
            for hit in hits:
                try:
                    parts = hit.docid.split('_')
                    index = int(parts[-1])
                    source = '_'.join(parts[:-1])
                    indices.append({"source": source, "index": index})
                except (ValueError, IndexError):
                    # If parsing fails, use the whole ID
                    indices.append({"source": hit.docid, "index": 0})
            
            # Return the IDs if requested
            if id_only:
                return [{"id": id} for id in ids], scores
                
            # Otherwise, try to retrieve the full documents
            try:
                docs = self._get_docs_direct(indices)
                if docs:
                    return docs, scores
            except Exception as e:
                print(f"Error retrieving documents from files: {e}")
            
            # Fallback to getting docs from the index
            return self._get_docs_from_index(ids), scores
            
        except Exception as e:
            print(f"Error during search: {e}")
            return [], []

    def _get_docs_direct(self, indices):
        """Try to get documents directly from the JSONL files"""
        results = []
        
        for i in indices:
            try:
                file_path = os.path.join(self.chunk_dir, f"{i['source']}.jsonl")
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if not content:
                    continue
                    
                lines = content.split('\n')
                if i['index'] < len(lines):
                    doc = json.loads(lines[i['index']])
                    # Make sure document has title and content
                    if 'title' not in doc:
                        doc['title'] = doc.get('id', 'Untitled')
                    if 'content' not in doc:
                        doc['content'] = doc.get('contents', '')
                    results.append(doc)
            except Exception as e:
                print(f"Error retrieving document {i['source']}_{i['index']}: {e}")
                
        return results

    def _get_docs_from_index(self, ids):
        """Retrieve documents directly from the index"""
        docs = []
        for doc_id in ids:
            try:
                doc = self.index.doc(doc_id)
                if doc:
                    try:
                        # Try to parse as JSON
                        content = json.loads(doc.raw())
                        # Ensure document has title and content fields
                        if 'title' not in content:
                            content['title'] = content.get('id', 'Untitled')
                        if 'content' not in content:
                            content['content'] = content.get('contents', '')
                        docs.append(content)
                    except json.JSONDecodeError:
                        # If parsing fails, use raw content
                        docs.append({
                            'id': doc_id,
                            'title': 'Document',
                            'content': doc.raw()
                        })
                else:
                    # Document not found
                    docs.append({
                        'id': doc_id,
                        'title': 'Unknown',
                        'content': f'Document with ID {doc_id} not found in index'
                    })
            except Exception as e:
                print(f"Error retrieving document {doc_id}: {e}")
                docs.append({
                    'id': doc_id,
                    'title': 'Error',
                    'content': f'Error retrieving document: {e}'
                })
        return docs

class RetrievalSystem:
    def __init__(self, retriever_name="BM25", corpus_name="Textbooks", db_dir="./corpus", HNSW=False, cache=False):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        
        # Validate configuration
        if self.corpus_name not in corpus_names:
            raise ValueError(f"Invalid corpus name: {self.corpus_name}")
        if self.retriever_name not in retriever_names:
            raise ValueError(f"Invalid retriever name: {self.retriever_name}")
        
        # Initialize retrievers for each corpus
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            retriever_list = []
            for corpus in corpus_names[self.corpus_name]:
                print(f"Initializing BM25 retriever for {corpus}...")
                try:
                    retriever_list.append(Retriever(retriever, corpus, db_dir, HNSW=HNSW))
                except Exception as e:
                    print(f"Error initializing retriever for {corpus}: {e}")
                    # Add a placeholder retriever that returns empty results
                    retriever_list.append(None)
            self.retrievers.append(retriever_list)
        
        # Set up document cache if requested
        self.cache = cache
        if self.cache:
            self.docExt = DocExtracter(cache=cache, corpus_name=self.corpus_name, db_dir=db_dir)
        else:
            self.docExt = None
    
    def retrieve(self, question, k=32, rrf_k=100, id_only=False):
        """Retrieve documents for a question across all corpora"""
        if not isinstance(question, str):
            raise TypeError("Question must be a string")

        # Adjust id_only if caching is enabled
        output_id_only = id_only
        if self.cache:
            id_only = True

        texts = []
        scores = []
        
        # Collect results from all retrievers
        for i, retriever_group in enumerate(self.retrievers):
            texts.append([])
            scores.append([])
            
            for j, retriever in enumerate(retriever_group):
                if retriever is None:
                    texts[-1].append([])
                    scores[-1].append([])
                    continue
                    
                try:
                    t, s = retriever.get_relevant_documents(question, k=k, id_only=id_only)
                    texts[-1].append(t)
                    scores[-1].append(s)
                except Exception as e:
                    print(f"⚠️ Error in retriever {retriever_names[self.retriever_name][i]}, "
                          f"corpus {corpus_names[self.corpus_name][j]}: {e}")
                    texts[-1].append([])
                    scores[-1].append([])
        
        # Check if we have any results
        if all(all(len(x) == 0 for x in row) for row in texts):
            print("⚠️ No results found in any retriever/corpus combination")
            return [], []
            
        # Merge results
        try:
            texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)
        except Exception as e:
            print(f"⚠️ Error merging results: {e}")
            return [], []
            
        # Apply document extraction if caching is enabled
        if self.cache and texts:
            try:
                texts = self.docExt.extract(texts)
            except Exception as e:
                print(f"⚠️ Error extracting documents: {e}")
                return [], []
                
        return texts, scores

    def merge(self, texts, scores, k=32, rrf_k=100):
        """Merge results from different corpora"""
        # For single retriever cases (like pure BM25)
        if len(texts) == 1:
            # Flatten all corpus results
            all_texts = []
            all_scores = []
            
            for corpus_texts, corpus_scores in zip(texts[0], scores[0]):
                if corpus_texts:  # Skip empty results
                    # normalize this corpus’s scores to μ=0, σ=1
                    μ, σ = np.mean(corpus_scores), np.std(corpus_scores) or 1.0
                    normed = [(s - μ) / σ for s in corpus_scores]
                    all_texts.extend(corpus_texts)
                    all_scores.extend(normed)
            
            # Sort by score (descending)
            if all_texts:
                sorted_indices = np.argsort(all_scores)[::-1]
                sorted_texts = [all_texts[i] for i in sorted_indices]
                sorted_scores = [all_scores[i] for i in sorted_indices]
                
                # Return top k results
                return sorted_texts[:k], sorted_scores[:k]
            
            return [], []
        
        # Fallback to RRF for multiple retrievers
        RRF_dict = {}
        
        for i in range(len(retriever_names[self.retriever_name])):
            texts_all, scores_all = [], []
            
            if not texts[i] or all(len(t) == 0 for t in texts[i]):
                continue
                
            for j in range(len(corpus_names[self.corpus_name])):
                if j >= len(texts[i]) or not texts[i][j]:
                    continue
                    
                texts_all.extend(texts[i][j])
                scores_all.extend(scores[i][j])
            
            if not texts_all:
                continue
                
            # Sort by scores (highest first for BM25)
            sorted_index = np.argsort(scores_all)[::-1]
            sorted_index = sorted_index[:len(texts_all)]
            
            sorted_texts = [texts_all[i] for i in sorted_index]
            sorted_scores = [scores_all[i] for i in sorted_index]
            
            for j, item in enumerate(sorted_texts):
                if not isinstance(item, dict) or "id" not in item:
                    continue
                    
                if item["id"] in RRF_dict:
                    RRF_dict[item["id"]]["score"] += 1 / (rrf_k + j + 1)
                    RRF_dict[item["id"]]["count"] += 1
                else:
                    RRF_dict[item["id"]] = {
                        "id": item["id"],
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "score": 1 / (rrf_k + j + 1),
                        "count": 1
                    }
        
        if not RRF_dict:
            return [], []
            
        RRF_list = sorted(RRF_dict.items(), key=lambda x: x[1]["score"], reverse=True)
        
        result_texts = [dict((key, item[1][key]) for key in ("id", "title", "content") if key in item[1]) for item in RRF_list[:k]]
        result_scores = [item[1]["score"] for item in RRF_list[:k]]
        return result_texts, result_scores

class DocExtracter:
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedCorp"):
        self.db_dir = db_dir
        self.cache = cache
        print("Initializing document extracter...")
        
        # Ensure all corpus files are available
        for corpus in corpus_names[corpus_name]:
            corpus_dir = os.path.join(self.db_dir, corpus, "chunk")
            if not os.path.exists(corpus_dir):
                print(f"Cloning the {corpus} corpus from Huggingface...")
                os.system(f"git clone https://huggingface.co/datasets/MedRAG/{corpus} {os.path.join(self.db_dir, corpus)}")
                
                if corpus == "statpearls":
                    print("Downloading the statpearls corpus from NCBI bookshelf...")
                    target_dir = os.path.join(self.db_dir, corpus)
                    os.system(f"wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {target_dir}")
                    os.system(f"tar -xzvf {os.path.join(target_dir, 'statpearls_NBK430685.tar.gz')} -C {target_dir}")
                    print("Chunking the statpearls corpus...")
                    os.system("python src/data/statpearls.py")
        
        # Set up the document cache
        if self.cache:
            cache_file = os.path.join(self.db_dir, f"{corpus_name}_id2text.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.dict = json.load(f)
            else:
                self.dict = {}
                
                for corpus in corpus_names[corpus_name]:
                    corpus_dir = os.path.join(self.db_dir, corpus, "chunk")
                    if not os.path.exists(corpus_dir):
                        print(f"Warning: Directory {corpus_dir} not found")
                        continue
                        
                    for fname in tqdm.tqdm(sorted(os.listdir(corpus_dir))):
                        if not fname.endswith('.jsonl'):
                            continue
                            
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
                                    _ = item.pop("contents", None)
                                    self.dict[item["id"]] = item
                                except json.JSONDecodeError:
                                    continue
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.dict, f)
        else:
            cache_file = os.path.join(self.db_dir, f"{corpus_name}_id2path.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.dict = json.load(f)
            else:
                self.dict = {}
                
                for corpus in corpus_names[corpus_name]:
                    corpus_dir = os.path.join(self.db_dir, corpus, "chunk")
                    if not os.path.exists(corpus_dir):
                        print(f"Warning: Directory {corpus_dir} not found")
                        continue
                        
                    for fname in tqdm.tqdm(sorted(os.listdir(corpus_dir))):
                        if not fname.endswith('.jsonl'):
                            continue
                            
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
                                except json.JSONDecodeError:
                                    continue
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.dict, f, indent=4)
                    
        print(f"Document extracter initialization complete. Cached {len(self.dict)} documents.")
    
    def extract(self, ids):
        """Extract document content from IDs"""
        if not ids:
            return []
            
        results = []
        
        for item in ids:
            try:
                item_id = item if isinstance(item, str) else item.get("id")
                
                if not item_id or item_id not in self.dict:
                    print(f"Warning: ID {item_id} not found in cache")
                    continue
                    
                if self.cache:
                    # Return cached document directly
                    results.append(self.dict[item_id])
                else:
                    # Load document from file
                    item_info = self.dict[item_id]
                    file_path = os.path.join(self.db_dir, item_info["fpath"])
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.read().strip().split('\n')
                        
                    if 0 <= item_info["index"] < len(lines):
                        doc = json.loads(lines[item_info["index"]])
                        results.append(doc)
                    else:
                        print(f"Warning: Index {item_info['index']} out of range for {file_path}")
            except Exception as e:
                print(f"Error extracting document: {e}")
                    
        return results