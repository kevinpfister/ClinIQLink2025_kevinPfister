import os
import re
import json
import tqdm
import torch
import transformers
from transformers import AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import sys
sys.path.append("src")
from utils import RetrievalSystem
from template import prompt_templates, system_prompt
from huggingface_hub import HfFolder, hf_hub_download, login, hf_hub_download, try_to_load_from_cache 
from llama_cpp import Llama
from dotenv import load_dotenv,dotenv_values
import re, json

def extract_valid_json(text: str) -> str:
    """Pull out the first {...} that actually contains an 'answer' or 'answer_choice'."""
    # this simple regex will only handle one‐level JSON, but should match your outputs:
    matches = re.findall(r'\{[^{}]+\}', text)
    for m in matches:
        try:
            obj = json.loads(m)
            if "answer" in obj or "answer_choice" in obj:
                return m
        except json.JSONDecodeError:
            continue
    return "{}"

def safe_json(response: str) -> dict:
    """Parse the JSON we extracted and normalize it into {'answer':…} or {'answer_choice':…}."""
    try:
        data = json.loads(response)
        if "answer" in data:
            return {"answer": data["answer"]}
        if "answer_choice" in data:
            return {"answer_choice": data["answer_choice"]}
        # if your LLM sometime wraps it in {"data": {"answer": …}}
        if "data" in data and isinstance(data["data"], dict) and "answer" in data["data"]:
            return {"answer": data["data"]["answer"]}
    except json.JSONDecodeError:
        pass
    return {}

class RAG:
    def __init__(self, rag=True, retriever_name="BM25", 
         corpus_name="MedCorp", db_dir="./corpus", 
         cache_dir=None, corpus_cache=False, HNSW=False, 
         llm_name="google/gemma-3-12b-it-qat-q4_0-gguf",
         model_type="local",  
         api_key=None,        
         hf_token=None):
         # Basic configuration.
        self.llm_name = llm_name
        self.model_type = model_type.lower()
        self.api_key = api_key
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        #Load environment variables from .env file
        config = dotenv_values(".env")
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAPI_KEY")  # OpenAI Key

    # Get HF token from environment variable or use provided token
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Only need llama_cpp for local models
        self.use_llama_cpp = "gguf" in self.llm_name.lower() and self.model_type == "local"

        # Initialize API clients if needed
        if self.model_type == "openai":
            # Check for OpenAI API key
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env file or pass as api_key parameter.")
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.api_key)
                print("Initialized OpenAI client for GPT-3.5-turbo-16k")
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
        
        elif self.model_type == "gemini":
            # Check for Google API key
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Google API key not provided. Set GOOGLE_API_KEY in .env file or pass as api_key parameter.")
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.gemini_client = genai
                print("Initialized Google Generative AI client for Gemini-2.0-flash")
            except ImportError:
                raise ImportError("Google Generative AI package not installed. Install with 'pip install google-generativeai'")
        
        elif self.model_type == "local":
            if not self.hf_token and "gemma" in self.llm_name.lower():
                raise ValueError("HuggingFace token not provided for Gemma model. Set HUGGINGFACE_TOKEN in .env or pass as hf_token.")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: 'local', 'openai', and 'gemini'")

        # Initialize the retrieval system if RAG is enabled.
        if self.rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
        else:
            self.retrieval_system = None

        # Set up prompt templates 
        self.templates = prompt_templates

        # Model-specific configuration for context lengths
        if self.model_type == "local" and "gemma-3" in self.llm_name.lower():
            self.max_length = 8192
            self.context_length = self.max_length - 3072   # leaves 3K tokens for the generated output
        elif self.model_type == "openai":
            self.max_length = 16384  # GPT-4o-mini context length
            self.context_length = 12000
        elif self.model_type == "gemini":
            self.max_length = 32768  # Gemini-2.0-flash context length
            self.context_length = 25000
        else:
            self.max_length = 2048
            self.context_length = 1024

        # Only load local model when using that option
        if self.model_type == "local":
            # Login to HuggingFace and cache token for local models
            if self.hf_token:
                login(self.hf_token)                
                HfFolder.save_token(self.hf_token)
        
            # Fix for Gemma model paths
            if "gemma" in self.llm_name.lower():
                # The correct repository ID for Gemma GGUF models
                repo_id = "google/gemma-3-12b-it-qat-q4_0-gguf"
                # The correct filename in the repo
                filename = "gemma-3-12b-it-q4_0.gguf"
            else:
                # For other models, use the provided path
                repo_id = self.llm_name
                filename = self.llm_name.split("/")[-1]
                if not filename.endswith(".gguf"):
                    filename = f"{filename}.gguf"
            
            print(f"Attempting to download {filename} from {repo_id}")
            
            # Check if model is already downloaded
            try:       
                # First check if model exists in cache
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
                    
            except Exception as e:
                print(f"Error checking/downloading model: {e}")
                
                # Fallback to local file if specified
                if os.path.exists(self.llm_name):
                    print(f"Using local model file: {self.llm_name}")
                    local_path = self.llm_name
                else:
                    raise RuntimeError(f"Failed to load model and no local fallback found: {e}")
            
            # Initialize llama-cpp model
            self.model = Llama(
                model_path=local_path,
                n_gpu_layers=-1,    # offload all layers
                n_batch=512,        # batch size for GPU decoding
                n_ctx=self.max_length,  # Use the model's max_length as context window
                verbose=True,       # prints device setup info
            )
            # For consistent API, we still need a tokenizer from HF
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-pt", cache_dir=self.cache_dir)
        
        # Set the answer function to the RAG-based answer generator.
        self.answer = self.rag_answer

    def custom_stop(self, stop_words, input_len=0):
        # Expects stop_words as a list of strings.
        return StoppingCriteriaList([CustomStoppingCriteria(stop_words, self.tokenizer, input_len)])

    def generate(self, messages, **kwargs):
        # —— OpenAI API branch ——
        if self.model_type == "openai":
            try:
                resp = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 1024),
                )
                
                if hasattr(resp, 'choices') and resp.choices and len(resp.choices) > 0:
                    return resp.choices[0].message.content.strip()
                else:
                    print(f"⚠️ [OpenAI API] unexpected response: {resp}")
                    return ""
            except Exception as e:
                print(f"⚠️ [OpenAI API] call failed: {e}")
                return ""
        
        # —— Google Gemini API branch ——
        elif self.model_type == "gemini":
            try:
                # Convert messages to Gemini format
                gemini_messages = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
                
                # Create model and generate response
                model = self.gemini_client.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(
                    gemini_messages,
                    generation_config={
                        "temperature": kwargs.get("temperature", 0.3),
                        "max_output_tokens": kwargs.get("max_tokens", 1024),
                    }
                )
                
                if response and hasattr(response, "text"):
                    return response.text.strip()
                else:
                    print(f"⚠️ [Gemini API] unexpected response: {response}")
                    return ""
            except Exception as e:
                print(f"⚠️ [Gemini API] call failed: {e}")
                return ""
        
        # —— llama-cpp chat branch (your existing code) ——
        elif self.use_llama_cpp:
            try:
                resp = self.model.create_chat_completion(
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.3),
                )
            except Exception as e:
                print(f"⚠️ [llama-cpp] API call failed: {e}")
                return ""
            # Must be a dict with a non-empty 'choices' list:
            if not isinstance(resp, dict):
                print(f"⚠️ [llama-cpp] unexpected response type: {type(resp)} → {resp}")
                return ""
            choices = resp.get("choices")
            if not isinstance(choices, list) or not choices:
                print(f"⚠️ [llama-cpp] empty or missing choices: {resp}")
                return ""
            first = choices[0] or {}
            # Look for both chat‐style and completion‐style outputs:
            content = (
                (first.get("message") or {}).get("content")
                or first.get("text")
                or ""
            )
            return content.strip()

        # —— HF text-generation branch (your existing code) ——
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        stopping = self.custom_stop(
            ["###", "User:", "\n\n\n"],
            input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True))
        )
        try:
            output = self.model(
                prompt,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=self.max_length,
                truncation=True,
                stopping_criteria=stopping,
                **kwargs
            )
        except Exception as e:
            print(f"⚠️ [HF pipeline] generation failed: {e}")
            return ""

        # Must be a non-empty list:
        if not isinstance(output, list) or not output:
            print(f"⚠️ [HF pipeline] unexpected output: {output}")
            return ""

        # Try each entry for known text fields:
        for entry in output:
            if not isinstance(entry, dict):
                continue
            if "generated_text" in entry:
                text = entry["generated_text"]
            elif "text" in entry:
                text = entry["text"]
            else:
                continue
            # Strip out the echoed prompt if present:
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()

        # If we never found anything useful:
        print(f"⚠️ [HF pipeline] no usable field found in {output}")
        return ""


    def create_direct_prompt(self, q):
        """
        Create a prompt in the exact style of the baseline code.
        This serves as a fallback when template rendering fails.
        """
        if q["type"] == "true_false":
            prompt = (
                f"Question: {q.get('question', 'No question provided')}\n"
                "Options:\n"
                "A. True\n"
                "B. False\n\n"
                "Please respond **only** with a single valid JSON object in the following format:\n"
                '{"answer": "True"}  ← if the answer is true\n'
                '{"answer": "False"} ← if the answer is false\n'
                "Do not include any other text or comments. Output must be strictly JSON."
            )
        elif q["type"] == "multiple_choice":
            # Convert options dict to formatted string
            if "options" in q and isinstance(q["options"], dict):
                options_str = "\n".join([f"{key}. {value}" for key, value in q["options"].items()])
            else:
                options_str = ""
                
            prompt = (
                f"Question: {q.get('question', 'No question provided')}\n"
                f"Options:\n{options_str}\n\n"
                "Respond strictly in valid JSON format as follows:\n"
                '{"answer_choice": "A"} ← if A is the answer\n'
                "Output only the JSON object. Do not include any explanation, commentary, markdown, or extra text."
            )
        elif q["type"] == "list":
            # Convert options list to formatted string
            if "options" in q and isinstance(q["options"], list):
                options_str = "\n".join([f"{idx+1}. {option}" for idx, option in enumerate(q["options"])])
            else:
                options_str = ""
                
            prompt = (
                f"Question: {q.get('question', 'No question provided')}\n"
                f"Options:\n{options_str}\n\n"
                "Respond strictly in valid JSON format as shown below:\n"
                '{"answer": ["1", "3"]} ← if options 1 and 3 are correct\n'
                "Only output the JSON object. Do not include explanations, labels, markdown, or any other text."
            )
        else:
            # Default to multiple choice format for unknown types
            prompt = f"Question: {q.get('question', 'No question provided')}\nRespond in JSON format with {{\"answer\": \"your answer\"}}"
            
        return prompt

    def rag_answer(self, question_data, k=5, rrf_k=85, save_dir=None, **kwargs):
        """
        RAG answer generation that uses the new prompt templates (from template.py) and supports the different input types.
        """
        # Make a copy of question_data to avoid modifying the original
        question_data = question_data.copy()
        
        # Check if essential fields exist
        if "question" not in question_data:
            question_data["question"] = "No question provided"
            
        # Format options based on question type
        if "options" in question_data:
            if isinstance(question_data["options"], dict):
                # For multiple choice, keep the original dict for direct prompt creation
                unformatted_options = question_data["options"].copy()
                # Format for template
                options_str = "\n".join([f"{key}. {value}" for key, value in question_data["options"].items()])
                question_data["options"] = options_str
            elif isinstance(question_data["options"], list):
                # For list questions, keep the original list for direct prompt creation
                unformatted_options = question_data["options"].copy()
                # Format for template
                options_str = "\n".join([f"{idx+1}. {option}" for idx, option in enumerate(question_data["options"])])
                question_data["options"] = options_str
            elif question_data["options"] is None:
                unformatted_options = None
                question_data["options"] = ""
        else:
            # If no options provided, set to empty
            unformatted_options = None
            question_data["options"] = ""
        
        # Build the retrieval query with options if available
        base_q = question_data.get("question", "")
        opts   = question_data.get("options", "").strip()
        if opts:
            retrieval_query = f"{base_q} Options: {opts}"
        else:
            retrieval_query = base_q

        # Initialize default values
        context_str = ""
        retrieved_snippets = []
        scores = []
        
        # Retrieve context via the RAG system if enabled
        if self.rag and self.retrieval_system is not None:
            try:
                retrieved_snippets, scores = self.retrieval_system.retrieve(
                    retrieval_query, k=k, rrf_k=rrf_k
                )
                # More defensive check - ensure returned values are valid
                if not isinstance(retrieved_snippets, list) or not isinstance(scores, list):
                    print("⚠️ Warning: invalid return values from retrieval")
                    retrieved_snippets, scores = [], []
            except IndexError as e:
                print(f"⚠️ Warning: retrieval index error: {e}")
                retrieved_snippets, scores = [], []
            except Exception as e:
                print(f"⚠️ Warning: retrieval failed: {e}")
                retrieved_snippets, scores = [], []
                
        # Safely create context string
        contexts = []
        for idx, snippet in enumerate(retrieved_snippets):
            if isinstance(snippet, dict) and "title" in snippet and "content" in snippet:
                contexts.append(f"Document {idx+1} (Title: {snippet['title']}): {snippet['content']}")
        
        context_str = "\n".join(contexts)
        question_data["context"] = context_str
        
        # First try to use the template system
        use_direct_prompt = False
        qtype = question_data.get("type", "multiple_choice")
        try:
            if qtype in self.templates:
                prompt = self.templates[qtype].render(**question_data)
            else:
                prompt = self.templates["multiple_choice"].render(**question_data)
        except Exception as e:
            print(f"Template rendering error: {e}. Falling back to direct prompt creation.")
            use_direct_prompt = True
        
        # If template rendering failed, use the direct prompt creation approach
        if use_direct_prompt:
            # Restore the original unformatted options for direct prompt creation
            original_q = question_data.copy()
            if unformatted_options is not None:
                original_q["options"] = unformatted_options
            prompt = self.create_direct_prompt(original_q)
        
        # Add context to the prompt if using RAG
        if self.rag and context_str:
            prompt = f"With your own knowledge and the help of the following document:\n\n{context_str}\n\n{prompt}"
        
        # Prepend the system prompt
        system_content = "You are a medical expert. Answer the question strictly in JSON format with no additional text."
        
        # Build the message list for the generation pipeline
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        # Log the prompt and retrieved snippets if a save directory is provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "prompt.txt"), "w") as f:
                f.write(prompt)
            with open(os.path.join(save_dir, "snippets.json"), "w") as f:
                json.dump(retrieved_snippets, f, indent=4)
        
        # Generate the answer using your generate method
        answer = self.generate(messages, **kwargs)
        
        # Log the generated answer if a save directory is provided
        if save_dir is not None:
            with open(os.path.join(save_dir, "response.txt"), "w") as f:
                f.write(answer)
            with open(os.path.join(save_dir, "response.json"), "w") as f:
                json.dump({"response": answer}, f, indent=4)
        
        return answer, retrieved_snippets, scores


    def rag_answer_textgrad(self, question_data, k=3, rrf_k=85, save_dir=None, prompt=None,step: int = 0, **kwargs):
        """
        RAG answer generation that uses the new prompt templates (from template.py) and supports the different input types.
        """
        print(f"PROMPT BEING USED FOR RAG {prompt}")
        # Make a copy of question_data to avoid modifying the original

        if prompt is None:
            question_data = question_data.copy()
        # Check if essential fields exist
            if "question" not in question_data:
                question_data["question"] = "No question provided"

            # Format options based on question type
            if "options" in question_data:
                if isinstance(question_data["options"], dict):
                    # For multiple choice, keep the original dict for direct prompt creation
                    unformatted_options = question_data["options"].copy()
                    # Format for template
                    options_str = "\n".join([f"{key}. {value}" for key, value in question_data["options"].items()])
                    question_data["options"] = options_str
                elif isinstance(question_data["options"], list):
                    # For list questions, keep the original list for direct prompt creation
                    unformatted_options = question_data["options"].copy()
                    # Format for template
                    options_str = "\n".join([f"{idx+1}. {option}" for idx, option in enumerate(question_data["options"])])
                    question_data["options"] = options_str
                elif question_data["options"] is None:
                    unformatted_options = None
                    question_data["options"] = ""
            else:
                # If no options provided, set to empty
                unformatted_options = None
                question_data["options"] = ""

            # Initialize default values
        context_str = ""
        retrieved_snippets = []
        scores = []

        prompt_head,prompt_tail = split_prompt_parts(prompt)
#----------------------------------------------------------------------------------------------important
        # Retrieve context via the RAG system if enabled
        print("FOR IF STATEMENT IF SELF.RAG")
        if self.rag and self.retrieval_system is not None:
            print("in RAG drin")
            try:
                """retrieved_snippets, scores = self.retrieval_system.retrieve(
                    question_data.get("question", ""), k=k, rrf_k=rrf_k
                )"""
                retrieved_snippets, scores = self.retrieval_system.retrieve(
                    prompt_head, k=3, rrf_k=rrf_k
                )
                # More defensive check - ensure returned values are valid
                if not isinstance(retrieved_snippets, list) or not isinstance(scores, list):
                    print("⚠️ Warning: invalid return values from retrieval")
                    retrieved_snippets, scores = [], []
            except IndexError as e:
                print(f"⚠️ Warning: retrieval index error: {e}")
                retrieved_snippets, scores = [], []
            except Exception as e:
                print(f"⚠️ Warning: retrieval failed: {e}")
                retrieved_snippets, scores = [], []

        # Safely create context string
        contexts = []
        for idx, snippet in enumerate(retrieved_snippets):
            if isinstance(snippet, dict) and "title" in snippet and "content" in snippet:
                print("Document added")
                contexts.append(f"Document {idx+1} (Title: {snippet['title']}): {snippet['content']}")

        context_str = "\n".join(contexts)
        #finished_prompt = context_str + prompt
#-------------------------------------------------------------------------------------------------------------------------------imbportant
            # First try to use the template system
        if prompt is None:
            use_direct_prompt = False
            qtype = question_data.get("type", "multiple_choice")
            try:
                if qtype in self.templates:
                    prompt = self.templates[qtype].render(**question_data)
                else:
                    prompt = self.templates["multiple_choice"].render(**question_data)
            except Exception as e:
                print(f"Template rendering error: {e}. Falling back to direct prompt creation.")
                use_direct_prompt = True

            # If template rendering failed, use the direct prompt creation approach
            if use_direct_prompt:
                # Restore the original unformatted options for direct prompt creation
                original_q = question_data.copy()
                if unformatted_options is not None:
                    original_q["options"] = unformatted_options
                prompt = self.create_direct_prompt(original_q)

            # Add context to the prompt if using RAG
        if self.rag and context_str:
            prompt = f"With your own knowledge and the help of the following document:\n\n{context_str}\n\n{prompt}"

        # Prepend the system prompt
        system_content = "You are a medical expert. Answer the question strictly in JSON format with no additional text."
        
        # Build the message list for the generation pipeline
        print(f"PROMPT BEING USED FOR RAG {prompt}")

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        # Log the prompt and retrieved snippets if a save directory is provided

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"prompt_{step}.txt"), "w") as f:
                f.write(prompt)
            with open(os.path.join(save_dir, f"snippets_{step}.json"), "w") as f:
                json.dump(retrieved_snippets, f, indent=4)
        
        # Generate the answer using your generate method

        answer = self.generate(messages, **kwargs)
        
        # Log the generated answer if a save directory is provided
        if save_dir is not None:
            with open(os.path.join(save_dir, f"response_{step}.txt"), "w") as f:
                f.write(answer)

            with open(os.path.join(save_dir, f"response_{step}.json"), "w") as f:
                json.dump({"response": answer}, f, indent=4)
        
        return answer, retrieved_snippets, scores, prompt


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_words = stop_words  # A list of strings to trigger stopping.
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Decode generated tokens (skip special tokens if desired)
        generated_tokens = self.tokenizer.decode(input_ids[0][self.input_len:], skip_special_tokens=True)
        return any(stop in generated_tokens for stop in self.stop_words)


def split_prompt_parts(prompt: str) -> tuple[str, str]:
    print(f"PROMPT WHICH IS BEING SPLITTET {prompt} ")
    match = re.search(r"(Respond.*)", prompt, re.DOTALL)
    if not match:
        match = re.search(r"(Please.*)", prompt, re.DOTALL)


    head = prompt[:match.start()].strip()
    if not head:
        raise ValueError("Options-Block not found.")
    tail = match.group(1).strip()
    return head, tail

