import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

with open('Benchmark_validation_testset.json', 'r') as file:
    questions = json.load(file)

from huggingface_hub import HfFolder, hf_hub_download, login
from llama_cpp import Llama
from getpass import getpass

class ModelManager:
    def __init__(self, model_type="local", llm_name="google/gemma-3-12b-it-qat-q4_0-gguf", api_key=None, hf_token=None):
        """
        Initialize the model manager with support for local and API-based models.
        
        Args:
            model_type (str): Type of model to use. Options: 
                - "local" (uses llama-cpp for local models)
                - "openai" (uses GPT-3.5 via API)
                - "gemini" (uses Gemini-2.0-flash via API)
            llm_name (str): Name of the local model to use (only for model_type="local")
            api_key (str): API key for OpenAI or Google (required when using those models)
            hf_token (str): HuggingFace token (required for local Gemma models)
        """
        self.model_type = model_type.lower()
        self.llm_name = llm_name
        self.api_key = api_key
        
        # Get HF token from environment variable or use provided token
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Initialize the appropriate model
        if self.model_type == "local":
            # Local model initialization
            if "gemma" in self.llm_name.lower() and not self.hf_token:
                raise ValueError("HuggingFace token not provided for Gemma model. Set HUGGINGFACE_TOKEN in .env or pass as hf_token.")
            
            # 1) log in & cache the token
            if self.hf_token:
                login(self.hf_token)                
                HfFolder.save_token(self.hf_token)
            
            # 2) download the .gguf file, passing the token
            if "gemma" in self.llm_name.lower():
                repo_id = "google/gemma-3-12b-it-qat-q4_0-gguf"
                filename = "gemma-3-12b-it-q4_0.gguf"
            else:
                repo_id = self.llm_name
                filename = self.llm_name.split("/")[-1]
                if not filename.endswith(".gguf"):
                    filename = f"{filename}.gguf"
            
            print(f"Downloading {filename} from {repo_id}")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=self.hf_token,
            )
            
            # 3) initialize the local model
            self.model = Llama(
                model_path=local_path,
                n_gpu_layers=-1,    # offload all layers
                n_batch=512,        # batch size for GPU decoding
                verbose=True,       # prints device setup info
            )
            print(f"Initialized local model: {llm_name}")
            
        elif self.model_type == "openai":
            # OpenAI API initialization
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env file or pass as api_key parameter.")
            
            try:
                from openai import OpenAI
                self.model = OpenAI(api_key=self.api_key)
                print("Initialized OpenAI client for GPT")
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
                
        elif self.model_type == "gemini":
            # Google Gemini API initialization
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Google API key not provided. Set GOOGLE_API_KEY in .env file or pass as api_key parameter.")
            
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key) # type: ignore
                self.model = genai
                print("Initialized Google Generative AI client for Gemini-2.0-flash")
            except ImportError:
                raise ImportError("Google Generative AI package not installed. Install with 'pip install google-generativeai'")
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: 'local', 'openai', and 'gemini'")
    
    def generate(self, messages, temperature=0.3, max_tokens=150):
        """
        Generate text using the selected model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            String containing the generated text
        """
        if self.model_type == "local":
            try:
                completion = self.model.create_chat_completion( # type: ignore
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return completion["choices"][0]["message"]["content"] # type: ignore
            except Exception as e:
                print(f"Error with local model: {e}")
                return ""
                
        elif self.model_type == "openai":
            try:
                response = self.model.chat.completions.create( # type: ignore
                    model="gpt-3.5-turbo-16k",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error with OpenAI API: {e}")
                return ""
                
        elif self.model_type == "gemini":
            try:
                # Convert messages to Gemini format
                gemini_messages = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
                
                # Generate response
                gen_model = self.model.GenerativeModel("gemini-2.0-flash") # type: ignore
                response = gen_model.generate_content(
                    gemini_messages,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                return response.text
            except Exception as e:
                print(f"Error with Gemini API: {e}")
                return ""


def extract_valid_json(text):
    matches = re.findall(r'\{[^{}]+\}', text)
    for m in matches:
        try:
            candidate = json.loads(m)
            if "answer" in candidate or "answer_choice" in candidate:
                return m
        except json.JSONDecodeError:
            continue
    return "{}"

def create_prompt(q: dict) -> str:
    """
    Create a zero-shot prompt for the given question.
    The prompt instructs the model to answer strictly in JSON format with no additional text.
    """
    if q["type"] == "true_false":
        print("true/false detected")
        prompt = (
            f"Question: {q['question']}\n"
            "Options:\n"
            "A. True\n"
            "B. False\n\n"
            "Please respond **only** with a single valid JSON object in the following format:\n"
            '{"answer": "True"}  ← if the answer is true\n'
            '{"answer": "False"} ← if the answer is false\n'
            "Do not include any other text or comments. Output must be strictly JSON."
        )
    elif q["type"] == "multiple_choice":
        print("multiple choice detected")
        options_str = "\n".join([f"{key}. {value}" for key, value in q["options"].items()])
        prompt = (
            f"Question: {q['question']}\n"
            f"Options:\n{options_str}\n\n"
            "Respond strictly in valid JSON format as follows:\n"
            '{"answer_choice": "A"} ← if A is the answer\n'
            "Output only the JSON object. Do not include any explanation, commentary, markdown, or extra text."
        )
    elif q["type"] == "list":
        options_str = "\n".join([f"{idx+1}. {option}" for idx, option in enumerate(q["options"])])
        prompt = (
            f"Question: {q['question']}\n"
            f"Options:\n{options_str}\n\n"
            "Respond strictly in valid JSON format as shown below:\n"
            '{"answer": ["1", "3"]} ← if options 1 and 3 are correct\n'
            "Only output the JSON object. Do not include explanations, labels, markdown, or any other text."
        )
    else:
        raise ValueError("Unknown question type.")
    return prompt

# Function to generate an answer using the model
def generate_answer(prompt: str, model_manager: ModelManager) -> dict:
    """
    Generate an answer from the model using the given prompt.
    Uses the model's chat completion interface with specified generation parameters.
    Extracts valid JSON from the output and returns the parsed Python dictionary.
    """
    print("\n--- Prompt ---\n", prompt, "\n")

    try:
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert. Answer the question strictly in JSON format with no additional text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Generate text using the model manager
        text = model_manager.generate(
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )

        print("Full model output:\n", text, "\n")

        # Use the helper to extract a valid JSON substring
        raw_json = extract_valid_json(text)
        print("Raw extracted JSON:\n", raw_json, "\n")

        parsed_answer = safe_json(raw_json)
        if not parsed_answer:
            print("Warning: Failed to parse valid JSON from the model response.")

        return parsed_answer

    except Exception as e:
        print(f"Error during generation: {e}")
        return {}

def safe_json(response: str):
    """
    Safely parse a JSON string into a Python object.
    Always returns a dict (possibly empty) if the expected keys are not found.
    """
    try:
        data = json.loads(response)
        if "answer" in data:
            return data
        elif "data" in data and isinstance(data["data"], dict) and "answer" in data["data"]:
            return {"answer": data["data"]["answer"]}
        elif "answer_choice" in data:
            return data
        else:
            # If no expected key found, return an empty dictionary.
            return {}
    except json.JSONDecodeError:
        return {}


# -----------------------------------------------------------------------------
# 1) PROCESSING: read the file, generate & parse every answer, collect (exp,pred)
# -----------------------------------------------------------------------------

def process_questions(file_path, model_manager):
    """
    Run through the entire benchmark, generate model answers, and
    return a dict of lists of (expected, predicted) pairs by question type.
    """
    with open(file_path, 'r') as f:
        questions = json.load(f)

    results = {
        "true_false":       [],  # list of (str, str)
        "multiple_choice":  [],  # list of (str, str)
        "list":             []   # list of (set[str], set[str])
    }

    for idx, q in enumerate(questions, 1):
        time.sleep(5)  # Rate limit for API calls
        print(f"Processing {idx}/{len(questions)}…", end="\r")
        prompt = create_prompt(q)
        answer = generate_answer(prompt, model_manager)

        if q["type"] == "true_false":
            exp = q["answer"].strip().lower()
            pred = answer.get("answer", "").strip().lower()
            results["true_false"].append((exp, pred))

        elif q["type"] == "multiple_choice":
            exp = q["correct_answer"].strip().upper()
            pred = answer.get("answer_choice", "").strip().upper()
            results["multiple_choice"].append((exp, pred))

        elif q["type"] == "list":
            exp_set  = {x.strip().lower() for x in q["answer"]}
            pred_set = {x.strip().lower() for x in answer.get("answer", [])}
            results["list"].append((exp_set, pred_set))

        else:
            print("  ⚠️  Unknown type:", q["type"])

    print("\nDone processing.")
    return results


# -----------------------------------------------------------------------------
# 2) METRIC COMPUTATION: take an existing `results` dict and compute scores
# -----------------------------------------------------------------------------

def compute_metrics(results_list, qtype):
    """
    Compute TP, FP, FN, precision, recall, F1, and accuracy for one question type.
    - Precision = TP / (TP + FP)
    - Recall    = TP / (TP + FN)
    - F1        = 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy  = TP / (TP + FP + FN)
    """
    TP = FP = FN = 0

    for expected, predicted in results_list:
        if qtype == "list":
            exp_set = set(expected)
            pred_set = set(predicted)
            TP += len(exp_set & pred_set)
            FP += len(pred_set - exp_set)
            FN += len(exp_set - pred_set)

        else:
            # no answer → FN
            if not predicted:
                FN += 1
                continue

            if qtype == "true_false":
                if predicted == expected:
                    TP += 1
                else:
                    FN += 1

            elif qtype == "multiple_choice":
                if predicted == expected:
                    TP += 1
                else:
                    FP += 1

    # metrics
    total     = TP + FP + FN
    accuracy  = TP / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)


    return {
        "TP":         TP,
        "FP":         FP,
        "FN":         FN,
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1_score":   f1,
    }


def compute_all_metrics(results):
    """
    Given `results = {"true_false": [...], "multiple_choice": [...], "list": [...]}`,
    return a dict of metrics-per-type (precision, recall, F1, accuracy).
    """
    scores = {}
    for qtype, res_list in results.items():
        scores[qtype] = compute_metrics(res_list, qtype)
    return scores

if __name__ == "__main__":
    file_path = 'Benchmark_validation_testset.json'
    
    # Choose the model type: "local", "openai", or "gemini"
    model_type = "openai"  # Change this to "openai" or "gemini" to use API models
    
    # Initialize the appropriate API key if using an API model
    api_key = None
    if model_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")  # Set this in your environment or .env file
    elif model_type == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")  # Set this in your environment or .env file
    
    # Initialize the model manager
    model_manager = ModelManager(
        model_type=model_type,
        llm_name="google/gemma-3-12b-it-qat-q4_0-gguf",
        api_key=api_key,
        hf_token=os.getenv("HUGGINGFACE_TOKEN")
    )

    # Record model name in the results for reference
    model_name = "Gemma-3-12B-it (local)"
    if model_type == "openai":
        model_name = "GPT-3.5 Turbo 16k (OpenAI API)"
    elif model_type == "gemini":
        model_name = "Gemini-2.0-flash (Google API)"
    print(f"Running benchmark with model: {model_name}")

    # 1) PROCESS (this is the 30 min step)
    results = process_questions(file_path, model_manager)
    
    # Add timestamp and model info to results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{timestamp}_{model_type}"

    # 2) TURN SETS → LISTS
    serializable = {
        "model_info": {
            "name": model_name,
            "type": model_type,
            "timestamp": timestamp
        }
    }
    
    for qtype, pairs in results.items():
        new_pairs = []
        for exp, pred in pairs:
            if isinstance(exp, set):
                exp = list(exp)
            if isinstance(pred, set):
                pred = list(pred)
            new_pairs.append((exp, pred))
        serializable[qtype] = new_pairs # type: ignore

    # 3) SAVE the JSON‑safe version
    with open(f'{output_prefix}_raw_results.json', 'w') as out:
        json.dump(serializable, out, indent=2)

    # 4) COMPUTE METRICS (fast)
    scores = compute_all_metrics(results)
    
    # Add model info to scores
    scores["model_info"] = serializable["model_info"]

    # 5) WRITE out scores
    with open(f'{output_prefix}_benchmark_scores.json', 'w') as out:
        json.dump(scores, out, indent=2)

    # 6) PRINT
    print(f"\nScores for {model_name}:")
    for qtype, m in scores.items():
        if qtype == "model_info":
            continue
        print(f"{qtype}:")
        print(f"  Accuracy:  {m['accuracy']:.3f}")
        print(f"  Precision: {m['precision']:.3f}")
        print(f"  Recall:    {m['recall']:.3f}")
        print(f"  F1 score:  {m['f1_score']:.3f}")