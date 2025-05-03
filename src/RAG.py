import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import prompt_templates, system_prompt


class RAG:
    def __init__(self, rag=True, retriever_name="SPECTER", 
             corpus_name="MedText", db_dir="./corpus", 
             cache_dir=None, corpus_cache=False, HNSW=False, 
             llm_name="aaditya/Llama3-OpenBioLLM-70B"):
        # Basic configuration.
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        # Initialize the retrieval system if RAG is enabled.
        if self.rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
        else:
            self.retrieval_system = None

        # Set up prompt templates 
        self.templates = prompt_templates

        # Model-specific configuration for Llama3.
        if "llama3" in self.llm_name.lower():
            self.max_length = 8192
            self.context_length = 7168
        elif self.llm_name.lower() == "gpt2":
            self.max_length = 1024
            self.context_length = 512  # adjust if needed
        else:
            self.max_length = 2048
            self.context_length = 1024

        # Load the tokenizer and text-generation pipeline.
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
        self.model = transformers.pipeline(
            "text-generation",
            model=self.llm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            model_kwargs={"cache_dir": self.cache_dir}
        )

        # Set the answer function to the RAG-based answer generator.
        self.answer = self.rag_answer

    def custom_stop(self, stop_words, input_len=0):
        # Expects stop_words as a list of strings.
        return StoppingCriteriaList([CustomStoppingCriteria(stop_words, self.tokenizer, input_len)])

    def generate(self, messages, **kwargs):
        # Convert the list of message dicts into a single prompt string.
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Optionally, if we use custom stopping criteria, set it up here.
        stopping_criteria = None
        input_len = len(self.tokenizer.encode(prompt, add_special_tokens=True))

        # Example: use a list of stop tokens.
        stop_words = ["###", "User:", "\n\n\n"]
        stopping_criteria = self.custom_stop(stop_words, input_len=input_len)

        # Generate the text using the text-generation pipeline.
        response = self.model(
            prompt,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_length,
            truncation=True,
            stopping_criteria=stopping_criteria,
            **kwargs
        )

        # Extract and return the text generated after the prompt.
        generated_text = response[0]["generated_text"][len(prompt):].strip()
        return generated_text



    def rag_answer(self, question_data, k=32, rrf_k=100, save_dir=None, **kwargs):
        """
        Generate an answer for various question types using RAG and log the prompt and response.
        
        Parameters:
        question_data (dict): Dictionary containing the question text, its type, and other relevant keys.
            Example keys:
                - "question": the question text.
                - "type": one of ["true_false", "multiple_choice", "list", "short_answer", 
                                "short_inverse", "multi_hop", "multi_hop_inverse"].
                - Additional keys as required by the specific template (e.g. "options", "false_answer", etc.).
        k, rrf_k: Retrieval parameters.
        save_dir (str): Optional directory to log the prompt and response.
        **kwargs: Additional parameters for generation.
        
        Returns:
        answer (str): The generated (and optionally simplified) answer.
        retrieved_snippets (list): List of retrieved document snippets.
        scores (list): Their corresponding scores.
        """
        # Retrieve context via the RAG system if enabled.
        if self.rag:
            retrieved_snippets, scores = self.retrieval_system.retrieve(question_data.get("question", ""), k=k, rrf_k=rrf_k)
            contexts = [
                f"Document {idx} (Title: {snippet['title']}): {snippet['content']}"
                for idx, snippet in enumerate(retrieved_snippets)
            ]
            context_str = "\n".join(contexts)
        else:
            retrieved_snippets, scores = [], []
            context_str = ""
        
        # Add the retrieved context to question_data for use in the prompt template.
        question_data["context"] = context_str
        
        # Select and render the appropriate prompt template based on question type.
        qtype = question_data.get("type", "multiple_choice")
        if qtype in self.templates:
            prompt = self.templates[qtype].render(**question_data)
        else:
            # Fallback to multiple choice template if type is unrecognized.
            prompt = self.templates["multiple_choice"].render(**question_data)
        
        # Prepend the system prompt (imported from template.py).
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Build the message list for the generation pipeline.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        # Log the prompt and retrieved snippets if a save directory is provided.
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "prompt.txt"), "w") as f:
                f.write(full_prompt)
            with open(os.path.join(save_dir, "snippets.json"), "w") as f:
                json.dump(retrieved_snippets, f, indent=4)
        
        # Generate the answer using your generate method.
        answer = self.generate(messages, **kwargs)
        
        # Log the generated answer if a save directory is provided.
        if save_dir is not None:
            with open(os.path.join(save_dir, "response.txt"), "w") as f:
                f.write(answer)
            with open(os.path.join(save_dir, "response.json"), "w") as f:
                json.dump({"response": answer}, f, indent=4)
        
        return answer, retrieved_snippets, scores

    def i_rag_answer(self, question_data, k=32, rrf_k=100, save_path=None, n_rounds=4, n_queries=3, qa_cache_path=None, **kwargs):
        """
        Iterative RAG answer generation (renamed from i_medrag_answer to i_rag_answer) that uses
        the new prompt templates (from template.py) and supports the different input types.

        Parameters:
        question_data (dict): Dictionary representing the question. For example:
            {
                "question": "What artery primarily supplies blood to the spleen?",
                "type": "multiple_choice",
                "options": {
                    "A": "Right gastric artery",
                    "B": "Splenic artery proper",
                    "C": "Left gastroepiploic artery",
                    "D": "Superior mesenteric artery"
                },
                ... (other keys for inverse or multi-hop types)
            }
        k, rrf_k: Retrieval parameters.
        save_path (str): Optional file path to log the conversation (prompts and responses).
        n_rounds (int): Number of iterative rounds to refine queries.
        n_queries (int): Number of queries to ask for in each round.
        qa_cache_path (str): Optional file path to cache iterative context.
        **kwargs: Additional parameters for generation.
        
        Returns:
        final_answer (str): The final generated answer.
        messages (list): The full conversation message history.
        """
        # Render the main prompt using the appropriate template based on question type.
        # (If the type is unrecognized, default to the multiple choice template.)
        qtype = question_data.get("type", "multiple_choice")
        if qtype in prompt_templates:
            main_prompt = prompt_templates[qtype].render(**question_data)
        else:
            main_prompt = prompt_templates["multiple_choice"].render(**question_data)
        
        # Use the rendered main prompt as the base.
        QUESTION_PROMPT = main_prompt

        # Initialize context from cache if provided.
        context = ""
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            with open(qa_cache_path, 'r') as f:
                qa_cache = json.load(f)[:n_rounds]
            if qa_cache:
                context = qa_cache[-1]
            n_rounds = n_rounds - len(qa_cache)
        
        last_context = ""
        max_iterations = n_rounds + 3
        # Start conversation with the system prompt.
        saved_messages = [{"role": "system", "content": system_prompt}]
        
        # Define inline follow-up instructions (for iterative clarification).
        follow_up_ask = f"Please analyze the above information and provide {n_queries} concise queries for further clarification."
        follow_up_answer = "Based on all the gathered information, please provide your final answer in JSON format with the appropriate keys."

        for i in range(max_iterations):
            # Build the user prompt based on iteration stage.
            if i < n_rounds:
                if not context:
                    user_msg = f"{QUESTION_PROMPT}\n\n{follow_up_ask}"
                else:
                    user_msg = f"{context}\n\n{QUESTION_PROMPT}\n\n{follow_up_ask}"
            else:
                user_msg = f"{context}\n\n{QUESTION_PROMPT}\n\n{follow_up_answer}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
            saved_messages.append(messages[-1])
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(saved_messages, f, indent=4)
            
            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(saved_messages, f, indent=4)
            
            # Check if the response appears to contain a final answer.
            if i >= n_rounds and (('"answer":' in last_content) or ('"answer_choice":' in last_content)):
                messages.append(response_message)
                final_prompt = "Output the answer in JSON: {'answer': your_answer}"
                messages.append({"role": "user", "content": final_prompt})
                saved_messages.append(messages[-1])
                final_answer = self.generate(messages, **kwargs)
                final_response_message = {"role": "assistant", "content": final_answer}
                messages.append(final_response_message)
                saved_messages.append(final_response_message)
                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump(saved_messages, f, indent=4)
                return final_answer, messages
            
            # Otherwise, if the response contains a queries section, parse and process the queries.
            elif "queries" in last_content.lower():
                # Expect the queries to be output in a JSON format like: {"output": ["query1", "query2", ...]}
                try:
                    m = re.search(r'["\']output["\']\s*:\s*(\[[^\]]*\])', last_content)
                    if m:
                        query_list_str = m.group(1)
                        query_list = eval(query_list_str)
                    else:
                        query_list = []
                except Exception as e:
                    print("Error parsing queries:", e)
                    query_list = []
                
                for query in query_list:
                    if not query.strip():
                        continue
                    try:
                        # For each extracted query, call the non-iterative rag_answer.
                        sub_question_data = {"question": query, "type": "short_answer"}
                        rag_result, _, _ = self.rag_answer(sub_question_data, k=k, rrf_k=rrf_k, **kwargs)
                        context += f"\n\nQuery: {query}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as e:
                        print("Error during query processing:", e)
                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, 'w') as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer detected. Continuing to next iteration.")
                continue
        
        # If no final answer is produced within the iterations, return the last output.
        return messages[-1]["content"], messages


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