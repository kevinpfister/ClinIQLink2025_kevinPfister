# test_rag.py

import os
import json
from RAG import RAG

def simple_chat_template(messages, tokenize=False, add_generation_prompt=True):
    """
    A simple replacement for the tokenizer's apply_chat_template method.
    Concatenates the contents of all messages.
    """
    return "\n".join(message["content"] for message in messages)

def optimize_prompt(text):
    """
    Optimize the given text using TextGrad's prompt optimizer.
    
    This function attempts to import PromptOptimizer from textgrad.optimizer.optimizer_prompts.
    If that fails, it prints an error and returns the original text.
    """
    try:
        from textgrad.optimizer.optimizer_prompts import PromptOptimizer
    except ImportError as e:
        print("PromptOptimizer not found in textgrad.optimizer.optimizer_prompts:", e)
        return text

    try:
        optimizer = PromptOptimizer()
        optimized_text = optimizer.optimize(text)
        return optimized_text
    except Exception as e:
        print("TextGrad prompt optimization error:", e)
        return text

def dummy_retrieve(question, k=32, rrf_k=100):
    """
    A dummy retrieval function that returns a single dummy snippet.
    This forces the RAG branch to be used.
    """
    dummy_snippet = {
        "title": "Dummy Document",
        "content": "This is a dummy context relevant to the question."
    }
    return [dummy_snippet], [1.0]

def main():
    # Instantiate the RAG system with a small model (gpt2) and rag flag enabled.
    rag_system = RAG(rag=True, llm_name="medicalai/ClinicalBERT", use_textgrad=False)
    
    # Override the tokenizer's apply_chat_template method for testing.
    rag_system.tokenizer.apply_chat_template = simple_chat_template

    # Set max_length explicitly for GPT-2.
    rag_system.max_length = 1024

    # Monkey-patch the retrieval system to use the dummy retrieval function.
    rag_system.retrieval_system.retrieve = dummy_retrieve

    # Define a sample multiple-choice question input.
    question_data = {
        "question": "What artery primarily supplies blood to the spleen?",
        "type": "multiple_choice",
        "options": {
            "A": "Right gastric artery",
            "B": "Splenic artery proper",
            "C": "Left gastroepiploic artery",
            "D": "Superior mesenteric artery"
        }
    }
    
    # Call the rag_answer method.
    answer, retrieved_snippets, scores = rag_system.rag_answer(question_data, save_dir="logs")
    
    # Print the results.
    print("Final Answer:")
    print(answer)
    print("\n\nRetrieved Snippets:")
    print(json.dumps(retrieved_snippets, indent=2))
    print("\n\nScores:")
    print(scores)
    
    # # Additionally, test the prompt optimizer separately.
    # sample_prompt = "Write a detailed answer to the following question: What is the capital of France?"
    # optimized = optimize_prompt(sample_prompt)
    # print("\nOriginal Prompt:")
    # print(sample_prompt)
    # print("\nOptimized Prompt:")
    # print(optimized)

if __name__ == "__main__":
    main()
