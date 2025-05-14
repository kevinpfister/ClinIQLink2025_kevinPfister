import os
import re
import json
import sys
import argparse
from tqdm import tqdm
from datetime import datetime

# Import the RAG system
sys.path.append("src")
from RAG import RAG

def extract_valid_json(text):
    """Extract a valid JSON object from text that contains 'answer' or 'answer_choice'."""
    matches = re.findall(r'\{[^{}]+\}', text)
    for m in matches:
        try:
            candidate = json.loads(m)
            if "answer" in candidate or "answer_choice" in candidate:
                return m
        except json.JSONDecodeError:
            continue
    return "{}"

def safe_json(response):
    """Parse JSON response safely and ensure it has the expected format."""
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

def process_questions(questions, rag_system, iterative=False, save_dir=None, **kwargs):
    """Process all questions using the RAG system and collect expected vs predicted answers."""
    results = {
        "true_false": [],      # list of (str, str)
        "multiple_choice": [], # list of (str, str)
        "list": []             # list of (set[str], set[str])
    }
    
    for idx, q in enumerate(tqdm(questions, desc="Processing questions")):
        # Create question-specific output directory
        q_save_dir = None
        if save_dir:
            q_save_dir = os.path.join(save_dir, f"question_{idx+1}")
            os.makedirs(q_save_dir, exist_ok=True)
            
            # Save the original question
            with open(os.path.join(q_save_dir, "question.json"), "w") as f:
                json.dump(q, f, indent=2)
        
        try:
            # Use either standard or iterative RAG based on parameter
            if iterative:
                # Use the iterative RAG method
                save_path = os.path.join(q_save_dir, "iterative_conversation.json") if q_save_dir else None
                answer, messages = rag_system.i_rag_answer(
                    q, 
                    save_path=save_path,
                    **kwargs
                )
            else:
                # Use the standard RAG method
                answer, snippets, scores = rag_system.rag_answer(q, save_dir=q_save_dir, **kwargs)
            
            # Extract valid JSON from the response
            raw_json = extract_valid_json(answer)
            parsed_answer = safe_json(raw_json)
            
            # Save the parsed answer
            if q_save_dir:
                with open(os.path.join(q_save_dir, "parsed_answer.json"), "w") as f:
                    json.dump(parsed_answer, f, indent=2)
            
            # Process the answer based on question type
            if q["type"] == "true_false" and "answer" in q:
                exp = q["answer"].strip().lower()
                pred = parsed_answer.get("answer", "").strip().lower()
                results["true_false"].append((exp, pred))
                
            elif q["type"] == "multiple_choice" and "correct_answer" in q:
                exp = q["correct_answer"].strip().upper()
                pred = parsed_answer.get("answer_choice", "").strip().upper()
                results["multiple_choice"].append((exp, pred))
                
            elif q["type"] == "list" and "answer" in q:
                exp_set = {x.strip().lower() for x in q["answer"]}
                pred_set = {x.strip().lower() for x in parsed_answer.get("answer", [])}
                results["list"].append((exp_set, pred_set))
                
        except Exception as e:
            print(f"Error processing question {idx+1}: {e}")
            
            # Add a placeholder for failed questions if we have expected answers
            if q["type"] == "true_false" and "answer" in q:
                results["true_false"].append((q["answer"].strip().lower(), ""))
            elif q["type"] == "multiple_choice" and "correct_answer" in q:
                results["multiple_choice"].append((q["correct_answer"].strip().upper(), ""))
            elif q["type"] == "list" and "answer" in q:
                results["list"].append(({x.strip().lower() for x in q["answer"]}, set()))
    
    return results

def compute_metrics(results_list, qtype):
    """Compute TP, FP, FN, precision, recall, F1, and accuracy for one question type."""
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
    total = TP + FP + FN
    accuracy = TP / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

def compute_all_metrics(results):
    """Compute metrics for all question types."""
    scores = {}
    for qtype, res_list in results.items():
        if res_list:  # Only compute metrics if we have results for this type
            scores[qtype] = compute_metrics(res_list, qtype)
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RAG system on medical benchmark')
    
    # Input/output parameters
    parser.add_argument('--benchmark_file', type=str, default='./Benchmark_validation_testset.json',
                        help='Path to the benchmark file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: auto-generated with timestamp)')
    
    # System parameters
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions (for testing)')
    
    args = parser.parse_args()
    
    # Create an auto-generated output directory if none is specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./logs/rag_results_{timestamp}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the run configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load benchmark questions
    with open(args.benchmark_file, 'r') as f:
        questions = json.load(f)
    
    # Limit questions if requested (for testing)
    if args.limit and args.limit > 0:
        questions = questions[:args.limit]
        print(f"Limited to {args.limit} questions for testing")
    
    print(f"Loaded {len(questions)} benchmark questions")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = RAG()  # RAG initialization moved to its __init__ method
    
    # Process questions
    print("Processing benchmark questions...")
    results = process_questions(
        questions, 
        rag_system,
        iterative=False,
        save_dir=os.path.join(args.output_dir, "questions")
    )
    
    # Save raw results (converting sets to lists for JSON serialization)
    print("Saving raw results...")
    serializable = {}
    for qtype, pairs in results.items():
        new_pairs = []
        for exp, pred in pairs:
            if isinstance(exp, set):
                exp = list(exp)
            if isinstance(pred, set):
                pred = list(pred)
            new_pairs.append((exp, pred))
        serializable[qtype] = new_pairs
    
    with open(os.path.join(args.output_dir, 'raw_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # Compute and save metrics
    print("Computing metrics...")
    scores = compute_all_metrics(results)
    
    with open(os.path.join(args.output_dir, 'benchmark_scores.json'), 'w') as f:
        json.dump(scores, f, indent=2)
    
    # Print results summary
    print("\nBenchmark Results:")
    for qtype, m in scores.items():
        print(f"{qtype} ({len(results[qtype])} questions):")
        print(f"  Accuracy:  {m['accuracy']:.3f}")
        print(f"  Precision: {m['precision']:.3f}")
        print(f"  Recall:    {m['recall']:.3f}")
        print(f"  F1 score:  {m['f1_score']:.3f}")
