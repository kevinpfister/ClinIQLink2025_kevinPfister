import os
import re
import json
import sys
import argparse
from tqdm import tqdm
from datetime import datetime
import time

sys.path.append("src")
from medical_rag import create_medical_rag

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

def process_questions(questions, model, rag_system, save_dir=None, **kwargs):
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
            # Use the standard RAG method
            answer, snippets, scores = rag_system.answer(q, save_dir=q_save_dir, **kwargs)
            
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

def run_single_experiment(model, documents, file_path, base_output_dir, limit=None):
    """Run a single experiment configuration."""
    
    # Fixed parameters
    retriever = "MedCPT"
    corpus = "MedCorp"
    
    # Determine testset size identifier
    if "enhanced" in file_path.lower():
        num_questions = '150QA'
    else:
        num_questions = '30QA'
    
    # Create experiment-specific output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"rag_results_{retriever}_{num_questions}_{documents}_{corpus}_{model}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {model} | {documents} docs | {num_questions}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Save the run configuration
    config_data = {
        'benchmark_file': file_path,
        'output_dir': output_dir,
        'limit': limit,
        'retriever': retriever,
        'corpus': corpus,
        'model': model,
        'documents': documents,
        'testset_type': num_questions
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Load benchmark questions
    with open(file_path, 'r') as f:
        questions = json.load(f)
    
    # Limit questions if requested (for testing)
    if limit and limit > 0:
        questions = questions[:limit]
        print(f"Limited to {limit} questions for testing")
    
    print(f"Loaded {len(questions)} benchmark questions")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = create_medical_rag(
        llm_provider=model,
        use_rag=True,
        retriever_name=retriever,
        corpus_name=corpus,
        db_dir="./corpus"
    )
    
    # Process questions
    print("Processing benchmark questions...")
    results = process_questions(
        questions,
        model, 
        rag_system,
        save_dir=os.path.join(output_dir, "questions"),
        k=documents  # Pass the documents parameter as k
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
    
    with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # Compute and save metrics
    print("Computing metrics...")
    scores = compute_all_metrics(results)
    
    with open(os.path.join(output_dir, 'benchmark_scores.json'), 'w') as f:
        json.dump(scores, f, indent=2)
    
    # Print results summary
    print(f"\nResults for {model} | {documents} docs | {num_questions}:")
    for qtype, m in scores.items():
        print(f"  {qtype} ({len(results[qtype])} questions):")
        print(f"    Accuracy:  {m['accuracy']:.3f}")
        print(f"    Precision: {m['precision']:.3f}")
        print(f"    Recall:    {m['recall']:.3f}")
        print(f"    F1 score:  {m['f1_score']:.3f}")
    
    return output_dir, scores

if __name__ == "__main__":
    # Experimental setup parameters
    models = ["gemini", "gemma3", "openai"]
    document_counts = [3,15]
    testset_files = [
        # "./Benchmark_validation_testset.json",          # smaller testset (30QA)
        "./Benchmark_validation_enhanced_testset.json"  # bigger testset (150QA)
    ]
    
    parser = argparse.ArgumentParser(description='Run comprehensive RAG system evaluation')
    
    # Parameters
    parser.add_argument('--base_output_dir', type=str, default="./logs",
                        help='Base directory for all experiment results')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions per experiment (for testing)')
    parser.add_argument('--small_testset_only', action='store_true',
                        help='Run only on the smaller testset')
    parser.add_argument('--big_testset_only', action='store_true',
                        help='Run only on the bigger testset')
    parser.add_argument('--models', nargs='+', choices=['gemini', 'gemma3', 'openai'], 
                        default=models, help='Models to test')
    parser.add_argument('--documents', nargs='+', type=int, choices=[3, 5, 15], 
                        default=document_counts, help='Document counts to test')
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # Determine which testsets to use
    testsets_to_use = testset_files.copy()
    if args.small_testset_only:
        testsets_to_use = [testset_files[0]]  # only validation testset
    elif args.big_testset_only:
        testsets_to_use = [testset_files[1]]  # only full testset
    
    # Check if testset files exist
    valid_testsets = []
    for testset in testsets_to_use:
        if os.path.exists(testset):
            valid_testsets.append(testset)
        else:
            print(f"Warning: Testset file not found: {testset}")
    
    if not valid_testsets:
        print("Error: No valid testset files found!")
        sys.exit(1)
    
    # Calculate total number of experiments
    total_experiments = len(args.models) * len(args.documents) * len(valid_testsets)
    print(f"Starting comprehensive evaluation with {total_experiments} experiments")
    print(f"Models: {args.models}")
    print(f"Document counts: {args.documents}")
    print(f"Testsets: {[os.path.basename(ts) for ts in valid_testsets]}")
    
    # Store all results for summary
    all_results = []
    experiment_count = 0
    
    # Run all experiments
    for model in args.models:
        for documents in args.documents:
            for testset_file in valid_testsets:
                experiment_count += 1
                print(f"\n\nExperiment {experiment_count}/{total_experiments}")
                
                try:
                    output_dir, scores = run_single_experiment(
                        model=model,
                        documents=documents,
                        file_path=testset_file,
                        base_output_dir=args.base_output_dir,
                        limit=args.limit
                    )
                    
                    # Store result summary
                    testset_name = "150QA" if "enhanced" in testset_file.lower() else "30QA"
                    all_results.append({
                        "model": model,
                        "documents": documents,
                        "testset": testset_name,
                        "output_dir": output_dir,
                        "scores": scores
                    })
                    
                except Exception as e:
                    print(f"Error in experiment {experiment_count}: {e}")
                    continue
    
    # Save comprehensive summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.base_output_dir, f"comprehensive_results_summary_{timestamp}.json")
    
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments completed: {len(all_results)}/{total_experiments}")
    print(f"Summary saved to: {summary_file}")
    
    # Print overall summary table
    print(f"\nOverall Results Summary:")
    print(f"{'Model':<8} {'Docs':<5} {'Testset':<8} {'TF_F1':<6} {'MC_F1':<6} {'List_F1':<7}")
    print("-" * 50)
    
    for result in all_results:
        model = result["model"]
        docs = result["documents"]
        testset = result["testset"]
        scores = result["scores"]
        
        tf_f1 = scores.get("true_false", {}).get("f1_score", 0.0)
        mc_f1 = scores.get("multiple_choice", {}).get("f1_score", 0.0)
        list_f1 = scores.get("list", {}).get("f1_score", 0.0)
        
        print(f"{model:<8} {docs:<5} {testset:<8} {tf_f1:<6.3f} {mc_f1:<6.3f} {list_f1:<7.3f}")