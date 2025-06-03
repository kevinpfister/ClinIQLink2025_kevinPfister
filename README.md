

## ClinIQLink 2025: LLM Prompt Optimization for Medical QA
This project explores whether **TextGrad**, an automated prompt optimization technique, can improve accuracy and formatting of LLMs when answering closed-ended medical questions (multiple choice, list, true/false).

We evaluate TextGrad in two settings:

- A **standalone LLM**
- An **LLM with RAG Architecture**, which supplements the model with relevant documents from a medical corpus.

## Goals

- Integrate TextGrad into LLM and RAG pipelines.
- Measure its impact on answer accuracy across question types.
- Compare performance with and without prompt optimization.

We use OpenAI GPT models and a curated medical dataset. Prior work suggests TextGrad can boost zero-shot performance; we test if similar gains are possible in medical QA.


---


## Setup Instructions

> [!NOTE]  
> To run the TextGrad Library an API Key for OpenAI is needed. To run it on a different LLM. The LLM needs to be compatible with the LiteLLM Library  

### Clone Repository
```bash
git clone https://github.com/kevinpfister/ClinIQLink2025_kevinPfister.git
cd ClinIQLink2025_kevinPfister
```

### Create Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix or MacOS
venv\Scripts\activate     # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

To run the project use following command:

```bash
python src/test_rag_textgrad.py --limit 150
```

## Methodology

**Architecture of whole Project**
![Textgrad_architecture drawio](https://github.com/user-attachments/assets/26ee7161-4e6a-45b0-a695-16a46c71ff16)



**Architecture of TextGrad**
![fertiges_diagramm drawio](https://github.com/user-attachments/assets/611b5df6-c418-42f9-9efc-563296ed649a)


**Important Code parts**

This part focuses on interesting parts from the code. Since the code is a bit larger we only focus on certain lines to give an overall understanding of the code.
-  Getting First Answer either from LLM with RAG implementation or straight from LLM in test_rag_textgrad.py file

From LLM standalone
```bash
first_answer = run_chatgpt_prompt(q, q_save_dir)
```

From LLM with RAG implementation
```bash
first_answer, snippets, scores = rag_system.answer(q, save_dir=q_save_dir, **kwargs)
```
-  Give First Answer to TextGrad to classify answer or to enter TextGrad loop in test_rag_textgrad.py file
  ```bash
final_answer = refine_prompt_with_textgrad_from_example(q , rag_system=rag_system,save_dir_folder=q_save_dir, answer=first_answer,version="")
```

-  Define Classification Function and let the answer be checked in textgrad.py file
  ```bash
  classification_instruction = (
        f"Evaluate the model's answer to the prompt. with the original question {question_text} "
        "Reply ONLY with `1` if the answer is fully correct and exactly follows the expected JSON format. "
        "Otherwise, reply ONLY with `0`. No explanation."
    )
classifier_fn = tg.TextLoss(classification_instruction) 
classification_result = classifier_fn(answer_tg_object)


if str(classification_result.value).strip() == "1":
```
- If Answer is considered Wrong enter TextGrad Loop. Feedback function is defined and prompt should be optimized
```bash
  evaluation_instruction = (
        "You are a strict evaluator. Review the answer in the context of the prompt.\n"
        "Return ONE short sentence that starts with a category (e.g., 'Incorrect answer:', 'Invalid JSON:', etc.) "
        "followed by a brief explanation of what was wrong **and how the prompt could be improved** to avoid the issue.\n"
        "Do not include any extra commentary or polite language."
    )
  eval_loss_fn = tg.TextLoss(evaluation_instruction)
  evaluation = eval_loss_fn(evaluation_input)
  optimizer.step()
```
- Prompt optimizing function according to instructions
```bash
  system_prompt = (
            "You are an expert prompt engineer. "
            "Wrap the improved prompt strictly between these tags: <IMPROVED_VARIABLE> and </IMPROVED_VARIABLE>. "
            "Your task is to improve prompts. Return ONLY the improved prompt. "
            "Do not include any other text, comments, or explanations."
        )

        user_prompt = (
            f"Original prompt:\n{prompt_text}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Now return ONLY the improved prompt without anything in addition. Do not Include any Meta information like 'Question:' 'Revised Prompt' or any explanation."
        )
```
- Give new Prompt to RAG system or LLM standalone (based on version)
```bash
  if version == "RAG":
            print("ENTERING RAG")
            new_answer, _, _, _ = rag_system.rag_answer_textgrad(
                  question_data,
                  save_dir = save_dir_folder,
                  prompt = full_prompt,
                  step = step,
                  **kwargs
              )
        else:
            new_answer = run_chatgpt_prompt(full_prompt)
```



## Reproducibility

- **LLM Usage:**
  - The answering Model should be set to 'gpt-3.5-turbo-16k' (default).
  - The feedback Model should be set to 'gpt-4o' (default) 
- **Data:**
  - Benchmark_validation_enhanced_testset.json should be in the project folder (default)


---


## Log Description
Description of the most important log files
-  `questions.json` = log files for every question
-  `benchmark_scores.json`  = shows the results from the test(accuracy,f1 score etc.)


**In questions folder:**
- `failed_attempt_stepX_Question.txt` =`  
  Contains the following elements:
  - **Initial prompt**: First prompt given to the LLM. Includes the unchanged question and instruction.  
  - **Full prompt**: The prompt actually sent to the LLM (may differ due to TextGrad prompt refinement).  
  - **Answer**: The answer generated by the LLM.  
  - **Feedback**: Feedback provided by the Feedback LLM regarding the given answer.  
  - **Improved Full Prompt**: Refined prompt for the next iteration.
  - Due to several loops every Improved Prompt is the Full prompt for the next iteration loop. 
- `parsed_answer.json` = Contains Final Answer for autoevaluation
- `question.json` = Contains question with options and ground truth
- `snippets.json` = Contains content extracted from RAG which is then given to the LLM
- `prompt_X.txt` = Contains the full prompt with extracted Content from the RAG
- `parsed_answer.json` = final answer for evaluation

---

## Results & Evaluation

### **Baseline Results (Chatgpt 3.5 Turbot)**

**True/False**
- Accuracy: 0.64  
- Precision: 1.0  
- Recall: 0.64  
- F1-Score: 0.7805  

**Multiple Choice**
- Accuracy: 0.54  
- Precision: 0.54  
- Recall: 1.0  
- F1-Score: 0.7013  

**List**
- Accuracy: 0.6221  
- Precision: 0.8629  
- Recall: 0.6903  
- F1-Score: 0.7670

### TextGrad without RAG Results

**True/False**
- Accuracy: 0.6  
- Precision: 1.0  
- Recall: 0.6  
- F1-Score: 0.7500  

**Multiple Choice**
- Accuracy: 0.58  
- Precision: 0.58  
- Recall: 1.0  
- F1-Score: 0.7342  

**List**
- Accuracy: 0.6802  
- Precision: 0.8731  
- Recall: 0.7548  
- F1-Score: 0.8097

### TextGrad with RAG

**True/False**
- Accuracy: 0.44  
- Precision: 1.0  
- Recall: 0.44  
- F1-Score: 0.6111  

**Multiple Choice**
- Accuracy: 0.4  
- Precision: 0.4  
- Recall: 1.0  
- F1-Score: 0.5714  

**List**
- Accuracy: 0.5434  
- Precision: 0.8393  
- Recall: 0.6065  
- F1-Score: 0.7041    

TextGrad was able to outperform the Baseline when it comes to List questions and multiple choice. 
![Baseline_vs_Textgrad](https://github.com/user-attachments/assets/2c5afc03-4104-4262-a8ad-3178d29f69fb)


TextGrad was not able to show an improving performance on a RAG Architecture
![Textgrad_with_rag](https://github.com/user-attachments/assets/3ce07277-8fd4-49c2-8649-a17c4d6ab89e)

Further Informations and Evaluations can be found on the Report


---
## Team Contributions

| Name              | Contributions                                  |
|-------------------|------------------------------------------------|
| Kevin Pfister     | Textgrad Implementation. Preparing Dataset for final Testset |
| Zakaria Omarar     | Rag Implementation, Preparing Dataset for final Testset, Evaluation Script Implementation        |




---

## References

- [1] Yuksekgonul, M. et al. (2024). *ClinIQLink: A Benchmark for LLMs Detecting Factual Errors in Medical Q&A*. [PDF](https://arxiv.org/pdf/2406.07496.pdf)



---
