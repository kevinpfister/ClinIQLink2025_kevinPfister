

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
> To run the TextGrad Library an API Key for OpenAI is needed. To run it on a different LLM. The LLM needs to be compatibility with the LiteLLM Library  

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



---

## Reproducibility

- **LLM Usage:**
  - The answering Model should be set to 'gpt-3.5-turbo-16k' (default).
  - The feedback Model should be set to 'gpt-4o' (default) 
- **Data:**
  - Benchmark_validation_enhanced_testset.json should be in the project folder (default)


---

## Team Contributions

| Name              | Contributions                                  |
|-------------------|------------------------------------------------|
| Kevin Pfister     | Textgrad Implementation. Preparing Dataset for final Testset |
| Zakaria Omarar     | Rag Implementation, Preparing Dataset for final Testset, Evaluation Script Implementation        |




---

## Results & Evaluation

- [Briefly summarize your evaluation metrics, improvements from baseline, and insights drawn from experiments.]
- All detailed results are documented in `metrics/firstResults.json`.

---

## References

[List here any relevant papers, sources, libraries, or resources used in your project.]

- Doe, J. (2024). *Great NLP Paper*. Conference Name.
- [Library Used](https://example-library.com)

---
