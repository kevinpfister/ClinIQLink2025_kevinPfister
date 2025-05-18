import textgrad as tg
import httpx
from textgrad.engine_experimental.litellm import LiteLLMEngine
import os
import logging
from textgrad.variable import Variable
from textgrad.engine_experimental.litellm import LiteLLMEngine
import re
from litellm import completion


from src.RAG import RAG
from src.askchatgpt import run_chatgpt_prompt

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAPI_KEY") #OpenAI Key
tg.set_backward_engine("gpt-4o", override=True)

class PromptImprovingEngine:
    def __init__(self, model_name="gpt-4o"):
        self.llm = LiteLLMEngine(model_string=model_name)

    def __call__(self, variable, feedback: str = "", **kwargs):
        prompt_text = variable.value if hasattr(variable, "value") else str(variable)
        print(f"PROMPT TEXT IN ENGINE {prompt_text}")
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

        raw_output = self.llm(content=user_prompt, system_prompt=system_prompt).strip()

        # Entferne z. B. "Revised prompt:" falls der LLM es trotzdem hinzufügt

        role = variable.role_description if hasattr(variable, "role_description") else "Improved prompt"

        print(f"FINAL RETURN TO OPTIMIZER:\n{raw_output}")

        if "<IMPROVED_VARIABLE>" in raw_output and "</IMPROVED_VARIABLE>" in raw_output:
            return raw_output
        else:
            return f"<IMPROVED_VARIABLE>{raw_output}</IMPROVED_VARIABLE>"


engine = PromptImprovingEngine()


def refine_prompt_with_textgrad_from_example(
    q: dict = None,
    prompt: str = None,
    answer: str = "",
    question_text: str = "",
    rag_system=None,
    question_data=None,
    save_dir=None,
    kwargs=None,
    max_steps: int = 5,
    save_dir_folder=None,

):

    if q is not None:
        prompt = create_prompt_from_question(q)
        question_text = q.get("question", "")

    if kwargs is None:
        kwargs = {}
    # TextGrad Variablen
    logging.getLogger("textgrad").disabled = True

    if answer == "":
        #answer = ask_question_direct(rag_system.model,prompt)
        answer =run_chatgpt_prompt(full_prompt=prompt)
        print(f"ERSTE ANTanswer{answer}")

    # === Save initial prompt and answer ===
    if save_dir_folder:
        os.makedirs(save_dir_folder, exist_ok=True)
        with open(os.path.join(save_dir_folder, f"initial_prompt_answer.txt"), "w", encoding="utf-8") as f:
            f.write("== Initial Prompt ==\n")
            f.write(prompt.strip() + "\n\n")
            f.write("== Initial Answer ==\n")
            f.write(answer.strip() + "\n")

    prompt_tg_object = tg.Variable(
        prompt,
        requires_grad=True,
        role_description="Prompt that can be improved"
    )
    answer_tg_object = tg.Variable(
        answer,
        requires_grad=False,
        role_description="generated answer"
    )

    evaluation_input = tg.Variable(
        f"Prompt:\n{prompt_tg_object.value}\n\nAnswer:\n{answer_tg_object.value}",
        requires_grad=False,  # Wichtig! Nur prompt_tg_object wird optimiert
        role_description="Evaluation input containing prompt and answer"
    )

    classification_instruction = (
        f"Evaluate the model's answer to the prompt. with the original question {question_text} "
        "Reply ONLY with `1` if the answer is fully correct and exactly follows the expected JSON format. "
        "Otherwise, reply ONLY with `0`. No explanation."
    )

    classifier_fn = tg.TextLoss(classification_instruction)



    evaluation_instruction = (
        "You are a strict evaluator. Review the answer in the context of the prompt.\n"
        "Return ONE short sentence that starts with a category (e.g., 'Incorrect answer:', 'Invalid JSON:', etc.) "
        "followed by a brief explanation of what was wrong **and how the prompt could be improved** to avoid the issue.\n"
        "Do not include any extra commentary or polite language."
    )

    print(evaluation_instruction)

    eval_loss_fn = tg.TextLoss(evaluation_instruction)

    #optimizer = tg.TGD(parameters=[prompt_tg_object])


    prompt_head, prompt_tail = split_prompt_parts(prompt)

    prompt_tg_object = tg.Variable(
        prompt_head,
        requires_grad=True,
        role_description="Prompt that can be improved"
    )

    optimizer = tg.TextualGradientDescent(engine=engine, parameters=[prompt_tg_object])


    for step in range(3):
        print(f"SCHRITT {step}")

        if save_dir_folder:
            file_safe_prompt = re.sub(r'[^\w\-_. ]', '_', prompt[:50])
            fail_path = os.path.join(save_dir_folder, f"failed_attempt_step{step + 1}_{file_safe_prompt}.txt")

        classification_result = classifier_fn(answer_tg_object)


        if str(classification_result.value).strip() == "1":
            print("Answer is correct, optimization is being stopped")
            print(f"ANSWER: {answer_tg_object.value}")
            return answer_tg_object.value

        print(f"WRONG CONSIDERED ANSWER {answer_tg_object.value}")
        evaluation = eval_loss_fn(evaluation_input)
        feedback_str = str(evaluation.value)

        print(f"GPT Feedback {evaluation}")
        print(f"DEBUG FEEDBACK: {feedback_str}")



        full_prompt_to_write = prompt if step == 0 else rebuild_prompt(prompt_tg_object.value, prompt_tail)

        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("== Original Prompt ==\n")
            f.write(prompt.strip() + "\n\n")
            f.write("== Full Prompt ==\n")
            f.write(full_prompt_to_write.strip() + "\n")
            f.write("== Answer ==\n")
            f.write(answer_tg_object.value.strip() + "\n\n")
            f.write("== Feedback ==\n")
            f.write(feedback_str.strip() + "\n\n")



        # Jetzt erst den Optimierungsschritt durchführen
        optimizer.step()
        print(f"AFTER STEP – UPDATED prompt_tg_object.value:\n{prompt_tg_object.value}")


        print(f"AFTER STEP – UPDATED prompt_tg_object.value:\n{prompt_tg_object.value}")

        full_prompt = rebuild_prompt(prompt_tg_object.value, prompt_tail)

        if save_dir_folder:
            with open(fail_path, "a", encoding="utf-8") as f:
                f.write("== Improved Full Prompt ==\n")
                f.write(full_prompt.strip() + "\n\n")


        evaluation_input.value = f"Prompt:\n{full_prompt}\n\nAnswer:\n{answer_tg_object.value}"



        print(f"NEW FULL PROMPT {full_prompt}")

        #print(f"RAG WIRD AUFGERUFEN")
        # Neue Antwort mit optimiertem Prompt generieren
        """new_answer, _, _, _ = rag_system.rag_answer_textgrad(
            question_data,
            save_dir = save_dir_folder,
            prompt = full_prompt,
            step = step,
            **kwargs
        )"""
        #new_answer = ask_question_direct(rag_system.model,full_prompt) #FOR GEMMA
        new_answer = run_chatgpt_prompt(full_prompt)
        #print(f"ANTWORT CVON GEMMA {new_answer}")
        print(f"ANSWER FROM CHATGPT RAG{new_answer}")
        answer_tg_object = tg.Variable(new_answer, requires_grad=False, role_description="updated generated answer")

    print(f"NO PERFECT ANSWER COULD BE FOUND")
    return answer_tg_object.value


def split_prompt_parts(prompt: str) -> tuple[str, str]:
    match = re.search(r"(Options:.*)", prompt, re.DOTALL)
    if not match:
        raise ValueError("Options-Block nicht gefunden.")

    head = prompt[:match.start()].strip()
    tail = match.group(1).strip()
    return head, tail

def rebuild_prompt(new_prompt: str, preserved_tail: str) -> str:
    return f"{new_prompt.strip()}\n\n{preserved_tail.strip()}"

def ask_question_direct(model, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
    try:
        result = model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["###", "User:", "\n\n\n"]
        )
        return result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"❌ Fehler bei der Generierung: {e}")
        return ""



def create_prompt_from_question(question_data: dict) -> str:
    """Erstellt den Prompt je nach Fragetyp: true_false, multiple_choice, list."""
    question = question_data.get("question", "No question provided")
    qtype = question_data.get("type", "").lower()

    if qtype == "true_false":
        return (
            f"Question: {question}\n"
            "Options:\n"
            "A. True\n"
            "B. False\n\n"
            "Please respond strictly with one of the following JSON formats:\n"
            '{"answer": "True"}\n'
            'or\n'
            '{"answer": "False"}\n\n'
            "Please respond **only** with a single valid JSON object in the following format:\n"
                '{"answer": "True"}  ← if the answer is true\n'
                '{"answer": "False"} ← if the answer is false\n'
                "Do not include any other text or comments. Output must be strictly JSON."
        )

    elif qtype == "multiple_choice":
        options = question_data.get("options", {})
        if isinstance(options, dict):
            options_str = "\n".join(f"{k}. {v}" for k, v in options.items())
        else:
            options_str = ""
        return (
            f"Question: {question}\n"
            f"Options:\n{options_str}\n\n"
            "Respond strictly in valid JSON format as follows:\n"
                '{"answer_choice": "A"} ← if A is the answer\n'
                "Output only the JSON object. Do not include any explanation, commentary, markdown, or extra text."
        )

    elif qtype == "list":
        options = question_data.get("options", [])
        if isinstance(options, list):
            options_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
        else:
            options_str = ""
        return (
            f"Question: {question}\n"
            f"Options:\n{options_str}\n\n"
             "Respond strictly in valid JSON format as shown below:\n"
                '{"answer": ["1", "3"]} ← if options 1 and 3 are correct\n'
                "Only output the JSON object. Do not include explanations, labels, markdown, or any other text."
        )

    else:
        return (
            f"Question: {question}\n\n"
            "Respond strictly in JSON like this:\n"
            '{"answer": "your_answer"}\n\n'
            "Only output the JSON object. No explanation."
        )

def run_chatgpt_prompt(full_prompt) -> str:

        # Hole die Antwort von GPT
        print(f"→ Sende an GPT:\n{full_prompt}")
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=512,
            temperature=0.3
        )
        answer = response['choices'][0]['message']['content'].strip()

        return answer

