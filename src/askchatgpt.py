import os

from litellm import completion




def run_chatgpt_prompt(q: dict, save_dir_folder: str) -> str:
    prompt = create_prompt_from_question(q)

    os.makedirs(save_dir_folder, exist_ok=True)

    try:
        # Hole die Antwort von GPT
        print(f"→ Sende an GPT:\n{prompt}")
        response = completion(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )
        answer = response['choices'][0]['message']['content'].strip()

        # Speichere alles
        with open(os.path.join(save_dir_folder, "chatgpt_output.txt"), "w", encoding="utf-8") as f:
            f.write("== Prompt ==\n")
            f.write(prompt + "\n\n")
            f.write("== Answer ==\n")
            f.write(answer + "\n")

        return answer
    except Exception as e:
        error_path = os.path.join(save_dir_folder, "chatgpt_error.txt")
        with open(error_path, "w") as f:
            f.write(str(e))
        print(f"❌ Fehler bei GPT: {e}")
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
