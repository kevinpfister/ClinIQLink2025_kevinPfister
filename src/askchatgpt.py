import os

from litellm import completion

from src.TEXTGRAD import create_prompt_from_question


def run_chatgpt_prompt(q: dict, save_dir_folder: str) -> str:
    prompt = create_prompt_from_question(q)

    os.makedirs(save_dir_folder, exist_ok=True)

    try:
        # Hole die Antwort von GPT
        print(f"→ Sende an GPT:\n{prompt}")
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )S
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
        print(f"❌ Fehler bei GPT: {e}")S
        return ""
