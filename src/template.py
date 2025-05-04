from liquid import Template

# System prompt (if needed)
system_prompt = "You are a helpful medical expert."

# True/False template
true_false_template = Template('''
Answer the following true/false question.
Question: {{ question }}
Options:
A. True
B. False

Please respond **only** with a single valid JSON object in the following format:
{"answer": "True"}  ← if the answer is true
{"answer": "False"} ← if the answer is false
Do not include any other text or comments. Output must be strictly JSON.
'''.strip())

# Multiple Choice template
multiple_choice_template = Template('''
Answer the following multiple-choice question.
Question: {{ question }}
Options:
{{ options }}

Respond strictly in valid JSON format as follows:
{"answer_choice": "A"} ← if A is the answer
Output only the JSON object. Do not include any explanation, commentary, markdown, or extra text.                                    
'''.strip())

# List question template
list_template = Template('''
Answer the following list question.
Question: {{ question }}
Options:
{{ options }}

Respond strictly in valid JSON format as shown below:
{"answer": ["1", "3"]} ← if options 1 and 3 are correct
Only output the JSON object. Do not include explanations, labels, markdown, or any other text.
'''.strip())



# A mapping for easier access in your main code.
prompt_templates = {
    "true_false": true_false_template,
    "multiple_choice": multiple_choice_template,
    "list": list_template
}
