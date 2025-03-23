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

Provide your answer in JSON format with the keys:
{"step_by_step_thinking": "explanation", "answer": "True/False"}
'''.strip())

# Multiple Choice template
multiple_choice_template = Template('''
Answer the following multiple-choice question.
Question: {{ question }}
Options:
{{ options }}

Provide your answer in JSON format with the keys:
{"step_by_step_thinking": "explanation", "answer_choice": "A/B/C/D"}
'''.strip())

# List question template
list_template = Template('''
Answer the following list question.
Question: {{ question }}
Options (if provided): {{ options }}

Provide your answer in JSON format with the keys:
{"step_by_step_thinking": "explanation", "answer_list": ["item1", "item2", ...]}
'''.strip())

# Short Answer template
short_answer_template = Template('''
Answer the following short answer question.
Question: {{ question }}

Provide your answer in JSON format with the keys:
{"step_by_step_thinking": "explanation", "answer": "your short answer"}
'''.strip())

# Short Inverse template
short_inverse_template = Template('''
Answer the following short inverse question.
Question: {{ question }}

A false answer was provided: {{ false_answer }}
and it was noted that: {{ incorrect_explanation }}

Provide the correct answer in JSON format with the keys:
{"step_by_step_thinking": "explanation", "answer": "your corrected short answer"}
'''.strip())

# Multi-Hop template
multi_hop_template = Template('''
Answer the following multi-hop question.
Question: {{ question }}

Please provide detailed step-by-step reasoning.
Output your answer in JSON format with the keys:
{"step_by_step_thinking": "detailed explanation", "answer": "your answer", "reasoning": ["step 1", "step 2", ...]}
'''.strip())

# Multi-Hop Inverse template
multi_hop_inverse_template = Template('''
Answer the following multi-hop inverse question.
Question: {{ question }}

The following incorrect reasoning steps have been flagged:
{{ incorrect_reasoning_step }}

Additionally, please consider the following correct reasoning context:
{{ reasoning }}

Provide your final answer in JSON format with the keys:
{"step_by_step_thinking": "detailed explanation including corrections", "answer": "your answer"}
'''.strip())

# A mapping for easier access in your main code.
prompt_templates = {
    "true_false": true_false_template,
    "multiple_choice": multiple_choice_template,
    "list": list_template,
    "short_answer": short_answer_template,
    "short_inverse": short_inverse_template,
    "multi_hop": multi_hop_template,
    "multi_hop_inverse": multi_hop_inverse_template
}
