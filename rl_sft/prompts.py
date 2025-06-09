WSD_PROMPT_FORMAT = """
User: Using the numbers {numbers}, create an equation that equals {target}.
You may use only the basic operations +, -, *, /, and each number exactly once.
You must start your answer with "Assistant: "!
First, show your reasoning inside <think>…</think> tags.
Then, on its own final line, output ONLY the equation wrapped in <answer>…</answer> tags.
For example:
Assistant:
<think>74 - 45 = 29</think>
<answer>(29 * 19) - 9</answer>
"""