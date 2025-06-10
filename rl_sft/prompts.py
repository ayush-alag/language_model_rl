OLD_WSD_PROMPT_FORMAT = """
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

WSD_PROMPT_FORMAT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
"""