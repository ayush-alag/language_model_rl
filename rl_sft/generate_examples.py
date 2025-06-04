import random
import json

class Add:
    def __init__(self):
        self.operator = "+"

    def __str__(self):
        return self.operator

    def __call__(self, left, right):
        return left + right

class Sub:
    def __init__(self):
        self.operator = "-"

    def __str__(self):
        return self.operator

    def __call__(self, left, right):
        return left - right

class Mul:
    def __init__(self):
        self.operator = "*"

    def __str__(self):
        return self.operator

    def __call__(self, left, right):
        return left * right

class Div:
    def __init__(self):
        self.operator = "/"

    def __str__(self):
        return self.operator

    def __call__(self, left, right):
        return left / right

    def __can_divide__(self, left, right):
        return right != 0 and left % right == 0

# return expr and target
def build_partial_expression(nums):
    if len(nums) == 1:
        return str(nums[0]), nums[0]

    split = random.randint(1, len(nums) - 1)
    left_nums = nums[:split]
    right_nums = nums[split:]
    left_expr, left_target = build_partial_expression(left_nums)
    right_expr, right_target = build_partial_expression(right_nums)

    operator = random.choice([Add(), Sub(), Mul(), Div()])
    if operator.operator == "/":
        if not operator.__can_divide__(left_target, right_target):
            operator = random.choice([Add(), Sub(), Mul()])

    target = operator(left_target, right_target)
    return f"({left_expr} {operator} {right_expr})", target

def generate_examples(num_numbers, num_examples, max_number, max_target):
    examples = []
    for i in range(len(num_numbers)):
        for j in range(num_examples[i]):
            nums = [random.randint(1, max_number) for _ in range(num_numbers[i])]
            target = None
            while target is None or abs(target) > max_target:
                expression, target = build_partial_expression(nums)
            # examples.append({"nums": nums, "target": target, "expression": expression})
            examples.append({"nums": nums, "target": target, "chain_of_thought": "", "answer": ""})
    return examples

def write_examples(examples, file_name):
    with open(file_name, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    # num_numbers = [2, 5, 6]
    num_numbers = [2]
    num_examples = [200]
    # num_examples = [200, 100, 100]
    # num_numbers = [5]
    # num_examples = [5]
    max_number = 100
    max_target = 200
    examples = generate_examples(num_numbers, num_examples, max_number, max_target)
    write_examples(examples, "synthetic_examples_to_fill.json")