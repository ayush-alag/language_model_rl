from collections import defaultdict
from datasets import Dataset
import ast

class CurriculumDataset:
    def __init__(self, hf_dataset: Dataset):
        self.dataset = hf_dataset

        num_elements_to_examples = defaultdict(list)
        for idx, ex in enumerate(self.dataset):
            nums = ast.literal_eval(ex["query"].split("Using the numbers ")[1].split(" create")[0][:-1])
            num_elements = len(nums)
            num_elements_to_examples[num_elements].append(idx)

        self.unsorted_elements_to_examples = {k: v[:] for k, v in num_elements_to_examples.items()}
        self.sorted_num_elements_to_examples = sorted(self.unsorted_elements_to_examples.keys())

    def get_examples(self, num_elements: int) -> Dataset:
        if num_elements not in self.unsorted_elements_to_examples:
            return self.dataset.select([])

        idxs = self.unsorted_elements_to_examples[num_elements]
        return self.dataset.select(idxs)

    def iter_examples(self):
        for num_elements in self.sorted_num_elements_to_examples:
            yield num_elements, self.get_examples(num_elements)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]