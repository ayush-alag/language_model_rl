from collections import defaultdict
from datasets import Dataset

class CurriculumDataset:
    def __init__(self, hf_dataset: Dataset):
        self.dataset = hf_dataset

        num_elements_to_examples = defaultdict(list)
        for idx, ex in enumerate(self.dataset):
            num_elements = len(ex["nums"])
            num_elements_to_examples[num_elements].append(idx)

        self._num_elements_to_examples = {k: v[:] for k, v in num_elements_to_examples.items()}
        self.num_elements_to_examples = sorted(self._num_elements_to_examples.keys())

    def get_examples(self, num_elements: int) -> Dataset:
        if num_elements not in self._num_elements_to_examples:
            return self.dataset.select([])

        idxs = self._num_elements_to_examples[num_elements]
        return self.dataset.select(idxs)

    def iter_examples(self):
        for num_elements in self.num_elements_to_examples:
            yield num_elements, self.get_examples(num_elements)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]