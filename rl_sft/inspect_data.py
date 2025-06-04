from dataloader import load_countdown_dataset
from collections import Counter
import matplotlib.pyplot as plt

def inspect_data(countdown_dataset):
    counter = Counter()
    for example in countdown_dataset:
        counter[len(example["numbers"])] += 1
    return counter

def inspect_data_ranges(countdown_dataset):
    counter = Counter()
    min_num = None
    max_num = None
    for example in countdown_dataset:
        for i in range(len(example["numbers"])):
            if min_num is None:
                min_num = example["numbers"][i]
            if max_num is None:
                max_num = example["numbers"][i]
            min_num = min(min_num, example["numbers"][i])
            max_num = max(max_num, example["numbers"][i])
            counter[example["numbers"][i]] += 1
    print(min_num, max_num)
    return counter

def inspect_target_ranges(countdown_dataset):
    min_num = None
    max_num = None
    for example in countdown_dataset:
        if min_num is None:
            min_num = example["target"]
        if max_num is None:
            max_num = example["target"]
        min_num = min(min_num, example["target"])
        max_num = max(max_num, example["target"])
    print(min_num, max_num)
    return min_num, max_num

def make_plot(counter, filename):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(counter.keys(), counter.values(), color='skyblue', edgecolor='navy', alpha=0.7)

    total = sum(counter.values())

    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{percentage:.1f}%',
                ha='center', va='center', fontsize=22, color='navy', weight='bold')

    plt.title('Distribution of Number Lengths in Countdown Dataset', fontsize=16, pad=15)
    plt.xlabel('Number of Elements', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    plt.xticks(list(counter.keys()), fontsize=22)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    _, countdown_dataset = load_countdown_dataset(1, 1024, from_json=False)
    # counter = inspect_data(countdown_dataset)
    # counter2 = inspect_data_ranges(countdown_dataset)
    # make_plot(counter, "countdown_data_length.png")
    # make_plot(counter2, "countdown_data_ranges.png")
    min_num, max_num = inspect_target_ranges(countdown_dataset)
    print(min_num, max_num)