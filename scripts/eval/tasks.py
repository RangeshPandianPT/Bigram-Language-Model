from datasets import load_dataset

class EvalTask:
    def get_dataset(self, split='validation'):
        raise NotImplementedError
        
    def format_example(self, example):
        # returns context, list_of_choices, correct_label_index
        raise NotImplementedError

class HellaSwagTask(EvalTask):
    def get_dataset(self, split='validation'):
        # HellaSwag has a 'validation' split
        return load_dataset("Rowan/hellaswag", split=split)
        
    def format_example(self, example):
        context = example["ctx"]
        choices = example["endings"]
        label = int(example["label"]) if example["label"] != "" else -1
        return context, choices, label

class PIQATask(EvalTask):
    def get_dataset(self, split='validation'):
        return load_dataset("ybisk/piqa", split=split)
        
    def format_example(self, example):
        context = example["goal"]
        choices = [example["sol1"], example["sol2"]]
        label = example["label"]
        return context, choices, label

class MMLUTask(EvalTask):
    def get_dataset(self, split='test'):
        # MMLU typically uses test split for evaluation
        return load_dataset("cais/mmlu", "all", split=split)
        
    def format_example(self, example):
        context = example["question"]
        choices = example["choices"]
        label = example["answer"]
        return context, choices, label

class ARCTask(EvalTask):
    def get_dataset(self, split='validation'):
        return load_dataset("ai2_arc", "ARC-Challenge", split=split)
        
    def format_example(self, example):
        context = example["question"]
        choices = example["choices"]["text"]
        label_str = example["answerKey"]
        
        # ARC answers are sometimes 'A', 'B', 'C', 'D' or '1', '2', '3', '4'
        labels_map = {l: i for i, l in enumerate(example["choices"]["label"])}
        label = labels_map.get(label_str, -1)
        return context, choices, label

class TruthfulQATask(EvalTask):
    def get_dataset(self, split='validation'):
        return load_dataset("truthful_qa", "multiple_choice", split=split)
        
    def format_example(self, example):
        context = example["question"]
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        # In TruthfulQA mc1, there is only one correct answer (label == 1)
        label = labels.index(1) if 1 in labels else -1
        return context, choices, label

TASKS = {
    "hellaswag": HellaSwagTask,
    "piqa": PIQATask,
    "mmlu": MMLUTask,
    "arc": ARCTask,
    "truthfulqa": TruthfulQATask
}
