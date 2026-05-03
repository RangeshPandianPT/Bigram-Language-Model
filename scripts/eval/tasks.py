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

TASKS = {
    "hellaswag": HellaSwagTask,
    "piqa": PIQATask
}
