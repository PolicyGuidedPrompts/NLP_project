import logging
import os
import re
import string
from abc import ABC, abstractmethod
from collections import Counter

from datasets import load_dataset

logger = logging.getLogger('root')


# TODO - remember test part of dataset
class Dataset(ABC):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(script_dir, "../saved_datasets")

    def __init__(self):
        self.data = self.load_from_repository()
        logger.info(f"Scoring method: {self.get_scoring_method_name()}")

    def score(self, ground_truth, generated_answer):
        prediction_tokens = self.normalize_answer(generated_answer).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return 2 * f1 - 1  # normalize between -1 and 1

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_scoring_method_name():
        return "f1_score"

    @property
    def dataset_path(self):
        if hasattr(self, 'repository'):
            return os.path.join(self.repository, self.dataset_name)
        return self.dataset_name

    @abstractmethod
    def load_from_repository(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update_prompt(self, action, current_prompt):
        pass


# TODO - implement reset + update_prompt for other datasets
class StrategyQaDataset(Dataset):
    repository = "wics"
    dataset_name = "strategy-qa"

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} from {self.repository}")
        return load_dataset(self.dataset_path, cache_dir=self.datasets_dir)["test"].to_pandas()[
            ["question", "answer", "facts", "decomposition"]].assign(
            answer=lambda x: x['answer'].astype(str))

    def reset(self):
        sample = self.data.sample(1).iloc[0]
        question, ground_truth = f'Question: {sample["question"]}\n' \
                                 f'Facts: {sample["facts"]}', sample["answer"]
        return question, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.data.iloc[action - 1]
        new_prompt = f'Question: {sample["question"]}\n' \
                     f'Facts: {sample["facts"]}\n' \
                     f'Answer: {sample["answer"]}\n' \
                     f'{current_prompt}'
        return new_prompt


class SquadDataset(Dataset):
    dataset_name = "squad"

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        data = load_dataset(self.dataset_path, cache_dir=self.datasets_dir)["train"].to_pandas()
        data['answer'] = data['answers'].apply(lambda x: x['text'][0] if x else None)
        return data[["question", "answer", "context"]]

    def reset(self):
        sample = self.data.sample(1).iloc[0]
        question, ground_truth = f'Question: {sample["question"]}\n' \
                                 f'Context: {sample["context"]}', sample["answer"]
        return question, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.data.iloc[action - 1]
        new_prompt = f'Question: {sample["question"]}\n' \
                     f'Context: {sample["context"]}\n' \
                     f'Answer: {sample["answer"]}\n' \
                     f'{current_prompt}'
        return new_prompt


# TODO - slow to load, skipping for now
class TriviaQaDataset(Dataset):
    dataset_name = "trivia_qa"

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        data = load_dataset(self.dataset_path, 'rc', cache_dir=self.datasets_dir)["train"].to_pandas()
        return data[["question", "answer"]]

    def reset(self):
        # TODO
        pass

    def update_prompt(self, action, current_prompt):
        # TODO
        pass


AVAILABLE_DATASETS = {
    'strategy-qa': StrategyQaDataset,
    'squad': SquadDataset,
    'trivia-qa': TriviaQaDataset
}


class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_name):
        if dataset_name in AVAILABLE_DATASETS:
            return AVAILABLE_DATASETS[dataset_name]()
        else:
            logger.error(f"Dataset {dataset_name} not supported!")
            raise ValueError(f"Dataset {dataset_name} not supported!")
