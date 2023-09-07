import logging
import os
from abc import ABC, abstractmethod

from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score

logger = logging.getLogger('root')


# TODO - remember test part of dataset
class Dataset(ABC):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(script_dir, "../saved_datasets")

    def __init__(self):
        self.data = self.load_from_repository()
        logger.info(f"Loaded dataset {self.dataset_name=}, scoring method: {self.get_scoring_method_name()}")

    @abstractmethod
    def score(self, ground_truth, generated_answer):
        pass

    @property
    def dataset_path(self):
        if hasattr(self, 'repository'):
            return os.path.join(self.repository, self.dataset_name)
        return self.dataset_name


class StrategyQaDataset(Dataset):
    repository = "wics"
    dataset_name = "strategy-qa"

    # TODO - trim, lower, ground_truth in generated answer or vise versa
    def score(self, ground_truth, generated_answer):
        return 1 if ground_truth == generated_answer else -1

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} from {self.repository}")
        return load_dataset(self.dataset_path, cache_dir=self.datasets_dir)["test"].to_pandas()[
            ["question", "answer"]].assign(
            answer=lambda x: x['answer'].astype(str))

    @classmethod
    def get_scoring_method_name(cls):
        return 'exact_match'


class SquadDataset(Dataset):
    dataset_name = "squad"

    def score(self, ground_truth, generated_answer):
        return sentence_bleu([ground_truth.split()], generated_answer.split())

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        data = load_dataset(self.dataset_path, cache_dir=self.datasets_dir)["train"].to_pandas()
        data['answer'] = data['answers'].apply(lambda x: x['text'][0] if x else None)
        return data[["question", "answer"]]

    @classmethod
    def get_scoring_method_name(cls):
        return 'sentence_bleu'


# TODO - slow to load, skipping for now
class TriviaQaDataset(Dataset):
    dataset_name = "trivia_qa"

    def score(self, ground_truth, generated_answer):
        return f1_score(ground_truth.split(), generated_answer.split(), average='micro')

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        data = load_dataset(self.dataset_path, 'rc', cache_dir=self.datasets_dir)["train"].to_pandas()
        return data[["question", "answer"]]

    @classmethod
    def get_scoring_method_name(cls):
        return 'f1_score'


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
