import logging
from abc import ABC, abstractmethod
from datasets import load_dataset

logger = logging.getLogger('root')


class Dataset(ABC):

    def __init__(self):
        logger.info(f"Loading dataset {self.dataset_name=}")

    @abstractmethod
    def load(self):
        pass


class StrategyQaDataset(Dataset):
    dataset_name = "wics/strategy-qa"

    def load(self):
        dataset = load_dataset(self.dataset_name)["test"].to_pandas()[
            ["question", "answer"]
        ].assign(answer=lambda x: x['answer'].astype(str))
        return dataset


class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_name):
        available_datasets = {StrategyQaDataset.dataset_name: StrategyQaDataset}
        if dataset_name in available_datasets:
            return available_datasets[dataset_name]().load()
        else:
            logger.error(f"Dataset {dataset_name} not supported!")
            raise ValueError(f"Dataset {dataset_name} not supported!")
