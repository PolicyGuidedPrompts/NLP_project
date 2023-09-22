from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging
from abc import abstractmethod, ABC
import numpy as np

from utils.network_utils import device, np2torch

logger = logging.getLogger('root')


class RetrieverModel(ABC):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(script_dir, "../saved_models/retriever_models")

    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        logger.info(f"Loading retriever model {self.model_name=}")

        self.model = SentenceTransformer(self.model_path, cache_folder=self.models_dir).to(device)

    @abstractmethod
    def retrieve(self, encoding):
        pass

    @property
    def model_path(self):
        if hasattr(self, 'repository'):
            return os.path.join(self.repository, self.model_name)
        return self.model_name


class SBertRetriever(RetrieverModel):
    model_name = "multi-qa-mpnet-base-dot-v1"
    repository = "sentence-transformers"

    def __init__(self, config, dataset):
        super().__init__(self.model_name, config)
        dataset_to_retriever = dataset.prepare_dataset_to_retriever()
        self.top_k = config.retriever_top_k
        self.dataset_embeddings = self.model.encode(dataset_to_retriever)

    # TODO now - add encode function

    def encode(self, input):
        return self.model.encode(input, convert_to_tensor=True).detach().cpu().numpy()

    def retrieve(self, encoding):
        encoding_tensor = util.normalize_embeddings(np2torch(encoding.reshape(1, -1)))
        corpus_embeddings_tensor = util.normalize_embeddings(np2torch(self.dataset_embeddings))

        hits = util.semantic_search(query_embeddings=encoding_tensor, corpus_embeddings=corpus_embeddings_tensor,
                                    score_function=util.dot_score, top_k=self.top_k)

        return np.array([hit['corpus_id'] for hit in hits[0]])


AVAILABLE_RETRIEVERS = {
    'sbert': SBertRetriever
}


class RetrieverFactory:
    @staticmethod
    def create_retriever(model_name, config, dataset):
        if model_name in AVAILABLE_RETRIEVERS:
            return AVAILABLE_RETRIEVERS[model_name](config, dataset)
        else:
            logger.error(f"Retriever {model_name} not supported!")
            raise ValueError(f"Retriever {model_name} not supported!")
