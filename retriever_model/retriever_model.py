from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging
from abc import abstractmethod, ABC

from utils.network_utils import device

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
    model_name = "paraphrase-distilroberta-base-v1"

    def __init__(self, config, dataset):
        super().__init__(self.model_name, config)
        self.dataset_embeddings = self.model.encode(dataset.data, convert_to_tensor=True)

    def retrieve(self, encoding):
        encoding_tensor = self.model.encode(encoding, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(encoding_tensor, self.dataset_embeddings)

        top_results = torch.topk(cosine_scores, k=self.config.retriever_top_k)
        indices = top_results.indices.squeeze(0).tolist()
        scores = top_results.values.squeeze(0).tolist()

        return [(index, score) for index, score in zip(indices, scores)]


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
