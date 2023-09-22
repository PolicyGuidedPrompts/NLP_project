from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging
from abc import abstractmethod, ABC
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils.network_utils import device, np2torch

logger = logging.getLogger('root')


class RetrieverModel(ABC):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(script_dir, "../saved_models/retriever_models")

    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        logger.info(f"Loading retriever model {self.model_name=}")

    @abstractmethod
    def retrieve(self, encoding):
        pass

    @abstractmethod
    def encode(self, input):
        pass

    @property
    def model_path(self):
        if hasattr(self, 'repository'):
            return os.path.join(self.repository, self.model_name)
        return self.model_name

    def _normalize_encoding(self, encoding):
        if self.config.normalize_encoding_method == 'l2':
            return encoding / np.linalg.norm(encoding)
        elif self.config.normalize_encoding_method == 'instance':
            return (encoding - encoding.mean()) / encoding.std()
        else:
            return encoding


class SBertRetriever(RetrieverModel):
    model_name = "multi-qa-mpnet-base-dot-v1"
    repository = "sentence-transformers"

    def __init__(self, config, dataset):
        super().__init__(self.model_name, config)
        self.model = SentenceTransformer(self.model_path, cache_folder=self.models_dir).to(device)
        dataset_to_retriever = dataset.prepare_dataset_to_retriever()
        self.top_k = config.retriever_top_k
        self.dataset_embeddings = self.model.encode(dataset_to_retriever)

    # TODO now - add encode function
    def encode(self, input):
        encoding = self.model.encode(input, convert_to_tensor=True).detach().cpu().numpy()
        return self._normalize_encoding(encoding)

    def retrieve(self, encoding):
        encoding_tensor = util.normalize_embeddings(np2torch(encoding.reshape(1, -1)))
        corpus_embeddings_tensor = util.normalize_embeddings(np2torch(self.dataset_embeddings))

        hits = util.semantic_search(query_embeddings=encoding_tensor, corpus_embeddings=corpus_embeddings_tensor,
                                    score_function=util.dot_score, top_k=self.top_k)

        return np.array([hit['corpus_id'] for hit in hits[0]])


class BertEncoder(RetrieverModel):
    model_name = "bert-base-uncased"

    def __init__(self, config, dataset):
        assert not config.retriever_top_k, "BertEncoder does not support retriever_top_k"
        super().__init__(self.model_name, config)
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.models_dir)
        self.model = AutoModel.from_pretrained(self.model_path, cache_dir=self.models_dir).to(device)

    def encode(self, input):
        inputs = self.tokenizer(input, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        encoding = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy()
        return self._normalize_encoding(encoding)

    def retrieve(self, _encoding):
        return np.array(range(len(self.dataset)))


class BgeLargeEnEncoder(RetrieverModel):
    repository = "BAAI"
    model_name = "bge-large-en"

    def __init__(self, config, dataset):
        super().__init__(self.model_name, config)
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.models_dir)
        self.model = AutoModel.from_pretrained(self.model_path, cache_dir=self.models_dir).to(device)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        encoding = sentence_embeddings.squeeze().detach().cpu().numpy()
        return self._normalize_encoding(encoding)

    def retrieve(self, _encoding):
        return np.array(range(len(self.dataset)))


class GteLargeEncoder(RetrieverModel):
    repository = "thenlper"
    model_name = "gte-large"

    def __init__(self, config, dataset):
        super().__init__(self.model_name, config)
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.models_dir)
        self.model = AutoModel.from_pretrained(self.model_path, cache_dir=self.models_dir).to(device)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        encoding = (last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]).squeeze().detach().cpu().numpy()
        return self._normalize_encoding(encoding)

    def retrieve(self, _encoding):
        return np.array(range(len(self.dataset)))


AVAILABLE_RETRIEVERS = {
    'sbert': SBertRetriever,
    'bert-no-op-retriever': BertEncoder,
    'bge-large-en-no-op-retriever': BgeLargeEnEncoder,
    'gte-large-no-op-retriever': GteLargeEncoder
}


class RetrieverFactory:
    @staticmethod
    def create_retriever(model_name, config, dataset):
        if model_name in AVAILABLE_RETRIEVERS:
            return AVAILABLE_RETRIEVERS[model_name](config, dataset)
        else:
            logger.error(f"Retriever {model_name} not supported!")
            raise ValueError(f"Retriever {model_name} not supported!")
