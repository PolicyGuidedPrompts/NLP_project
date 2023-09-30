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

    def retrieve(self, _encoding, _mode):
        return np.array(range(len(self.dataset.train_data)))

    def _normalize_encoding(self, encoding):
        if self.config.normalize_encoding_method == 'l2':
            return encoding / np.linalg.norm(encoding)
        elif self.config.normalize_encoding_method == 'instance':
            return (encoding - encoding.mean()) / encoding.std()
        else:
            return encoding

    @abstractmethod
    def encode(self, _input):
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
        assert config.retriever_top_k, "retriever_top_k is required when using sbert model"
        super().__init__(self.model_name, config)
        self.model = SentenceTransformer(self.model_path, cache_folder=self.models_dir).to(device)
        dataset_to_retriever = dataset.prepare_dataset_to_retriever()
        self.top_k = config.retriever_top_k

        self.cache_dir = os.path.join(RetrieverModel.script_dir, f"../dataset_encodings/{self.model_name}")
        self.cache_file = os.path.join(self.cache_dir, f"{dataset.dataset_name}.pt")

        if os.path.exists(self.cache_file):
            self.dataset_embeddings = torch.load(self.cache_file)
        else:
            self.dataset_embeddings = self.model.encode(dataset_to_retriever)

            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            torch.save(self.dataset_embeddings, self.cache_file)

    def encode(self, encoder_input):
        encoding = self.model.encode(encoder_input, convert_to_tensor=True).detach().cpu().numpy()
        return self._normalize_encoding(encoding)

    def retrieve(self, encoding, mode):
        encoding_tensor = util.normalize_embeddings(np2torch(encoding.reshape(1, -1)))
        corpus_embeddings_tensor = util.normalize_embeddings(np2torch(self.dataset_embeddings))

        if mode == 'train':
            hits = util.semantic_search(query_embeddings=encoding_tensor, corpus_embeddings=corpus_embeddings_tensor,
                                        score_function=util.dot_score, top_k=self.top_k + 1)
            # remove first hit which is the question itself
            return np.array([hit['corpus_id'] for hit in hits[0][1:]])

        # in test there is no need to remove first hit since it won't be the question itself
        hits = util.semantic_search(query_embeddings=encoding_tensor, corpus_embeddings=corpus_embeddings_tensor,
                                    score_function=util.dot_score, top_k=self.top_k)
        return np.array([hit['corpus_id'] for hit in hits[0]])


class Encoder(RetrieverModel):
    def __init__(self, config, dataset):
        assert not config.retriever_top_k, f"{self.model_name} encoder does not support retriever_top_k"
        super().__init__(self.model_name, config)
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.models_dir)
        self.model = AutoModel.from_pretrained(self.model_path, cache_dir=self.models_dir).to(device)


class BertEncoder(Encoder):
    model_name = "bert-base-uncased"

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        encoding = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy()
        return self._normalize_encoding(encoding)


class BgeLargeEnEncoder(Encoder):
    repository = "BAAI"
    model_name = "bge-large-en"

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        encoding = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy()
        return self._normalize_encoding(encoding)


class GteLargeEncoder(Encoder):
    repository = "thenlper"
    model_name = "gte-large"

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        encoding = (last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]).squeeze().detach().cpu().numpy()
        return self._normalize_encoding(encoding)


AVAILABLE_RETRIEVERS = {
    'sbert': SBertRetriever,
    'bert-no-op-retriever': BertEncoder,
    'bge-no-op-retriever': BgeLargeEnEncoder,
    'gte-no-op-retriever': GteLargeEncoder
}


class RetrieverFactory:
    @staticmethod
    def create_retriever(model_name, config, dataset):
        if model_name in AVAILABLE_RETRIEVERS:
            return AVAILABLE_RETRIEVERS[model_name](config, dataset)
        else:
            logger.error(f"Retriever {model_name} not supported!")
            raise ValueError(f"Retriever {model_name} not supported!")
