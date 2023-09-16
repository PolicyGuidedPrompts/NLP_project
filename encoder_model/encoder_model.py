import logging
import os
from abc import abstractmethod
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

from utils.network_utils import device

logger = logging.getLogger("root")


class EncoderModel:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(script_dir, "../saved_models/encoder_models")

    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        logger.info(f"Loading encoder model {self.model_name=}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.models_dir
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, cache_dir=self.models_dir
        ).to(device)

    @abstractmethod
    def encode(self, text):
        pass

    def _normalize_encoding(self, encoding):
        if self.config.normalize_encoding_method == "l2":
            return encoding / np.linalg.norm(encoding)
        elif self.config.normalize_encoding_method == "instance":
            return (encoding - encoding.mean()) / encoding.std()
        else:
            return encoding

    @property
    def model_path(self):
        if hasattr(self, "repository"):
            return os.path.join(self.repository, self.model_name)
        return self.model_name


class BertEncoder(EncoderModel):
    model_name = "bert-base-uncased"

    def __init__(self, config):
        super().__init__(self.model_name, config)

    def encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        encoding = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy()
        return self._normalize_encoding(encoding)


class BgeLargeEnEncoder(EncoderModel):
    repository = "BAAI"
    model_name = "bge-large-en"
    query_instruction = "Represent this sentence for searching relevant passages:"

    def __init__(self, config):
        super().__init__(self.model_name, config)

    # TODO - read about this model https://huggingface.co/BAAI/bge-large-en
    def encode(self, text, s2p_retrieval=False):
        if s2p_retrieval:
            text = self.query_instruction + text
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        encoding = sentence_embeddings.squeeze().detach().cpu().numpy()
        return self._normalize_encoding(encoding)


class GteLargeEncoder(EncoderModel):
    repository = "thenlper"
    model_name = "gte-large"

    def __init__(self, config):
        super().__init__(self.model_name, config)

    # TODO - read about this model https://huggingface.co/thenlper/gte-large
    def encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs["attention_mask"]
        last_hidden = outputs.last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        encoding = (
            (last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None])
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        return self._normalize_encoding(encoding)


AVAILABLE_ENCODERS = {
    "bert-base-uncased": BertEncoder,
    "bge-large-en": BgeLargeEnEncoder,
    "gte-large": GteLargeEncoder,
}


class EncoderFactory:
    @staticmethod
    def create_encoder(model_name, config):
        if model_name in AVAILABLE_ENCODERS:
            return AVAILABLE_ENCODERS[model_name](config)
        else:
            logger.error(f"Encoder {model_name} not supported!")
            raise ValueError(f"Encoder {model_name} not supported!")
