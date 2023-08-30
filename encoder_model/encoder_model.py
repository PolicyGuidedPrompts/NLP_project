import os
from abc import abstractmethod
import torch
from transformers import AutoTokenizer, AutoModel


class EncoderModel:
    models_dir = "./saved_models/encoder_models"

    def __init__(self, model_name):
        self.model_name = model_name
        model_dir = os.path.join(os.path.abspath(self.models_dir), self.model_name)

        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
        if all(os.path.exists(os.path.join(model_dir, file)) for file in required_files):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModel.from_pretrained(model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.tokenizer.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)

    @abstractmethod
    def encode(self, text):
        pass

    @property
    def model_path(self):
        if hasattr(self, 'repository'):
            return os.path.join(self.repository, self.model_name)
        return self.model_name


class BertEncoder(EncoderModel):
    model_name = "bert-base-uncased"

    def __init__(self):
        super().__init__(self.model_name)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0, 0, :]


class BgeLargeEnEncoder(EncoderModel):
    repository = "BAAI"
    model_name = "bge-large-en"
    query_instruction = "Represent this sentence for searching relevant passages:"

    def __init__(self):
        super().__init__(self.model_name)

    # TODO - read about this model https://huggingface.co/BAAI/bge-large-en
    def encode(self, text, s2p_retrieval=False):
        if s2p_retrieval:
            text = self.query_instruction + text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embeddings = outputs[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.squeeze()


class GteLargeEncoder(EncoderModel):
    repository = "thenlper"
    model_name = "gte-large"

    def __init__(self):
        super().__init__(self.model_name)

    # TODO - read about this model https://huggingface.co/thenlper/gte-large
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return (last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]).squeeze()


AVAILABLE_ENCODERS = {
    'bert-base-uncased': BertEncoder,
    'bge-large-en': BgeLargeEnEncoder,
    'gte-large': GteLargeEncoder
}


class EncoderFactory:
    @staticmethod
    def create_encoder(model_name):
        if model_name in AVAILABLE_ENCODERS:
            return AVAILABLE_ENCODERS[model_name]()
        else:
            raise ValueError(f"Encoder {model_name} not supported!")
