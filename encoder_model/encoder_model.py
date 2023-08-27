import os
from abc import abstractmethod
import torch
from transformers import BertTokenizer, BertModel


class EncoderModel:
    models_dir = "./saved_models/encoder_models"

    @abstractmethod
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def encode(self, text):
        pass


class BertEncoder(EncoderModel):
    model_name = "bert-base-uncased"

    def __init__(self):
        super().__init__()

        model_dir = os.path.join(os.path.abspath(EncoderModel.models_dir), self.model_name)

        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
        if all(os.path.exists(os.path.join(model_dir, file)) for file in required_files):
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model = BertModel.from_pretrained(model_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.tokenizer.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return the last hidden state as the text representation
        return outputs.last_hidden_state[0, 0, :]


AVAILABLE_ENCODERS = {
    'bert-base-uncased': BertEncoder
}


class EncoderFactory:
    @staticmethod
    def create_encoder(model_name):
        if model_name in AVAILABLE_ENCODERS:
            return AVAILABLE_ENCODERS[model_name]()
        else:
            raise ValueError(f"Encoder {model_name} not supported!")
