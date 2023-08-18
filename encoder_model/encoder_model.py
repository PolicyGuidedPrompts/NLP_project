import os
from transformers import BertTokenizer, BertModel


class EncoderModel:
    models_dir = "./saved_models/encoder_models"

    def load(self):
        raise NotImplementedError


# noinspection DuplicatedCode
class BertEncoder(EncoderModel):
    model_name = "bert-base-uncased"

    def load(self):
        model_dir = os.path.join(os.path.abspath(EncoderModel.models_dir), self.model_name)

        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
        if all(os.path.exists(os.path.join(model_dir, file)) for file in required_files):
            tokenizer = BertTokenizer.from_pretrained(model_dir)
            model = BertModel.from_pretrained(model_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
            model = BertModel.from_pretrained(self.model_name)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

        return tokenizer, model


class EncoderFactory:
    @staticmethod
    def create_encoder(model_name):
        available_encoders = {BertEncoder.model_name: BertEncoder}
        if model_name in available_encoders:
            return available_encoders[model_name]().load()
        else:
            raise ValueError(f"Encoder {model_name} not supported!")
