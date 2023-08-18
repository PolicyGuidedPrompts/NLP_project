import os
from transformers import AutoTokenizer, GPT2LMHeadModel


class LLMModel:
    models_dir = "./saved_models/llm_models"

    def load(self):
        raise NotImplementedError


class GPT2LLM(LLMModel):
    model_name = "gpt2"

    def load(self):
        model_dir = os.path.join(os.path.abspath(LLMModel.models_dir), self.model_name)

        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
        if all(os.path.exists(os.path.join(model_dir, file)) for file in required_files):
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = GPT2LMHeadModel.from_pretrained(model_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = GPT2LMHeadModel.from_pretrained(self.model_name)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

        model.max_prompt_len = 200

        return tokenizer, model


class LLMFactory:
    @staticmethod
    def create_llm(model_name):
        available_llms = {GPT2LLM.model_name: GPT2LLM}
        if model_name in available_llms:
            return available_llms[model_name]().load()
        else:
            raise ValueError(f"Model {model_name} not supported!")
