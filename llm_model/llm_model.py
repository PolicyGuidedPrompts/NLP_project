import os
from abc import abstractmethod
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import openai


class LLMModel:
    models_dir = "./saved_models/llm_models"

    @abstractmethod
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def generate_answer(self, prompt):
        pass

    # TODO - implement this for each model (in gpt3.5 should be no-op)
    @abstractmethod
    def parse_answer(self, generated_answer):
        pass


class GPT2LLM(LLMModel):
    model_name = "gpt2"

    def __init__(self):
        super().__init__()

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

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        # TODO - adjust this later
        self.max_prompt_tokenized_len = 50

        self.tokenizer, self.model = tokenizer, model

    # TODO - google this and make sure it's correct
    def generate_answer(self, prompt):
        # TODO check this max_length
        inputs = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_prompt_tokenized_len)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # TODO check this max_length
        # TODO - maybe allow adjusting temperature
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_length=input_ids.shape[-1]+50, temperature=0.7)

        generated_answer = self.tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], skip_special_tokens=True
        )
        return generated_answer.strip()


class GPT35TurboLLM0613(LLMModel):
    model_name = "gpt-3.5-turbo-0613"

    def __init__(self):
        super().__init__()
        # TODO - adjust this later
        self.max_prompt_tokenized_len = 50

    def generate_answer(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": prompt}],
            # TODO - adjust this later
            max_tokens=15  # Limit the output to 15 tokens
        )
        return response['choices'][0]['message']['content'].strip()


AVAILABLE_LLM_MODELS = {
    'gpt3.5': GPT35TurboLLM0613,
    'gpt2': GPT2LLM
}


class LLMFactory:
    @staticmethod
    def create_llm(model_name):
        if model_name in AVAILABLE_LLM_MODELS:
            return AVAILABLE_LLM_MODELS[model_name]()
        else:
            raise ValueError(f"Model {model_name} not supported!")
