import logging
import os
from abc import abstractmethod
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import openai

logger = logging.getLogger('root')


class LLMModel:
    models_dir = "./saved_models/llm_models"

    @abstractmethod
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.llm_max_prompt_tokenized_len = 0
        logger.info(f"Loading llm model {self.model_name=}")

    # model that doesn't support this tokenization should override this logic
    def is_prompt_too_long(self, prompt):
        tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        tokenized_len = tokenized['input_ids'].shape[1]
        return tokenized_len > self.llm_max_prompt_tokenized_len

    @abstractmethod
    def generate_answer(self, prompt):
        pass

    # TODO - implement this for each model (in gpt3.5 should be no-op)
    @abstractmethod
    def parse_answer(self, generated_answer):
        pass


class GPT2LLM(LLMModel):
    model_name = "gpt2"

    def __init__(self, config):
        super().__init__()

        model_dir = os.path.join(os.path.abspath(LLMModel.models_dir), self.model_name)

        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
        if all(os.path.exists(os.path.join(model_dir, file)) for file in required_files):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.tokenizer.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm_max_prompt_tokenized_len = config.llm_max_prompt_tokenized_len
        self.max_output_tokenized_len = config.llm_max_output_tokenized_len
        self.temperature = config.llm_temperature

    def generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            output = self.model.generate(input_ids,
                                         attention_mask=attention_mask,
                                         early_stopping=True,
                                         max_length=input_ids.shape[-1] + self.max_output_tokenized_len,
                                         temperature=self.temperature)

        generated_answer = self.tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], skip_special_tokens=True
        )
        return generated_answer.strip()


class GPT35TurboLLM0613(LLMModel):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    model_name = "gpt-3.5-turbo-0613"

    def __init__(self, config):
        super().__init__()

        model_dir = os.path.join(os.path.abspath(LLMModel.models_dir), self.model_name)

        # Set GPT-2 tokenizer as default for GPT-3.5
        required_tokenizer_files = ["tokenizer_config.json", "vocab.json"]
        if all(os.path.exists(os.path.join(model_dir, file)) for file in required_tokenizer_files):
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.save_pretrained(model_dir)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm_max_prompt_tokenized_len = config.llm_max_prompt_tokenized_len
        self.max_output_tokenized_len = config.llm_max_output_tokenized_len
        self.temperature = config.llm_temperature

    def generate_answer(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_output_tokenized_len,
            temperature=self.temperature
        )
        return response['choices'][0]['message']['content'].strip()


AVAILABLE_LLM_MODELS = {
    'gpt3.5': GPT35TurboLLM0613,
    'gpt2': GPT2LLM
}


class LLMFactory:
    @staticmethod
    def create_llm(model_name, config):
        if model_name in AVAILABLE_LLM_MODELS:
            return AVAILABLE_LLM_MODELS[model_name](config)
        else:
            logger.error(f"Model {model_name} not supported!")
            raise ValueError(f"Model {model_name} not supported!")
