import logging
import os
from abc import abstractmethod
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
import openai

from utils.network_utils import device

logger = logging.getLogger('root')

# TODO - all models here, tokenizers, everything should move to device

class LLMModel:
    models_dir = "./saved_models/llm_models"

    def __init__(self, config):
        self.model = None
        self.tokenizer = None
        self.max_prompt_tokenized_len = config.llm_max_prompt_tokenized_len
        self.max_output_tokenized_len = config.llm_max_output_tokenized_len
        self.temperature = config.llm_temperature
        logger.info(f"Loading llm model {self.model_name=}")

    # model that doesn't support this tokenization should override this logic
    def is_prompt_too_long(self, prompt):
        tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        tokenized_len = tokenized['input_ids'].shape[1]
        return tokenized_len > self.max_prompt_tokenized_len

    # model that doesn't support this tokenization should override this logic
    def generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            output = self.model.generate(input_ids,
                                         attention_mask=attention_mask,
                                         early_stopping=True,
                                         max_new_tokens=self.max_output_tokenized_len,
                                         temperature=self.temperature)

        generated_answer = self.tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], skip_special_tokens=True
        )
        return generated_answer.strip()

    @property
    def model_path(self):
        if hasattr(self, 'repository'):
            return os.path.join(self.repository, self.model_name)
        return self.model_name

    # TODO - think if the following even required:
    # TODO - implement this for each model (in gpt3.5 should be no-op)
    @abstractmethod
    def parse_answer(self, generated_answer):
        pass


class GPT2LLM(LLMModel):
    model_name = "gpt2"

    def __init__(self, config):
        super().__init__(config)

        model_dir = os.path.join(os.path.abspath(LLMModel.models_dir), self.model_name)

        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
        if all(os.path.exists(os.path.join(model_dir, file)) for file in required_files):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.tokenizer.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

# TODO - if I have time can make consistent with repository like in encoder and dataset
class Llama2LLM(LLMModel):
    model_name = 'Llama-2-7b-chat-hf'
    repository = 'meta-llama'
    hf_auth = os.environ.get('HF_TOKEN')

    def __init__(self, config):
        super().__init__(config)

        model_dir = os.path.join(os.path.abspath(LLMModel.models_dir), self.model_name)
        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]

        if all(os.path.exists(os.path.join(model_dir, file)) for file in required_files):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        else:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model_config = AutoConfig.from_pretrained(
                self.model_path,
                use_auth_token=self.hf_auth
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                config=self.model_config,
                quantization_config=self.bnb_config,
                device_map='auto',
                use_auth_token=self.hf_auth
            )
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_auth_token=self.hf_auth
            )

            # Save the model and tokenizer to disk for future usage
            self.tokenizer.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)


class GPT35TurboLLM0613(LLMModel):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    model_name = "gpt-3.5-turbo-0613"

    def __init__(self, config):
        super().__init__(config)

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
    'gpt2': GPT2LLM,
    'llama-2-7b': Llama2LLM
}


class LLMFactory:
    @staticmethod
    def create_llm(model_name, config):
        if model_name in AVAILABLE_LLM_MODELS:
            return AVAILABLE_LLM_MODELS[model_name](config)
        else:
            logger.error(f"Model {model_name} not supported!")
            raise ValueError(f"Model {model_name} not supported!")
