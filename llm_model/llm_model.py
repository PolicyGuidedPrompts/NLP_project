import logging
import os
from abc import abstractmethod

import openai
import torch
import transformers
from retrying import retry

from utils.network_utils import device
from utils.utils import timeout

logger = logging.getLogger('root')


class LLMModel:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(script_dir, "../saved_models/llm_models")

    def __init__(self, config):
        self.max_prompt_tokenized_len = config.llm_max_prompt_tokenized_len
        self.max_output_tokenized_len = config.llm_max_output_tokenized_len
        self.temperature = config.llm_temperature

        self.model_name = None
        self.model = None
        self.tokenizer = None
        logger.info(f"Loading llm model {self.model_name=}")

    def is_prompt_too_long(self, prompt):
        tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        tokenized_len = tokenized['input_ids'].shape[1]
        return tokenized_len > self.max_prompt_tokenized_len

    @abstractmethod
    def generate_answer(self, prompt):
        pass

    @property
    def model_path(self):
        if hasattr(self, 'repository'):
            return os.path.join(self.repository, self.model_name)
        return self.model_name


class GPT2LLM(LLMModel):
    model_name = "gpt2"

    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.models_dir)
        self.model = transformers.GPT2LMHeadModel.from_pretrained(self.model_path, cache_dir=self.models_dir).to(device)
        self.model.eval()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token


class GPT35TurboLLM0613(LLMModel):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    model_name = "gpt-3.5-turbo-0613"

    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=self.models_dir)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    #  retry for 3 times with a 1-minute wait
    @retry(stop_max_attempt_number=3, wait_fixed=60 * 1000)
    def generate_answer(self, prompt):
        try:
            with timeout(60):  # seconds
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_output_tokenized_len,
                    temperature=self.temperature)
                return response['choices'][0]['message']['content'].strip()
        except TimeoutError:
            logger.warning(f"TimeoutError while generating answer")
            raise


class BaseFlanT5LLM(LLMModel):
    repository = 'google'

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(self.model_path, cache_dir=self.models_dir)
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(self.model_path,
                                                                             cache_dir=self.models_dir).to(device)
        self.model.eval()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            output = self.model.generate(input_ids,
                                         attention_mask=attention_mask,
                                         early_stopping=True,
                                         max_new_tokens=self.max_output_tokenized_len,
                                         temperature=self.temperature).cpu()

        generated_answer = self.tokenizer.decode(
            output[0], skip_special_tokens=True
        )
        return generated_answer.strip()


class FlanT5SmallLLM(BaseFlanT5LLM):
    model_name = "flan-t5-small"


class FlanT5BaseLLM(BaseFlanT5LLM):
    model_name = "flan-t5-base"


class FlanT5LargeLLM(BaseFlanT5LLM):
    model_name = "flan-t5-large"


class FlanT5XLLLM(BaseFlanT5LLM):
    model_name = "flan-t5-xl"


class Llama2LLM(LLMModel):
    model_name = 'Llama-2-7b-chat-hf'
    repository = 'meta-llama'
    hf_auth = os.environ.get('HF_TOKEN')

    def __init__(self, config):
        super().__init__(config)
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.prefix = "Answer only the last question in the same fashion other questions were answered."

        self.model_config = transformers.AutoConfig.from_pretrained(
            self.model_path,
            use_auth_token=self.hf_auth,
            cache_dir=self.models_dir
        )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            config=self.model_config,
            device_map='auto',
            use_auth_token=self.hf_auth,
            cache_dir=self.models_dir
        ).to(device)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            use_auth_token=self.hf_auth,
            cache_dir=self.models_dir
        )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_answer(self, prompt):
        formatted_prompt = f"{self.B_INST} {self.B_SYS}{self.prefix}\n{prompt}{self.E_SYS} {self.E_INST}"
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt', truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            output = self.model.generate(input_ids,
                                         attention_mask=attention_mask,
                                         early_stopping=True,
                                         max_new_tokens=self.max_output_tokenized_len,
                                         temperature=self.temperature).cpu()

        generated_answer = self.tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], skip_special_tokens=True
        )
        return generated_answer.strip()


AVAILABLE_LLM_MODELS = {
    'gpt2': GPT2LLM,
    'gpt3.5': GPT35TurboLLM0613,
    'flan-t5-small': FlanT5SmallLLM,
    'flan-t5-base': FlanT5BaseLLM,
    'flan-t5-large': FlanT5LargeLLM,
    'flan-t5-xl': FlanT5XLLLM,
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
