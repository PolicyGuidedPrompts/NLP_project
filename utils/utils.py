import os
from transformers import BertModel, BertTokenizer, AutoTokenizer, GPT2LMHeadModel

import time
from functools import wraps


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper


def load_or_download_model(model_name='bert-base-uncased', model_dir="./saved_model"):
    model_dir = os.path.abspath(model_dir)

    # Try loading the model from a local directory. If it doesn't exist, download it and save it locally
    if (
            os.path.exists(os.path.join(model_dir, "config.json"))
            and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
            and os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
            and os.path.exists(os.path.join(model_dir, "vocab.json"))
    ):
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertModel.from_pretrained(model_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    return tokenizer, model

def load_or_download_llm_model(model_name='gpt2', model_dir="./llm_model/saved_model"):
    model_dir = os.path.abspath(model_dir)

    # Try loading the model from a local directory. If it doesn't exist, download it and save it locally
    if (
            os.path.exists(os.path.join(model_dir, "config.json"))
            and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
            and os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
            and os.path.exists(os.path.join(model_dir, "vocab.json"))
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    return tokenizer, model