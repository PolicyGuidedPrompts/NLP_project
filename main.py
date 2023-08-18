# -*- coding: UTF-8 -*-

import argparse
import random

import numpy as np
import torch

from config import get_config
from dataset.dataset import DatasetFactory
from encoder_model.encoder_model import EncoderFactory
from env import Environment
from llm_model.llm_model import LLMFactory
from policy_gradient import PolicyGradient
from ppo import PPO
from general import get_logger

ALLOWED_DATASETS = ['wics/strategy-qa']
ALLOWED_LLMS = ['gpt2']
ALLOWED_ENCODERS = ['bert-base-uncased']
ALLOWED_ALGORITHMS = ['pg', 'ppo']

parser = argparse.ArgumentParser()
# Required
parser.add_argument("--dataset", type=str, required=True, choices=ALLOWED_DATASETS)
parser.add_argument("--llm_model", type=str, required=True, choices=ALLOWED_LLMS)
parser.add_argument("--encoder_model", type=str, required=True, choices=ALLOWED_ENCODERS)
parser.add_argument("--algorithm", type=str, required=True, choices=ALLOWED_ALGORITHMS)

# Defaults
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--baseline", type=bool, default=False)
parser.add_argument("--retriever", type=bool, default=False)
parser.add_argument("--eps_clip", type=float, default=0.2)  # For PPO
parser.add_argument("--update_freq", type=int, default=5)  # For PPO
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--layer_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=3e-2)
parser.add_argument("--num_batches", type=int, default=100)  # number of batches trained on
parser.add_argument("--batch_size", type=int, default=200)  # number of steps used to compute each policy update
parser.add_argument("--gamma", type=float, default=1.0)  # the discount factor
parser.add_argument("--normalize_advantage", type=bool, default=True)


# TODO - add logic for different datasets
# TODO - add logic for different llms + max_prompt_len
# TODO - right now we have code regarding gpt2, need to remove/do something about it because work with API now
# TODO - if time permits add retriever logic

def set_seeds(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# TODO - we want this as chain of responsibility
def validate_namespace(namespace):
    if namespace.algorithm == 'ppo':
        assert namespace.baseline, "PPO requires baseline"


# TODO - verify models_dir and model_name args
# TODO - encapsulate with logger init parts and important prints
if __name__ == "__main__":
    namespace = parser.parse_args()
    validate_namespace(namespace)

    set_seeds(seed=namespace.seed)
    config = get_config(namespace=namespace)
    logger = get_logger(config.log_path)

    dataset = DatasetFactory.create_dataset(dataset_name=namespace.dataset)
    llm_tokenizer, llm_model = LLMFactory.create_llm(model_name=namespace.llm_model)
    encoder_tokenizer, encoder_model = EncoderFactory.create_encoder(model_name=namespace.encoder_model)
    retriever_model = None  # TODO - add retriever logic

    # Define the environment
    env = Environment(
        dataset=dataset,
        llm_tokenizer=llm_tokenizer,
        llm_model=llm_model,
        encoder_tokenizer=encoder_tokenizer,
        encoder_model=encoder_model,
        seed=namespace.seed,
    )

    # TODO - maybe use factory here as well
    policy_search_algorithms = {'pg': PolicyGradient, 'ppo': PPO}
    # train model
    policy_search_algorithm = policy_search_algorithms[namespace.algorithm](env, config,
                                                                            logger)  # TODO - fix PPO seed logic
    policy_search_algorithm.run()

# TODO - clean repo from garbage
# TODO - make sure models are being loaded instead of downloaded
