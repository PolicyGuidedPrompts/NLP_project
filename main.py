# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import torch
from policy_gradient import PolicyGradient
from ppo import PPO
from config import get_config
import random
from datasets import load_dataset
from env import Environment

ALLOWED_DATASETS = ['wics/strategy-qa']
ALLOWED_LLMS = ['gpt2']
ALLOWED_ENCODERS = ['bert-base-uncased']
ALLOWED_ALGORITHMS = ['pg', 'ppo']

parser = argparse.ArgumentParser()
# Required
parser.add_argument("--dataset", type=str, required=True, choices=ALLOWED_DATASETS)
parser.add_argument("--llm_model", type=str, required=True, choices=ALLOWED_LLMS)
parser.add_argument("--encoder", type=str, required=True, choices=ALLOWED_ENCODERS)
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


if __name__ == "__main__":
    namespace = parser.parse_args()

    set_seeds(namespace.seed)
    config = get_config(namespace)

    # Load the dataset
    dataset = load_dataset("wics/strategy-qa")["test"].to_pandas()[
        ["question", "answer"]
    ].assign(answer=lambda x: x['answer'].astype(str))

    # Define the environment
    env = Environment(
        training_dataset=dataset,
    )

    # train model
    model = PolicyGradient(env, config, namespace.seed) if not namespace.ppo else PPO(env, config, namespace.seed)
    model.run()
