import argparse
import logging
import random

import numpy as np
import torch

from config import get_config
from dataset.dataset import DatasetFactory, AVAILABLE_DATASETS
from encoder_model.encoder_model import EncoderFactory, AVAILABLE_ENCODERS
from environment.env import Environment
from llm_model.llm_model import LLMFactory, AVAILABLE_LLM_MODELS
from utils.network_utils import device
from policy_search.policy_gradient import PolicyGradient
from policy_search.ppo import PPO
from utils.utils import get_logger

# TODO - epsilon greedy
# TODO - minimize requirements

ALLOWED_DATASETS = AVAILABLE_DATASETS.keys()  # 'strategy-qa','squad','trivia-qa'
ALLOWED_LLMS = AVAILABLE_LLM_MODELS.keys()  # 'gpt2','gpt3.5','llama-2-7b','flan-t5-base','flan-t5-small'
ALLOWED_ENCODERS = AVAILABLE_ENCODERS.keys()  # 'bert-base-uncased','bge-large-en','gte-large'
ALLOWED_ALGORITHMS = ['pg', 'ppo']
ALLOWED_NORMALIZE_ENCODING_METHODS = ['l2', 'instance']  # can leave empty for no normalization

# linear: T = T_init + (T_end - T_init) * (current_batch/num_batches)
# exponential: T = T_init * (T_exploration_decay_factor)^(current_batch) + 1
ALLOWED_POLICY_EXPLORATION_LOGIC = ['epsilon_greedy', 'linear_temperature_decay', 'exponential_temperature_decay']

parser = argparse.ArgumentParser()
# Run name
parser.add_argument("--run_name", type=str, default=None)

# Required
parser.add_argument("--dataset", type=str, required=True, choices=ALLOWED_DATASETS)
parser.add_argument("--llm_model", type=str, required=True, choices=ALLOWED_LLMS)
# Encoder
parser.add_argument("--encoder_model", type=str, required=True, choices=ALLOWED_ENCODERS)
parser.add_argument("--normalize_encoding_method", type=str, default='', choices=ALLOWED_NORMALIZE_ENCODING_METHODS)
# Algorithm
parser.add_argument("--algorithm", type=str, required=True, choices=ALLOWED_ALGORITHMS)

# Defaults
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--baseline", action="store_true", default=False)
parser.add_argument("--retriever", type=bool, default=False)
parser.add_argument("--eps_clip", type=float, default=0.2)  # For PPO
parser.add_argument("--update_freq", type=int, default=5)  # For PPO
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--layer_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--num_batches", type=int, default=1)  # number of batches trained on
parser.add_argument("--num_episodes_per_batch", type=int,
                    default=10)  # number of steps used to compute each policy update
parser.add_argument("--gamma", type=float, default=1.0)  # discount factor
parser.add_argument("--normalize_advantage", type=bool, default=True)
# TODO - should be a function of the model, for example t5 flan base can only handle 512 hence should be configured to 450 +-
parser.add_argument("--llm_max_prompt_tokenized_len", type=int, default=450)
parser.add_argument("--llm_max_output_tokenized_len", type=int, default=15)
parser.add_argument("--llm_temperature", type=float, default=0.7)
# Policy exploration logic
parser.add_argument("--policy_exploration_logic", type=str, default='epsilon_greedy', choices=ALLOWED_ALGORITHMS)
parser.add_argument("--initial_temperature", type=float, default=400)  # For both linear and exponential
parser.add_argument("--end_temperature", type=float, default=1)  # For linear and epsilon greedy
# For exponential decay (should be ~0.85), and epsilon greedy (should be ~0.995)
parser.add_argument("--exploration_decay_factor", type=float, default=0.995)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# TODO - if time permits add retriever logic
# TODO - don't forget in readme to specify env variables roles
# TODO - logger should also log errors
# TODO - see if can reduce size of requirements.txt
# TODO - Download the best model file from a sweep https://wandb.ai/yair_nlp/NLP_project/overview

def set_seeds(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# TODO - create dataloader and somehow keep logic the same
# TODO - create predetermined configuration files instead that work together
def validate_namespace(namespace):
    if namespace.algorithm == 'ppo':
        assert namespace.baseline, "PPO requires baseline"


if __name__ == "__main__":
    namespace = parser.parse_args()
    validate_namespace(namespace)

    set_seeds(seed=namespace.seed)
    config = get_config(namespace=namespace)
    logger = get_logger(config.log_path)
    logger.info(f"Using device: {device}")
    logger.info(f"Config returned: {config.__dict__}")

    dataset = DatasetFactory.create_dataset(dataset_name=namespace.dataset)
    llm = LLMFactory.create_llm(model_name=namespace.llm_model, config=config)
    encoder = EncoderFactory.create_encoder(model_name=namespace.encoder_model, config=config)
    retriever_model = None  # TODO - add retriever logic

    env = Environment(
        dataset=dataset,
        llm=llm,
        encoder=encoder,
        seed=namespace.seed,
    )

    policy_search_algorithms = {'pg': PolicyGradient, 'ppo': PPO}
    policy_search_algorithm = policy_search_algorithms[namespace.algorithm](env, config)
    policy_search_algorithm.run()
