import random

import numpy as np
import torch

from utils.arg_parser import parser
from config import get_config
from dataset.dataset import DatasetFactory
from environment.env import Environment
from llm_model.llm_model import LLMFactory
from policy_search.policy_gradient import PolicyGradient
from policy_search.ppo import PPO
from retriever_model.retriever_model import RetrieverFactory
from utils.network_utils import device
from utils.utils import get_logger


# TODO - don't forget in readme to specify env variables roles
# TODO - use sweep for grid search

def set_seeds(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    namespace = parser.parse_args()

    set_seeds(seed=namespace.seed)
    config = get_config(namespace=namespace)
    logger = get_logger(config.log_path)
    logger.info(f"Using device: {device}")
    logger.info(f"Config returned: {config.__dict__}")

    dataset = DatasetFactory.create_dataset(dataset_name=namespace.dataset)
    llm = LLMFactory.create_llm(model_name=namespace.llm_model, config=config)
    retriever = RetrieverFactory.create_retriever(model_name=namespace.retriever_model, config=config, dataset=dataset)

    env = Environment(
        dataset=dataset,
        llm=llm,
        retriever=retriever,
        seed=namespace.seed,
    )

    if namespace.algorithm == 'ppo':
        policy_search_algorithm = PPO(env, config)
    else:
        policy_search_algorithm = PolicyGradient(env, config)

    policy_search_algorithm.run()
