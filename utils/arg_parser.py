import argparse

from dataset.dataset import AVAILABLE_DATASETS
from llm_model.llm_model import AVAILABLE_LLM_MODELS
from retriever_model.retriever_model import AVAILABLE_RETRIEVERS

ALLOWED_DATASETS = AVAILABLE_DATASETS.keys()  # 'strategy-qa','squad','open-tdb', 'aqua_rat'
# 'gpt2','gpt3.5','llama-2-7b','flan-t5-base','flan-t5-small', 'flan-t5-large', 'flan-t5-xl'
ALLOWED_LLMS = AVAILABLE_LLM_MODELS.keys()
# 'sbert', 'bert-no-op-retriever', 'bge-no-op-retriever', 'gte-no-op-retriever'
ALLOWED_RETRIEVERS = AVAILABLE_RETRIEVERS.keys()
ALLOWED_ALGORITHMS = ['pg', 'ppo']
ALLOWED_NORMALIZE_ENCODING_METHODS = ['l2', 'instance']  # can leave empty for no normalization
# linear: T = T_init + (T_end - T_init) * (current_batch/num_batches)
# exponential: T = T_init * (T_exploration_decay_factor)^(current_batch) + 1
ALLOWED_POLICY_EXPLORATION_LOGIC = ['epsilon_greedy', 'linear_temperature_decay', 'exponential_temperature_decay']

parser = argparse.ArgumentParser()
# Run name
parser.add_argument("--run_name", type=str, default=None)

# Required
# Dataset
parser.add_argument("--dataset", type=str, required=True, choices=ALLOWED_DATASETS)
# Retriever
parser.add_argument("--retriever_model", type=str, required=True, choices=ALLOWED_RETRIEVERS)
parser.add_argument("--retriever_top_k", type=int)
# Algorithm
parser.add_argument("--algorithm", type=str, required=True, choices=ALLOWED_ALGORITHMS)
# LLM
parser.add_argument("--llm_model", type=str, required=True, choices=ALLOWED_LLMS)
parser.add_argument("--llm_max_prompt_tokenized_len", type=int, required=True)
parser.add_argument("--llm_max_output_tokenized_len", type=int, required=True)

# Defaults
# Retriever
parser.add_argument("--normalize_encoding_method", type=str, default='', choices=ALLOWED_NORMALIZE_ENCODING_METHODS)
# LLM
parser.add_argument("--llm_temperature", type=float, default=0.7)
# Policy
parser.add_argument("--gamma", type=float, default=1.0)  # discount factor
parser.add_argument("--baseline", action="store_true", default=False)
parser.add_argument("--policy_instance_norm", action="store_true", default=False)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--first_layer_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--normalize_advantage", type=bool, default=True)
# PPO Specific
parser.add_argument("--eps_clip", type=float, default=0.2)  # For PPO
parser.add_argument("--update_freq", type=int, default=5)  # For PPO
# Policy Exploration Logic
parser.add_argument("--policy_exploration_logic", type=str, default='epsilon_greedy',
                    choices=ALLOWED_POLICY_EXPLORATION_LOGIC)
parser.add_argument("--initial_temperature", type=float, default=400.0)  # For both linear and exponential
parser.add_argument("--end_temperature", type=float, default=1.0)  # For linear and epsilon greedy
parser.add_argument("--exploration_decay_factor", type=float, default=0.995)
# Global
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_batches", type=int, default=1)  # number of batches trained on
# number of steps used to compute each policy update
parser.add_argument("--num_episodes_per_batch", type=int, default=10)
