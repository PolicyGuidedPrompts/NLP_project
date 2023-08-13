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

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", dest="use_baseline", action="store_true")
parser.add_argument("--no-baseline", dest="use_baseline", action="store_false")
parser.add_argument("--ppo", dest="ppo", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=1)

parser.set_defaults(use_baseline=True)


# TODO - should be able to remove numpy from entire project
def set_seeds(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = parser.parse_args()
    set_seeds(args.seed)
    print("Creating configuration...")
    config = get_config(args)
    print("Finished configuration...")

    print("Loading dataset...")
    # Load the dataset
    dataset = load_dataset("wics/strategy-qa")["test"].to_pandas()[
        ["question", "answer"]
    ].assign(answer=lambda x: x['answer'].astype(str))
    print("Finished loading dataset...")

    print("Creating environment...")
    # Define the environment
    env = Environment(
        training_dataset=dataset,
    )
    print("Finished creating environment...")

    # train model
    model = PolicyGradient(env, config, args.seed) if not args.ppo else PPO(env, config, args.seed)
    model.run()
