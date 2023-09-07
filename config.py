import os
import sys


class Config:
    def __init__(self, namespace):
        # General part
        self.seed = namespace.seed

        # Dataset part
        self.dataset = namespace.dataset
        self._dataset_name = self.dataset.replace('/', '-')

        # LLM part
        self.llm_model = namespace.llm_model
        self.llm_max_prompt_tokenized_len = namespace.llm_max_prompt_tokenized_len
        self.llm_max_output_tokenized_len = namespace.llm_max_output_tokenized_len
        self.llm_temperature = namespace.llm_temperature

        # Encoder part
        self.encoder_model = namespace.encoder_model

        # Retriever part
        self.retriever = namespace.retriever

        # Algorithm part
        self.algorithm = namespace.algorithm
        # hyperparameters for PPO
        self.eps_clip = namespace.eps_clip
        self.update_freq = namespace.update_freq

        # Policy + Baseline part
        self.n_layers = namespace.n_layers
        self.layer_size = namespace.layer_size
        self.learning_rate = namespace.learning_rate
        self.baseline = namespace.baseline
        self.num_batches = namespace.num_batches  # number of batches trained on
        self.batch_size = namespace.batch_size  # number of steps used to compute each policy update
        self.gamma = namespace.gamma  # the discount factor
        self.normalize_advantage = True

        # Get the directory where this script resides
        self.BASE_DIR = os.path.dirname(os.path.realpath(__file__))

        # Construct paths relative to this directory
        retriever_str = f"_retriever={self.retriever}" if self.retriever else ""
        baseline_str = f"_baseline={self.baseline}" if self.baseline else ""
        rel_output_path = os.path.join(
            "results",
            f"algorithm={self.algorithm}_dataset={self._dataset_name}_llm={self.llm_model}_encoder={self.encoder_model}{retriever_str}{baseline_str}"
        )
        self.output_path = os.path.join(self.BASE_DIR, rel_output_path)

        # Note: Using os.path.join ensures the path is constructed correctly for the operating system.
        self.model_output = os.path.join(self.output_path, "model.weights")
        self.log_path = os.path.join(self.output_path, "log.txt")
        self.scores_output = os.path.join(self.output_path, "scores.npy")
        self.plot_output = os.path.join(self.output_path, "scores.png")


def get_config(namespace):
    config = Config(namespace)
    return config
