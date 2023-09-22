import os


class Config:
    def __init__(self, namespace):
        # Run Name
        self.run_name = namespace.run_name

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

        # Retriever part
        self.retriever_model = namespace.retriever_model
        self.retriever_top_k = namespace.retriever_top_k

        # Algorithm part
        self.algorithm = namespace.algorithm
        # hyperparameters for PPO
        self.eps_clip = namespace.eps_clip
        self.update_freq = namespace.update_freq

        # TODO - when splitting baseline and policy network layers, remember to update output path
        # Policy + Baseline part
        self.policy_instance_norm = namespace.policy_instance_norm
        self.n_layers = namespace.n_layers
        self.first_layer_size = namespace.first_layer_size
        self.learning_rate = namespace.learning_rate
        self.baseline = namespace.baseline
        self.num_batches = namespace.num_batches  # number of batches trained on
        self.num_episodes_per_batch = namespace.num_episodes_per_batch  # number of steps used to compute each policy update
        self.gamma = namespace.gamma  # the discount factor
        self.normalize_advantage = True

        # Softmax Temperature Logic
        self.initial_temperature = namespace.initial_temperature
        self.end_temperature = namespace.end_temperature
        self.exploration_decay_factor = namespace.exploration_decay_factor
        self.policy_exploration_logic = namespace.policy_exploration_logic

        # Get the directory where this script resides
        self.BASE_DIR = os.path.dirname(os.path.realpath(__file__))

        # Construct paths relative to this directory
        baseline_str = f"_baseline={self.baseline}" if self.baseline else ""

        first_level = f"dataset={self._dataset_name}_llm={self.llm_model}" \
                      f"_retriever={self.retriever_model}_retriever_top_k={self.retriever_top_k}" \
                      f"_algorithm={self.algorithm}" \
                      f"_policy_exploration_logic={self.policy_exploration_logic}" \
                      f"_policy_instance_norm={self.policy_instance_norm}" \
                      f"{baseline_str}"
        second_level = f"llm_max_prompt_tokenized_len={self.llm_max_prompt_tokenized_len}" \
                       f"_llm_max_output_tokenized_len={self.llm_max_output_tokenized_len}" \
                       f"_llm_temperature={self.llm_temperature}_eps_clip={self.eps_clip}" \
                       f"_n_layers={self.n_layers}_learning_rate={self.learning_rate}" \
                       f"_num_batches={self.num_batches}_num_episodes_per_batch={self.num_episodes_per_batch}" \
                       f"_gamma={self.gamma}_initial_temperature={self.initial_temperature}" \
                       f"_end_temperature={self.end_temperature}" \
                       f"_exploration_decay_factor={self.exploration_decay_factor}"

        rel_output_path = os.path.join("results", first_level, second_level)
        self.output_path = os.path.join(self.BASE_DIR, rel_output_path)

        # Note: Using os.path.join ensures the path is constructed correctly for the operating system.
        self.model_output = os.path.join(self.output_path, "model.weights")
        self.log_path = os.path.join(self.output_path, "log.txt")
        self.scores_output = os.path.join(self.output_path, "scores.npy")
        self.plot_output = os.path.join(self.output_path, "scores.png")


def get_config(namespace):
    config = Config(namespace)
    return config
