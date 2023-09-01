class Config:
    def __init__(self, namespace):
        # General part
        self.seed = namespace.seed

        # Dataset part
        self.dataset = namespace.dataset
        self._dataset_name = self.dataset.replace('/', '-')

        # LLM part
        self.llm_model = namespace.llm_model

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

        retriever_str = f"_retriever={self.retriever}" if self.retriever else ""
        baseline_str = f"_baseline={self.baseline}" if self.baseline else ""
        # Output part
        self.output_path = f"results/" \
                           f"algorithm={self.algorithm}_" \
                           f"dataset={self._dataset_name}_" \
                           f"llm={self.llm_model}_" \
                           f"encoder={self.encoder_model}" \
                           f"{retriever_str}" \
                           f"{baseline_str}" \
                           f"/"
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"


def get_config(namespace):
    config = Config(namespace)
    return config
