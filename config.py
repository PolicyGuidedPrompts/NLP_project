class Config:
    def __init__(self, namespace):
        # General part
        self.seed = namespace.seed

        # Dataset part
        self.dataset = namespace.dataset

        # LLM part
        self.llm_model = namespace.llm_model
        self.max_prompt_len = namespace.max_prompt_len  # TODO - based on model

        # Encoder part
        self.encoder_model = namespace.encoder_model

        # Retriever part
        self.retriever = namespace.retriever

        # Algorithm part
        self.algorithm = namespace.algorithm  # TODO - PPO or PG or REINFORCE
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

        # Output part
        self.output_path = f"results/" \
                           f"dataset={self.dataset}_" \
                           f"llm_model={self.llm_model}_" \
                           f"encoder_model={self.encoder_model}_" \
                           f"retriever={self.retriever}_" \
                           f"baseline={self.baseline}/"
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"

        # TODO Record part
        # self.record = False
        # self.record_path = self.output_path
        # self.record_freq = 5
        # self.summary_freq = 1


def get_config(namespace):
    # TODO - generate derived attributes from given args
    config = Config(namespace)
    return config
