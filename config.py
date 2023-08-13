class Config:
    def __init__(self, use_baseline, ppo, seed):
        self.record = False
        baseline_str = ("baseline" if use_baseline else "no_baseline") if not ppo else "ppo"
        seed_str = "seed=" + str(seed)
        # output config
        self.output_path = f"results/{baseline_str}-{seed_str}/"
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 200  # number of steps used to compute each policy update
        # TODO - adjust this according to maximum prompt length
        self.max_ep_len = 8  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 1.0  # the discount factor
        self.use_baseline = use_baseline
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 1
        self.layer_size = 64

        # hyperparameters for PPO
        self.eps_clip = 0.2
        self.update_freq = 5

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


def get_config(args):
    config = Config(args.use_baseline, args.ppo, args.seed)
    # TODO - LLM Model, Retreiever, Keep Track Original Question, Scoring Method, DatasetName, Encoder Model, Hyper Parameters, Evaluation Metrics
    return config
