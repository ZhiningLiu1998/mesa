import argparse

parser = argparse.ArgumentParser(description='Mesa Arguments')
parser.add_argument('--env-name', default="MESA-SAC")

# SAC arguments
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.01)')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr_decay_steps', type=int, default=10, metavar='N',
                    help='step_size of StepLR learning rate decay scheduler (default: 10)')
parser.add_argument('--lr_decay_gamma', type=float, default=0.99, metavar='N',
                    help='gamma of StepLR learning rate decay scheduler (default: 0.99)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.1)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=None, metavar='N',
                    help='random seed (default: None)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--hidden_size', type=int, default=50, metavar='N',
                    help='hidden size (default: 50)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simul|ator step (default: 1)')
parser.add_argument('--update_steps', type=int, default=1000, metavar='N',
                    help='maximum number of steps (default: 1000)')
parser.add_argument('--start_steps', type=int, default=500, metavar='N',
                    help='Steps sampling random actions (default: 500)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000, metavar='N',
                    help='size of replay buffer (default: 1000)')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='run on CUDA (default: False)')

# MESA arguments
parser.add_argument('--dataset', type=str, default='Mammo', metavar='N',
                    help='the dataset used for meta-training (default: Mammo)')
parser.add_argument('--metric', type=str, default='aucprc', metavar='N',
                    help='the metric used for evaluate (default: aucprc)')
parser.add_argument('--reward_coefficient', type=float, default=100, metavar='N')
parser.add_argument('--num_bins', type=int, default=5, metavar='N', 
                    help='number of bins (default: 5). state-size = 2 * num_bins.')
parser.add_argument('--sigma', type=float, default=0.2, metavar='N', 
                    help='sigma of the Gaussian function used in meta-sampling (default: 0.2)')
parser.add_argument('--max_estimators', type=int, default=10, metavar='N',
                    help='maximum number of base estimators in each meta-training episode (default: 10)')
parser.add_argument('--meta_verbose', type=int, default=10, metavar='N',
                    help='number of episodes between verbose outputs. \
                    If \'full\' print log for each base estimator (default: 10)')
parser.add_argument('--meta_verbose_mean_episodes', type=int, default=25, metavar='N',
                    help='number of episodes used for compute latest mean score in verbose outputs.')
parser.add_argument('--verbose', type=bool, default=False, metavar='N',
                    help='enable verbose when ensemble fit (default: False)')
parser.add_argument('--random_state', type=int, default=None, metavar='N', 
                    help='random_state (default: None)')
parser.add_argument('--train_ir', type=float, default=1, metavar='N', 
                    help='imbalance ratio of the training set after meta-sampling (default: 1)')
parser.add_argument('--train_ratio', type=float, default=1, metavar='N', 
                    help='the ratio of the data used in meta-training. \
                    set train_ratio<1 to use a random subset for meta-training (default: 1)')