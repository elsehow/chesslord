# chess environment from PettingZoo
from pettingzoo.classic import chess_v6
# Ray Tune for hyperparameter tuning and experiment management
from ray import tune
# PettingZooEnv wrapper to make PettingZoo environments compatible with RLLib
from ray.rllib.env import PettingZooEnv
# PPO (Proximal Policy Optimization) algorithm configuration from RLLib
from ray.rllib.algorithms.ppo import PPOConfig
# environment registry to register custom environments with Ray
from ray.tune.registry import register_env

# Number of parallel environments
# Start with 2-4 runners and monitor your system
# Each additional runner may have diminishing returns - Ray has some coordination overhead.
NUM_ENV_RUNNERS = 4

# Number of episodes to train for
NUM_EPISODES = 100

# Function to create and configure the chess environment
def env_creator():
    # Create a new instance of the chess_v6 environment (standard chess game)
    env = chess_v6.env()
    # Set metadata flag to True - tells Ray that this environment can be run in parallel
    # This enables vectorization for faster training
    env.metadata["is_parallelizable"] = True
    # Return the configured environment instance
    return env

# Register our custom chess environment with Ray Tune using the name "chess_env"
# The lambda function creates a PettingZooEnv wrapper around our env_creator function
# This makes the PettingZoo chess environment compatible with RLLib's training framework
register_env("chess_env", lambda config: PettingZooEnv(env_creator()))

# Create a PPO algorithm configuration object for multi-agent chess training
config = (
    # Start with base PPO configuration
    PPOConfig()
    # Configure the environment settings
    .environment(
        env="chess_env",  # Use the registered chess environment
        clip_rewards=True  # Clip rewards to prevent exploding gradients during training
    )
    # Set number of parallel environments
    .env_runners(num_env_runners=NUM_ENV_RUNNERS)
    # Use PyTorch as the deep learning framework (instead of TensorFlow)
    .framework("torch")
    # Configure API stack settings for RLLib compatibility
    .api_stack(
        enable_rl_module_and_learner=False,  # Use older RLLib API for stability
        enable_env_runner_and_connector_v2=False  # Use older environment runner API
    )
    # Configure the neural network model architecture
    .training(
        model={
            # Define convolutional layers for processing the chess board (8x8 grid)
            "conv_filters": [
                # Layer 1: Basic pattern detection (32 filters, 3x3 kernel, stride 1)
                # - 32 filters: Learn fundamental chess patterns (piece attacks, basic threats)
                # - 3x3 kernel: Perfect size for local chess patterns (pieces influence immediate neighbors)
                # - Stride 1: Analyze every square to avoid missing tactical details
                [32, [3, 3], 1],
                
                # Layer 2: Tactical combination recognition (64 filters, 3x3 kernel, stride 1)
                # - 64 filters: Double the complexity to detect tactical combinations
                # - Combines features from layer 1 to recognize piece coordination
                # - Same 3x3 kernel maintains spatial consistency
                [64, [3, 3], 1],
                
                # Layer 3: Strategic pattern understanding (128 filters, 3x3 kernel, stride 1)
                # - 128 filters: Maximum complexity for high-level strategic concepts
                # - Processes complex board positions and long-term planning patterns
                # - Progressive doubling (32→64→128) is standard in deep learning architectures
                [128, [3, 3], 1],
            ],
            # Define fully connected (dense) layers after convolution
            "fcnet_hiddens": [256, 256],  # Two hidden layers with 256 neurons each
            "use_lstm": False,  # Don't use LSTM (Long Short-Term Memory) for this setup
        }
    )
    # Configure multi-agent settings for chess (two players)
    .multi_agent(
        policies=["shared_policy"],  # Both players use the same policy (shared neural network)
        # Function that maps any agent ID to the shared policy
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy")
    )
)

# Start training using Ray Tune with the configured PPO algorithm
tune.Tuner(
    "PPO",  # Specify the PPO algorithm to use
    # Configure training run settings
    run_config=tune.RunConfig(
        stop={"episodes_total": NUM_EPISODES},  # Stop after NUM_EPISODES 
        storage_path="~/ray_results", # after running, `tensorboard --logdir=~/ray_results` to see the training progress
    ),
    # Convert the PPO configuration to a dictionary format for Ray Tune
    param_space=config.to_dict()
).fit()  # Begin the training process