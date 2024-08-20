import os
import sys
import gymnasium as gym

from erl_config import Config, get_gym_env_args
from erl_agent import AgentDQN
from erl_agent import AgentPPO
from erl_run import train_agent, valid_agent

gym.logger.set_level(40)  # Block warning


def train_dqn_for_cartpole(gpu_id=0):
    agent_class = AgentPPO  # DRL algorithm
    env_class = gym.make
    env_args = {
        'env_name': 'CartPole-v0',  # A pole is attached by an un-actuated joint to a cart.
        # Reward: keep the pole upright, a reward of `+1` for every step taken

        'state_dim': 4,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 2,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }
    get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.95  # discount factor of future rewards

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)



if __name__ == "__main__":
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_dqn_for_cartpole(gpu_id=GPU_ID)