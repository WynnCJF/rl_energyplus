#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys
from matplotlib.pyplot import savefig

from mpi4py import MPI
from common.energyplus_util import (
    energyplus_arg_parser,
    energyplus_logbase_dir
)
from baselines.common.models import mlp
from baselines.trpo_mpi import trpo_mpi
import os
import datetime
from baselines import logger
from baselines_energyplus.bench import Monitor
import gym
import tensorflow as tf
from baselines.common.policies import PolicyWithValue


def make_energyplus_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for EnergyEnv
    """
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env


def train(env_id, num_timesteps, seed):
    # import baselines.common.tf_util as U
    # sess = U.single_threaded_session()
    # sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

    # Create a new base directory like /tmp/openai-2018-05-21-12-27-22-552435
    log_dir = os.path.join(energyplus_logbase_dir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    if not os.path.exists(log_dir + '/output'):
        os.makedirs(log_dir + '/output')
    os.environ["ENERGYPLUS_LOG"] = log_dir
    model = os.getenv('ENERGYPLUS_MODEL')
    if model is None:
        print('Environment variable ENERGYPLUS_MODEL is not defined')
        sys.exit(1)
    weather = os.getenv('ENERGYPLUS_WEATHER')
    if weather is None:
        print('Environment variable ENERGYPLUS_WEATHER is not defined')
        sys.exit(1)

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print('train: init logger with dir={}'.format(log_dir)) #XXX
        logger.configure(log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    env = make_energyplus_env(env_id, workerseed)

    policy = trpo_mpi.learn(env=env,
                    network=mlp(num_hidden=32, num_layers=2),
                    total_timesteps=num_timesteps,
                    #timesteps_per_batch=1*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                    timesteps_per_batch=16*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                    gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    # policy.save("/root/rl-testbed-for-energyplus/model.pth")
    
    # Save model
    policy.policy_network.save("save_policy_network")
    policy.value_network.save("save_value_network")
    ckpt = tf.train.Checkpoint(policy)
    save_path = ckpt.save("save_ckpts_1/save_ckpt")
    print(save_path)

    # Load model
    # ac_space = env.action_space
    # load_policy_network = tf.keras.models.load_model("save_policy_network")
    # load_value_network = tf.keras.models.load_model("save_value_network")
    # new_p = PolicyWithValue(ac_space, load_policy_network, load_value_network)
    # new_ckpt = tf.train.Checkpoint(new_p)
    # new_ckpt.restore(save_path)

    env.close()

def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
