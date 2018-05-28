#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import baselines.ppo2.ppo2 as ppo2
import gym_remote.exceptions as gre
import functools
import argparse
import sonic_util
import pandas as pd
from time import sleep
from baselines import logger
from multiprocessing import cpu_count
from baselines.ppo2.policies import LstmPolicy, CnnPolicy


def add_boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.
    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def decay(start, end):
    delta = start - end

    def lr_decay(frac):
        return start - frac * delta

    return lr_decay

def main(policy, clients_fn, total_timesteps=int(5e7), weights_path=None,
         adam_stats='all', save_interval=0, nmixup=3):
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=policy,
                   env=clients_fn,
                   nsteps=2046 if policy == LstmPolicy else 4500,
                   nminibatches=1,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=20,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=decay(5e-4, 1e-4),
                   cliprange=lambda _: 0.1,
                   total_timesteps=total_timesteps,
                   save_interval=save_interval,
                   weights_path=weights_path,
                   adam_stats=adam_stats,
                   nmixup=nmixup)


def run_train():
    def _parse_args():
        parser = argparse.ArgumentParser(description="Run commands")
        parser.add_argument(
            '--csv_file', type=str, default=None,
            help="Csv file with train games. If None - connect to tmp/socket ")
        parser.add_argument(
            '--num_envs', type=int, default=int(1.5 * cpu_count()) - 1,
            help="Number of parallele environments. Only if csv file is provided.")
        parser.add_argument(
            '--nmixup', type=int, default=3,
            help="Number of mixup environments.")
        parser.add_argument(
            '--weights_path', type=str, default=None,
            help="filename with weights")
        parser.add_argument(
            '--steps', type=int, default=int(10e7),
            help="Number of steps in environment.")
        parser.add_argument(
            '--save_interval', type=int, default=0,
            help="Periodicity of saving weights.")
        parser.add_argument(
            '--policy', type=str, default='cnn', choices=['lstm', 'cnn'],
            help="Policy to use.")
        parser.add_argument(
            '--adam_stats', default='weight_stats', choices=['all', 'weight_stats', 'none'],
            help="Adams params to restore.")
        parser.add_argument(
            '--exp_const', type=float, default=0.005,
            help="Exploration constant.")
        parser.add_argument(
            '--exp_type', type=str, default='x', choices=['x', 'obs', 'none'],
            help="Exploration type.")
        add_boolean_flag(
            parser, 'gray', True,
            help="Convert image to grayscale.")
        return parser.parse_args()

    args = _parse_args()

    if args.policy == 'lstm':
        policy = LstmPolicy
        stack = False
    elif args.policy == 'cnn':
        policy = CnnPolicy
        stack = True
    else:
        raise ValueError()

    if args.csv_file is not None:
        game_states = pd.read_csv(args.csv_file).values.tolist()
        clients_fn = SubprocVecEnv([
            functools.partial(
                sonic_util.make_rand_env, game_states, stack,
                gray=args.gray,
                exp_type=args.exp_type,
                exp_const=args.exp_const,)
            for _ in range(args.num_envs)])
        nmixup = args.nmixup
    else:
        clients_fn = DummyVecEnv([functools.partial(
            sonic_util.make_remote_env, stack,
            gray=args.gray,
            exp_const=args.exp_const,
            exp_type=args.exp_type,
            socket_dir="tmp/sock")
        ])
        nmixup = 0

    sleep(2)
    logger.configure('logs')
    main(policy, clients_fn, args.steps, args.weights_path, args.adam_stats, args.save_interval, nmixup)


if __name__ == '__main__':
    try:
        run_train()
    except gre.GymRemoteError as exc:
        print('exception', exc)
