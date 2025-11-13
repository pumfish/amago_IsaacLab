from argparse import ArgumentParser

import wandb

from amago.envs.builtin.isaaclab_task import IsaacLabTask
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.envs.exploration import BilevelEpsilonGreedy
from amago import cli_utils

import multiprocessing as mp
mp.set_start_method("spawn", force=True)


def add_cli(parser):
    parser.add_argument(
        "--env_name",
        type=str,
        default="Isaac-Grasp-Cube-Franka-DR",
        help="env_name for IsaacLab task",
    )
    # parser.add_argument("--k", type=int, default=3, help="K-Shots")
    parser.add_argument("--max_seq_len", type=int, default=250)
    parser.add_argument(
        "--hide_rl2s",
        action="store_true",
        help="hides the 'rl2 info' (previous actions, rewards)",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    cli_utils.add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    #TODO: How to set config
    config = {
        "amago.nets.tstep_encoders.FFTstepEncoder.hide_rl2s": args.hide_rl2s,
        # delete the next three lines to use the paper settings, which were
        # intentionally left wide open to avoid reward-specific tuning.
        "amago.nets.actor_critic.NCriticsTwoHot.min_return": -100.0,
        "amago.nets.actor_critic.NCriticsTwoHot.max_return": 5000,
        "amago.nets.actor_critic.NCriticsTwoHot.output_bins": 96,
    }
    traj_encoder_type = cli_utils.switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    agent_type = cli_utils.switch_agent(
        config,
        args.agent_type,
        reward_multiplier=1.0,
        num_critics=4
    )
    exploration_type = cli_utils.switch_exploration(
        config, "bilevel", steps_anneal=2_000_000, rollout_horizon=500
    )
    cli_utils.use_config(config, args.configs)

    #TODO: How to replace these params
    make_train_env = lambda: IsaacLabTask(args.env_name)

    group_name = (
        f"{args.run_name}_Isaaclab_{args.env_name}_L_{args.max_seq_len}"
    )
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = cli_utils.create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.max_seq_len,
            traj_save_len=args.max_seq_len * 4,
            group_name=group_name,
            run_name=run_name,
            tstep_encoder_type=FFTstepEncoder,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            val_timesteps_per_epoch=15 * 500 + 1,
            learning_rate=5e-4,
            grad_clip=2.0,
            exploration_wrapper_type=exploration_type,
        )

        experiment = cli_utils.switch_async_mode(experiment, args.mode)
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        # experiment.evaluate_test(make_test_env, timesteps=20_000, render=False)
        experiment.delete_buffer_from_disk()
        wandb.finish()