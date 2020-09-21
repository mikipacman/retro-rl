from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
import neptune

from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor
import MortalKombat2

from helpers.callbacks import get_callbacks
from helpers.saving_utils import save_exp_params


def make_experiment_env(params, train):
    clear = MortalKombat2.\
        make_mortal_kombat2_env(difficulties=params["difficulties"],
                                arenas=params["arenas"],
                                left_players=params["left_players"],
                                right_players=params["right_players"],
                                controllable_players=params["controllable_players"],
                                actions=params["actions"],
                                state_versions=params["state_versions"])
    env = FrameskipWrapper(clear, skip=params["frameskip"])

    if params["max_episode_length"]:
        env = MaxEpLenWrapper(env, max_len=params["params"] // params["frameskip"])

    env = WarpFrame(env, 48, 48)

    if train:
        env = Monitor(env, info_keywords=("P1_rounds", "P2_rounds", "P1_health", "P2_health", "steps",
                                          "difficulty", "arena", "P1", "P2", "state_version"))
        return env
    else:
        return clear, env, env


neptune.init("miki.pacman/MK2")
exp_name = "Framskip_tune_Raiden"
env_params = {
    'difficulties': ["Medium"],
    'arenas': ["DeadPool"],
    'left_players': ["Scorpion"],
    'right_players': ["Raiden"],
    'state_versions': list(range(16)),
    'actions': "ALL",
    'controllable_players': 1,
    'n_env': 16,
    'env_function': make_experiment_env,
}
learn_params = {
    'total_timesteps': int(2e7),
    'save_checkpoint_n_epoch': 10,
    'save_checkpoint_google_drive_path': "MK2/saves/",
    'send_video_n_epoch': 50,
    "algo_name": "PPO",
}
wrappers_params = {
    'frameskip': 5,
    'max_episode_length': None,
}
algo_params = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf":  None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "tensorboard_log": None,
    "create_eval_env": False,
    "policy_kwargs": None,
    "verbose": 0,
    "seed": None,
    "device": "auto",
}

if __name__ == "__main__":

    params = {
        **env_params,
        **learn_params,
        **wrappers_params,
        **algo_params,
    }

    with neptune.create_experiment(name=exp_name, params=params) as exp:
        save_exp_params(params)

        video_env_function = lambda: params["env_function"](params, train=False)
        env = SubprocVecEnv([lambda: params["env_function"](params, train=True)
                             for _ in range(params["n_env"])], start_method="forkserver")
        model = PPO(CnnPolicy, env, **algo_params)

        model.learn(total_timesteps=params["total_timesteps"], callback=get_callbacks(params, exp, video_env_function))
