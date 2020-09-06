import MortalKombat2
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
import neptune

from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor

from helpers.callbacks import get_callbacks


def make_env(params, train=True):
    clear = MortalKombat2. \
        make_mortal_kombat2_env(difficulties=params["difficulties"],
                                arenas=params["arenas"],
                                left_players=params["left_players"],
                                right_players=params["right_players"],
                                controllable_players=params["controllable_players"],
                                actions=params["actions"])

    env = FrameskipWrapper(clear, skip=params["frameskip"])

    if params["max_episode_length"]:
        env = MaxEpLenWrapper(env, max_len=params["params"] // params["frameskip"])

    env = WarpFrame(env, 48, 48)

    if train:
        env = Monitor(env, info_keywords=("P1_rounds", "P2_rounds", "P1_health", "P2_health", "steps"))
        return env
    else:
        return clear, env, env


neptune.init("miki.pacman/MK2")
exp_name = "3_arenas_Raiden"
env_params = {
    'difficulties': ["VeryEasy"],
    'arenas': ["DeadPool", "LivingForest", "Portal"],
    'left_players': ["Scorpion"],
    'right_players': ["Raiden"],
    'actions': "ALL",
    'controllable_players': 1,
    'n_env': 16,
}
learn_params = {
    'total_timesteps': int(1e7),
    'save_checkpoint_n_epoch': 5,
    'save_checkpoint_google_drive_path': "MK2/saves/",
    'send_video_n_epoch': 25,
}
wrappers_params = {
    'frameskip': 10,
    'max_episode_length': None,
}
algo_params = {
    "algo_name": "PPO",
    "learning_rate": 3e-4,
    "n_steps": 2048,
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
        env = SubprocVecEnv([lambda: make_env(params) for _ in range(params["n_env"])], start_method="forkserver")
        model = PPO(CnnPolicy, env)

        f = lambda: make_env(params, train=False)
        model.learn(total_timesteps=params["total_timesteps"], callback=get_callbacks(params, exp, f))
