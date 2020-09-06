import MortalKombat2
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import time
import os


exp_name = "VeryHard_Raiden"
exp_id = "MK-5"
env_params = {
    'difficulties': ["VeryEasy", "Medium", "VeryHard"],
    'arenas': ["LivingForest"],
    'left_players': ["Scorpion"],
    'right_players': ["Raiden"],
    'actions': "ALL",
    'controllable_players': 1,
    'n_env': 16,
}
learn_params = {
    'total_timesteps': int(1e7),
    'saving_freq': int(1e4),
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
params = {
    **env_params,
    **learn_params,
    **wrappers_params,
    **algo_params,
}


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


env = make_env(params)
x, y = [], []
for root, _, files in os.walk(f"../saves/{exp_id}"):
    d = {int(file[len("rl_model_"):-len("_steps.zip")]): file for file in files}
    for xdd in sorted(d.keys())[::-1]:
        r = []
        for _ in range(16):
            file = d[xdd]
            model = PPO.load(os.path.join(root, file))
            done = False
            obs = env.reset()
            while not done:
                obs, _, done, info = env.step(model.predict(obs)[0])
                env.render()
                time.sleep(1 / 12)
            print(info["episode"]["r"])
            r.append(info["episode"]["r"])

        y.append(np.mean(r))
        x.append(xdd)

plt.plot(x, y)
plt.show()