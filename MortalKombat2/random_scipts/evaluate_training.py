import MortalKombat2
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import time
import os


exp_name = "Medium_Raiden"
params = {
    'difficulties': ["Medium"],
    'arenas': ["DeadPool"],
    'left_players': ["Scorpion"],
    'right_players': ["Raiden"],
    'frameskip': 10,
    'actions': "ALL",
    'max_episode_length': None,
    'n_env': 16,
    'controllable_players': 1,
    'total_timesteps': int(1e7),
    'saving_freq': int(1e4),
    'send_video_n_epoch': 25,
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
for root, _, files in os.walk(f"saves/{exp_name}"):
    d = {int(file[len("rl_model_"):-len("_steps.zip")]): file for file in files}
    for xdd in sorted(d.keys()):
        r = []
        for _ in range(5):
            file = d[xdd]
            model = PPO.load(os.path.join(root, file))
            done = False
            obs = env.reset()
            while not done:
                obs, _, done, info = env.step(model.predict(obs)[0])

            r.append(info["episode"]["r"])

        y.append(np.mean(r))
        x.append(xdd)

plt.plot(x, y)
plt.show()