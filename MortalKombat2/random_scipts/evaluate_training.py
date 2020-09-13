from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import os
from helpers.saving_utils import get_exp_params

# Needed for loading pickle
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor
import MortalKombat2

project_name = "miki.pacman/sandbox"
exp_id = "SAN-2081"
if __name__ == "__main__":
    params = get_exp_params(exp_id, project_name)

    env = params["env_function"](params, train=True)
    x, y = [], []
    for root, _, files in os.walk(f"../saves/{exp_id}"):
        d = {int(file[len("rl_model_"):-len("_steps.zip")]): file for file in files}
        for xdd in sorted(d.keys())[::-1]:
            r = []
            for _ in range(8):
                file = d[xdd]
                model = PPO.load(os.path.join(root, file))
                done = False
                obs = env.reset()
                while not done:
                    obs, _, done, info = env.step(model.predict(obs)[0])
                    # env.render()
                    # time.sleep(1 / 12)
                print(info["episode"]["r"])
                r.append(info["episode"]["r"])

            y.append(np.mean(r))
            x.append(xdd)

    plt.plot(x, y)
    plt.show()