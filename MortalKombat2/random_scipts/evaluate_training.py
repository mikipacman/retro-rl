from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

from helpers.saving_utils import get_exp_params, GoogleDriveCheckpointer

# Needed for loading pickle
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor
import MortalKombat2

project_name = "miki.pacman/sandbox"
google_drive_checkpoints_path = "MK2/saves"
exp_id = "SAN-2089"
params = get_exp_params(exp_id, project_name)


with tempfile.TemporaryDirectory(dir="/tmp") as temp:
    checkpointer = GoogleDriveCheckpointer(project_experiments_path=google_drive_checkpoints_path, exp_id=exp_id)
    checkpoints_list = checkpointer.get_list_of_checkpoints()
    checkpointer.download_checkpoints(checkpoints_list, temp)

    env = params["env_function"](params, train=True)

    x, y = [], []
    for root, _, files in os.walk(temp):
        d = {int(file[len("rl_model_"):-len("_steps.zip")]): file for file in files[:-1]}
        for xdd in sorted(d.keys())[::-1]:
            r = []
            for _ in range(8):
                file = d[xdd]
                model = PPO.load(os.path.join(root, file))
                done = False
                obs = env.reset()
                while not done:
                    obs, _, done, info = env.step(model.predict(obs)[0])
                print(info["episode"]["r"])
                r.append(info["episode"]["r"])

            y.append(np.mean(r))
            x.append(xdd)

    plt.plot(x, y)
    plt.show()