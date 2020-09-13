from stable_baselines3 import PPO
import os
import tempfile
import time

from helpers.saving_utils import get_exp_params, GoogleDriveCheckpointer

# Needed for loading pickle
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor
import MortalKombat2

project_name = "miki.pacman/MK2"
google_drive_checkpoints_path = "MK2/saves"
exp_id = "MK-16"
params = get_exp_params(exp_id, project_name)


with tempfile.TemporaryDirectory(dir="/tmp") as temp:
    checkpointer = GoogleDriveCheckpointer(project_experiments_path=google_drive_checkpoints_path, exp_id=exp_id)
    checkpoints = checkpointer.get_list_of_checkpoints()
    checkpoint = checkpoints[-1]
    checkpointer.download_checkpoints([checkpoint], temp)

    env = params["env_function"](params, train=True)
    model = PPO.load(os.path.join(temp, checkpoint))

    print(checkpoints)
    print(f"loaded {checkpoint} checkpoint")

    done = False
    obs = env.reset()
    while not done:
        obs, _, done, info = env.step(model.predict(obs)[0])
        env.render()
        time.sleep(1 / 12)

    print(info["episode"])
