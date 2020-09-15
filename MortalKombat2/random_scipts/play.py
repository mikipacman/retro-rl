import tempfile
import os

from stable_baselines3.ppo import PPO

from helpers.interactive_env_recorder import PygameInteractiveEnvRecorder
from helpers.saving_utils import get_exp_params, GoogleDriveCheckpointer

# Needed for loading pickle
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor
import MortalKombat2


project_name = "miki.pacman/MK2"
google_drive_checkpoints_path = "MK2/saves"
exp_id = "MK-19"
params = get_exp_params(exp_id, project_name)

params.update({"state_versions": [16, 17, 18, 19]})

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(dir="/tmp") as temp:
        checkpointer = GoogleDriveCheckpointer(project_experiments_path=google_drive_checkpoints_path, exp_id=exp_id)
        checkpoints_list = checkpointer.get_list_of_checkpoints()
        checkpoint = checkpoints_list[len(checkpoints_list) // 2]
        checkpointer.download_checkpoints([checkpoint], temp)

        env1, env2, env3 = params["env_function"](params, train=False)
        model = PPO.load(os.path.join(temp, checkpoint))
        p1 = {
            "policy": model,
            "frameskip": params["frameskip"],
            "env": env2
        }
        p2 = {
            "policy": "human",
            "frameskip": 60,
            "env": env3
        }

        for i in range(4):
            PygameInteractiveEnvRecorder(fps=60, env=env1, p1=p1, p2=p2, render_n_frames_after_done=250,
                                         record_output_path=f"/tmp/{exp_id}_video_{i}.mp4").run()
