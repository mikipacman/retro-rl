import tempfile
import os

from stable_baselines3.ppo import PPO

from helpers.interactive_env_recorder import InteractiveEnvRecorder
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


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(dir="/tmp") as temp:
        checkpointer = GoogleDriveCheckpointer(project_experiments_path=google_drive_checkpoints_path, exp_id=exp_id)
        checkpoints_list = checkpointer.get_list_of_checkpoints()
        checkpoint = checkpoints_list[0]
        checkpointer.download_checkpoints([checkpoint], temp)

        env1, env2, env3 = params["env_function"](params, train=False)
        model1 = PPO.load(os.path.join(temp, checkpoint))

        ia = InteractiveEnvRecorder(env=env1,
                                    p1=model1,
                                    p1_env=env2,
                                    p1_frameskip=params["frameskip"],
                                    p2="human",
                                    p2_env=env3,
                                    p2_frameskip=1,
                                    record_output_path=None,#"/tmp/test.mp4",
                                    close_after_done=True,
                                    record_n_frames_after_done=300,
                                    resize_video=2,
                                    show_on_screen=True)
        ia.run()
