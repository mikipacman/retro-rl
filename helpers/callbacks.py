from stable_baselines3.common.callbacks import BaseCallback
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import tempfile
import os
import neptune
import time
import numpy as np
from helpers.interactive_env_recorder import InteractiveEnvRecorder
import base64


class GoogleDriveCheckpointCallback(BaseCallback):
    def __init__(self, exp, save_checkpoint_n_epoch, save_path, name_prefix="rl_model", verbose=0):
        super(GoogleDriveCheckpointCallback, self).__init__(verbose)
        self._exp = exp
        self._save_checkpoint_n_epoch = save_checkpoint_n_epoch
        self._save_path = save_path
        self._name_prefix = name_prefix
        self._drive = GoogleDrive(GoogleAuth())
        self._iteration = 0

    def _init_callback(self) -> None:
        parent_id = None
        for folder_name in self._save_path.strip("/").split("/"):
            parent_id = self._get_folder(folder_name, parent_id)
        self._exp_folder_id = self._get_folder(self._exp.id, parent_id)

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self._iteration % self._save_checkpoint_n_epoch == 0:
            name = f"{self._name_prefix}_{self._iteration}_{self.num_timesteps}_steps.zip"
            self._upload_checkpoint(name)

        self._iteration += 1

    def _on_training_end(self):
        name = f"{self._name_prefix}_last_{self.num_timesteps}_steps.zip"
        self._upload_checkpoint(name)

    def _upload_checkpoint(self, name):
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            path = os.path.join(temp, name)
            self.model.save(path)

            file = self._drive.CreateFile({'title': name, 'parents': [{"id": self._exp_folder_id}]})
            file.SetContentFile(path)
            file.Upload()

            if self.verbose > 1:
                print(f"uploaded {name} to {self._save_path.strip('/')}/{self._exp.id} on google drive ")

    def _add_folder(self, name, parent):
        meta = {'title': name, 'mimeType': 'application/vnd.google-apps.folder'}
        if parent:
            meta.update({"parents": [{"id": parent}]})
        folder = self._drive.CreateFile(meta)
        folder.Upload()
        return folder["id"]

    def _get_folder(self, title, parent):
        folder_list = self._drive.ListFile({'q': "trashed=false"}).GetList()

        if parent:
            folder_list = [f for f in folder_list if parent in f["parents"][0]["id"]]
        if title not in [f["title"] for f in folder_list]:
            folder_id = self._add_folder(title, parent)
        else:
            folder_id = [f["id"] for f in folder_list if f["title"] == title][0]
        return folder_id


class NeptuneLogger(BaseCallback):
    def __init__(self, exp, send_video_n_epoch, env_func, verbose=0):
        super(NeptuneLogger, self).__init__(verbose)
        self._exp = exp
        self._send_video_n_epoch = send_video_n_epoch
        self._env_func = env_func
        self._params = exp.get_parameters()

    def _on_training_start(self):
        self._context = self.locals["self"]
        self._iteration = 0

    def _on_rollout_end(self):
        context = self._context

        neptune.send_metric("time/fps", context.num_timesteps / (time.time() - context.start_time))
        neptune.send_metric("time/iterations", self._iteration)
        neptune.send_metric("time/time_elapsed", time.time() - context.start_time)
        neptune.send_metric("time/total_timesteps", context.num_timesteps)

        rollout_infos = [context.ep_info_buffer[i] for i in range(context.n_envs)]
        name_to_key = {
            "rollout/ep_rew": "r",
            "rollout/ep_len": "l",
            "rollout/p1_rounds": "P1_rounds",
            "rollout/p2_rounds": "P2_rounds",
            "rollout/p1_health": "P1_health",
            "rollout/p2_health": "P2_health",
            "rollout/real_ep_len": "steps",
        }

        for k, v in name_to_key.items():
            self._log_3m(k, self._get_by_key(rollout_infos, v))

        for k, v in self.logger.get_log_dict().items():
            neptune.send_metric(k, v)

        if self._iteration % self._send_video_n_epoch == 0:
            self._generate_eval_video()

        self._iteration += 1

    def _on_step(self):
        return True

    @staticmethod
    def _get_by_key(rollout_infos, key):
        return [d[key] for d in rollout_infos]

    @staticmethod
    def _log_3m(prefix, data):
        neptune.send_metric(prefix + "_mean", np.mean(data))
        neptune.send_metric(prefix + "_min", np.min(data))
        neptune.send_metric(prefix + "_max", np.max(data))

    def _generate_eval_video(self):
        env_main, env1, env2 = self._env_func()
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            path_to_video = os.path.join(temp, "movie.mp4")
            ia = InteractiveEnvRecorder(env=env_main,
                                        p1=self.model,
                                        p1_env=env1,
                                        p1_frameskip=self._params["frameskip"],
                                        p2="human",
                                        p2_env=env2,
                                        p2_frameskip=1,
                                        record_output_path=path_to_video,
                                        close_after_done=True,
                                        record_n_frames_after_done=300,
                                        resize_video=2,
                                        show_on_screen=False)
            ia.run()
            del ia

            encoded = base64.b64encode(open(path_to_video, "rb").read())
            html = f'<video controls><source type="video/mp4" ' \
                   f'src="data:video/mp4;base64,{encoded.decode("utf-8")}"></video>'
            open(path_to_video, "w+").write(html)

            neptune.send_artifact(path_to_video, f"movies/movie_{self._iteration}_{self._context.num_timesteps}.html")


def get_callbacks(params, exp, make_env_function):
    return [
        NeptuneLogger(exp=exp, send_video_n_epoch=params["send_video_n_epoch"], env_func=make_env_function),
        GoogleDriveCheckpointCallback(exp=exp, save_checkpoint_n_epoch=params["save_checkpoint_n_epoch"],
                                      save_path=params["save_checkpoint_google_drive_path"])
    ]
