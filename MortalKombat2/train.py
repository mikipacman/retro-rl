import MortalKombat2
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.atari_wrappers import WarpFrame
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import neptune
import numpy as np
import tempfile
from helpers.interactive_env_recorder import InteractiveEnvRecorder
import base64
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper


class NeptuneLogger(BaseCallback):
    def __init__(self, exp, send_video_n_epoch, env_func, verbose=0):
        super(NeptuneLogger, self).__init__(verbose)
        self._exp = exp
        self._send_video_n_epoch = send_video_n_epoch
        self._env_func = env_func

    def _on_training_start(self):
        self._context = self.locals["self"]
        self._iteration = 0

    def _on_training_end(self):
        pass

    def _on_rollout_start(self):
        pass

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
            ia = InteractiveEnvRecorder(env=env_main,
                                        p1=model,
                                        p1_env=env1,
                                        p1_frameskip=params["frameskip"],
                                        p2="human",
                                        p2_env=env2,
                                        p2_frameskip=1,
                                        record_output_path=temp + "/movie.mp4",
                                        close_after_done=True,
                                        record_n_frames_after_done=300,
                                        resize_video=2,
                                        show_on_screen=False)
            ia.run()
            del ia

            encoded = base64.b64encode(open(temp + "/movie.mp4", "rb").read())
            html = f'<video controls><source type="video/mp4" ' \
                   f'src="data:video/mp4;base64,{encoded.decode("utf-8")}"></video>'
            open(temp + "/movie.html", "w+").write(html)

            neptune.send_artifact(temp + "/movie.html", f"movie_{self._context.num_timesteps}.html")


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


if __name__ == "__main__":
    neptune.init("miki.pacman/MK2")

    exp_name = "VeryEasy_Raiden"
    env_params = {
        'difficulties': ["VeryEasy"],
        'arenas': ["DeadPool"],
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

    with neptune.create_experiment(name=exp_name, params=params) as exp:
        callbacks = [
            NeptuneLogger(exp=exp, send_video_n_epoch=params["send_video_n_epoch"],
                          env_func=lambda: make_env(params, train=False)),
            CheckpointCallback(save_freq=params["saving_freq"], save_path=f"saves/{exp.id}", verbose=2)
        ]

        env = SubprocVecEnv([lambda: make_env(params) for _ in range(params["n_env"])], start_method="forkserver")
        model = PPO(CnnPolicy, env)

        model.learn(total_timesteps=params["total_timesteps"], callback=callbacks)
