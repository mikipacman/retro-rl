from MortalKombat2.Env import make_mk2
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.atari_wrappers import WarpFrame
import time
from MortalKombat2.Env import FrameskipWithRealGameTracker, MaxEpLenWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import neptune
import numpy as np
import tempfile
from helpers.video import render_video
import base64

class NeptuneLogger(BaseCallback):
    def __init__(self, params, exp_name, send_video_n_steps, env_func, verbose=0):
        super(NeptuneLogger, self).__init__(verbose)
        neptune.init("miki.pacman/sandbox")
        self._params = params
        self._exp_name = exp_name
        self._send_video_n_steps = send_video_n_steps
        self._env_func = env_func

    def _on_training_start(self):
        self._context = self.locals["self"]
        self._iteration = 0
        list_of_params = ["learning_rate", "n_steps", "batch_size", "n_epochs",
                          "gamma", "gae_lambda", "clip_range", "clip_range_vf",
                          "ent_coef", "vf_coef", "max_grad_norm", "use_sde",
                          "sde_sample_freq", "target_kl", "policy_kwargs",
                          "seed", "device"]
        self._params.update({param: getattr(self._context, param) for param in list_of_params})
        neptune.create_experiment(self._exp_name, params=self._params)

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

        self._iteration += 1

    def _on_step(self):
        if self.n_calls % self._send_video_n_steps == 0:
            with tempfile.TemporaryDirectory(dir="/tmp") as temp:
                def step(obs):
                    return self.model.predict(obs)[0]
                render_video(self._env_func(), step, temp + "/movie.mp4", int(1e9))
                encoded = base64.b64encode(open(temp + "/movie.mp4", "rb").read())
                html = f'<video controls><source type="video/mp4" src="data:video/mp4;base64,{encoded.decode("utf-8")}"></video>'
                open(temp + "/movie.html", "w+").write(html)

                neptune.send_artifact(temp + "/movie.html", f"movie_{self.n_calls}.html")

        return True

    @staticmethod
    def _get_by_key(rollout_infos, key):
        return [d[key] for d in rollout_infos]

    @staticmethod
    def _log_3m(prefix, data):
        neptune.send_metric(prefix + "_mean", np.mean(data))
        neptune.send_metric(prefix + "_min", np.min(data))
        neptune.send_metric(prefix + "_max", np.max(data))


if __name__ == "__main__":
    exp_name = "MK2_KungLao_vs_Scorpion_VeryHard_1p"
    params = {
        'n_env': 16,
        'state': "KungLao_vs_Scorpion_VeryHard_1p",
        'players': 1,
        'frameskip': 10,
        # 'episode_max_len': 150,
        'total_timesteps': int(1e7),
        'saving_freq': int(1e4),
    }

    def make_env(params):
        env = make_mk2(state=params["state"], players=params["players"])
        env = WarpFrame(env, 48, 48)
        env = FrameskipWithRealGameTracker(env, skip=params["frameskip"])
        # env = MaxEpLenWrapper(env, params["episode_max_len"])
        return Monitor(env, info_keywords=("P1_rounds", "P2_rounds", "P1_health", "P2_health", "steps"))


    env = SubprocVecEnv([lambda: make_env(params) for _ in range(params["n_env"])], start_method="forkserver")
    model = PPO(CnnPolicy, env)

    model.learn(total_timesteps=params["total_timesteps"],
                callback=[
                    NeptuneLogger(params, exp_name, send_video_n_steps=int(1e6), env_func=lambda: make_env(params)),
                    CheckpointCallback(save_freq=params["saving_freq"],
                                       save_path="saves/KungLao_vs_Scorpion_VeryHard_1p", verbose=2)
                ])
