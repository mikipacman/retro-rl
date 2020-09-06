from stable_baselines3.common.atari_wrappers import WarpFrame
from helpers.interactive_env_recorder import InteractiveEnvRecorder
from stable_baselines3.common.monitor import Monitor
import MortalKombat2
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper



exp_name = "VeryHard_Raiden"
exp_id = "MK-4"
env_params = {
    'difficulties': ["VeryHard"],
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

if __name__ == '__main__':

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

    env1, env2, env3 = make_env(params, train=False)
    ia = InteractiveEnvRecorder(env=env1,
                                p1="human",
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
