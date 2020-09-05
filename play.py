from stable_baselines3.common.atari_wrappers import WarpFrame
from helpers.interactive_env_recorder import InteractiveEnvRecorder
from stable_baselines3.common.monitor import Monitor
import MortalKombat2
from MortalKombat2.wrappers import FrameskipWrapper, MaxEpLenWrapper


if __name__ == '__main__':
    params = {
        'difficulties': ["VeryEasy"],
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
