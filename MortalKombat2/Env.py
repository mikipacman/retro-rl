import gym
import numpy as np
import retro

available_opponents = [
    "Raiden",
    "Jax",
    "Baraka",
    "SubZero",
    "Scorpion"
]

available_arenas = [
    "DeadPool",
    "LivingForest",
    "Portal"
]

available_difficulties = [
    "VeryEasy",
    "Medium",
    "VeryHard",
]

class MK2Wrapper(gym.Wrapper):
    def __init__(self, env,
                 opp_health_factor=0.1,
                 own_health_factor=-0.1,
                 win_factor=10,
                 lose_factor=-10,
                 done_after="round"):
        super().__init__(env)
        self._previous_health = np.array([120, 120])
        self._previous_win = np.array([0, 0])
        self._previous_info = {}
        self._own_health_factor = own_health_factor
        self._opp_health_factor = opp_health_factor
        self._win_factor = win_factor
        self._lose_factor = lose_factor
        self._in_game = True
        self._players = env.action_space.shape[0] // 12
        self._done_after = done_after
        self._cumulative_reward = np.array([0.] * self._players)
        self._steps = 0
        assert done_after in ["match", "round"]

    def step(self, action):
        obs, rew, _, info = self.env.step(action)
        self._previous_info = dict(info)
        rew = self.reward(rew)
        done = max(self._previous_win) == (2 if self._done_after == "match" else 1)
        info = self._redefine_info(rew, info)
        self._steps += 1
        return obs, rew, done, info

    def reward(self, reward):
        info = self._previous_info
        if not info:
            return reward

        rew = np.array([0., 0.])

        if self._in_game:
            health = np.array([info["health"], info["enemy_health"]])
            delta_health = health - self._previous_health
            rew += delta_health * (-self._own_health_factor)
            rew += delta_health[::-1] * (-self._opp_health_factor)
            self._previous_health += delta_health

            win = np.array([info["rounds_won"], info["enemy_rounds_won"]])
            delta_win = win - self._previous_win
            rew += delta_win * self._win_factor
            rew += delta_win[::-1] * self._lose_factor
            self._previous_win += delta_win

            if max(delta_win) > 0:
                self._in_game = False
        elif min(info["health"], info["enemy_health"]) == 120:
            self._previous_health = np.array([120, 120])
            self._in_game = True

        return rew[0] if self._players == 1 else rew

    def _redefine_info(self, rew, info):
        self._cumulative_reward += rew

        info["P1_health"] = info["health"]
        info["P2_health"] = info["enemy_health"]
        info["P1_rounds"] = info["rounds_won"]
        info["P2_rounds"] = info["enemy_rounds_won"]
        del info["health"]
        del info["enemy_health"]
        del info["rounds_won"]
        del info["enemy_rounds_won"]
        del info["wins"]

        info["steps"] = self._steps
        info["cum_rew"] = self._cumulative_reward

        return info

    def reset(self, **kwargs):
        self._previous_health = np.array([120, 120])
        self._previous_win = np.array([0, 0])
        self._previous_info = {}
        self._in_game = True
        self._cumulative_reward = np.array([0.] * self._players)
        self._steps = 0
        return self.env.reset(**kwargs)

    def observation(self, frame):
        return frame


def make_mk2(state, players):
    env = retro.make(game="MortalKombatII-Genesis", state=state, players=players,)
    return MK2Wrapper(env)


class FrameskipWithRealGameTracker(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._real_images = []
        self._real_sounds = []
        self._real_infos = []

    def reset(self):
        self._real_images = []
        self._real_sounds = []
        self._real_infos = []

        return self.env.reset()

    def step(self, act):
        total_rew = 0.0
        done = None
        # self._real_images, self._real_sounds, self._real_infos = [], [], []
        for i in range(self._skip):
            obs, rew, done, info = self.env.step(act)
            total_rew += rew
            # self._real_images.append(self.env.render(mode="rgb_array"))
            # self._real_sounds.append(self.env.em.get_audio())
            # self._real_infos.append(info)

        return obs, total_rew, done, info

    def get_real_game_data(self):
        return self._real_images, self._real_sounds, self._real_infos


class MaxEpLenWrapper(gym.Wrapper):
    def __init__(self, env, max_len):
        super().__init__(env)
        self._max_len = max_len
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        o, r, d, i = self.env.step(action)
        if self._steps == self._max_len:
            return o, r, True, i
        self._steps += 1
        return o, r, d, i
