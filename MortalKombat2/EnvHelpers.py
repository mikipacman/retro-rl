import gym
import numpy as np


class MK2(gym.Wrapper):
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
        self._done_after = done_after
        assert done_after in ["match", "round"]

    def step(self, action):
        obs, rew, _, info = self.env.step(action)
        self._previous_info = dict(info)
        rew = self.reward(rew)
        done = max(self._previous_win) == (2 if self._done_after == "match" else 1)
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

        return rew

