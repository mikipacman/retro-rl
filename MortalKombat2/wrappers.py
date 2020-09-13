import gym
import numpy as np


class MK2Wrapper(gym.Wrapper):
    def __init__(self, env, states,
                 opp_health_factor,
                 own_health_factor,
                 win_factor,
                 lose_factor,
                 done_after):
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
        self._states = states
        self._current_state = np.random.choice(self._states)

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

        if self._current_state[0] == '1':
            _, d, a, l, _, r, v = self._current_state.split("_")
            info["difficulty"] = d
            info["arena"] = a
            info["P1"] = l
            info["P2"] = r
            info["state_version"] = v
        elif self._current_state[0] == '2':
            _, a, l, _, r = self._current_state.split("_")
            info["arena"] = a
            info["P1"] = l
            info["P2"] = r

        return info

    def reset(self, **kwargs):
        self._previous_health = np.array([120, 120])
        self._previous_win = np.array([0, 0])
        self._previous_info = {}
        self._in_game = True
        self._cumulative_reward = np.array([0.] * self._players)
        self._steps = 0
        self._current_state = np.random.choice(self._states)
        self.env.load_state( self._current_state)
        return self.env.reset(**kwargs)

    def observation(self, frame):
        return frame


class FrameskipWrapper(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        assert skip >= 1
        self._skip = skip

    def step(self, act):
        total_rew = 0.0
        for i in range(self._skip):
            obs, rew, done, info = self.env.step(act)
            total_rew += rew

        return obs, total_rew, done, info


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
        else:
            self._steps += 1
            return o, r, d, i
