import sys
import ctypes
import argparse
import abc
import time

import retro
import pyglet
from pyglet import gl
from pyglet.window import key as keycodes
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3.common.atari_wrappers import WarpFrame


class Interactive(abc.ABC):
    """
    Base class for making gym environments interactive for human use
    """
    def __init__(self, env, sync=True, tps=60, width=1000):
        obs = env.reset()
        self._image = self.get_image(obs, 0, False, {}, env)
        assert len(self._image.shape) == 3 and self._image.shape[2] == 3, 'must be an RGB image'
        image_height, image_width = self._image.shape[:2]

        aspect_ratio = image_width / image_height
        win_width = width
        win_height = int(win_width / aspect_ratio)

        win = pyglet.window.Window(width=win_width, height=win_height)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, image_width, image_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        self._env = env
        self._win = win

        self._key_previous_states = {}

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

        self._sound_buffer = np.array([])
        self._rate = self._env.em.get_audio_rate() * 0.85  # for eliminating clipping in real time sound generation
        self._steps = 0
        self._sound_buffer_len = 12


    def _update(self, dt):
        # cap the number of frames rendered so we don't just spend forever trying to catch up on frames
        # if rendering is slow
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        # catch up the simulation to the current time
        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self._tps

            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)

                if not self._key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                self._key_previous_states[key_code] = pressed

            if keycodes.ESCAPE in keys_pressed:
                self._on_close()

            # assume that for async environments, we just want to repeat keys for as long as they are held
            inputs = keys_pressed
            if self._sync:
                inputs = keys_clicked

            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            act = self.keys_to_act(keys)

            if not self._sync or act is not None:
                obs, rew, done, _info = self._env.step(act)
                self._image = self.get_image(obs, rew, done, _info, self._env)
                self._steps += 1

                sounds = self._env.em.get_audio()[:, 0]
                if self._steps % self._sound_buffer_len == 0:
                    sd.play(self._sound_buffer / 10000, int(self._rate))
                    self._sound_buffer = np.array(sounds)
                else:
                    self._sound_buffer = np.append(self._sound_buffer, sounds)

                if done:
                    self._env.reset()

    def _draw(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(self._image.tobytes(), ctypes.POINTER(ctypes.c_short))
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self._image.shape[1], self._image.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, video_buffer)

        x = 0
        y = 0
        w = self._win.width
        h = self._win.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
            ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        self._env.close()
        sys.exit(0)

    @abc.abstractmethod
    def get_image(self, obs, rew, done, info, venv):
        pass

    @abc.abstractmethod
    def keys_to_act(self, keys):
        """
        Given a list of keys that the user has input, produce a gym action to pass to the environment

        For sync environments, keys is a list of keys that have been pressed since the last step
        For async environments, keys is a list of keys currently held down
        """
        pass

    def run(self):
        """
        Run the interactive window until the user quits
        """
        # pyglet.app.run() has issues like https://bitbucket.org/pyglet/pyglet/issues/199/attempting-to-resize-or-close-pyglet
        # and also involves inverting your code to run inside the pyglet framework
        # avoid both by using a while loop
        prev_frame_time = time.time()
        while True:
            self._win.switch_to()
            self._win.dispatch_events()
            now = time.time()
            self._update(now - prev_frame_time)
            prev_frame_time = now
            self._draw()
            self._win.flip()


class RetroInteractive(Interactive):
    """
    Interactive setup for retro games
    """

    def __init__(self, game, state, scenario, players, width=1200):
        env = retro.make(game=game, state=state, scenario=scenario, players=players)
        self._buttons = env.buttons
        self._players = players
        self._p1 = {}
        self._p2 = {}
        self._width = width
        self._timesteps = 0
        self._agent_frames = 0
        self._cum_rew = np.array([0., 0.])
        super().__init__(env=env, sync=False, tps=60, width=width)

    def get_image(self, obs, rew, done, info, env):
        skip = 10
        scale_up = 2.5
        frame = env.render(mode='rgb_array')
        real_game_img = Image.fromarray(frame)

        env2 = WarpFrame(env, width=64, height=48)
        obs = env2.observation(frame[50:200, :, :])
        obs = obs.reshape(obs.shape[:2])
        if self._timesteps % skip == 0:
            agent_img = Image.fromarray(obs, 'L')
            agent_img = agent_img.resize((int(agent_img.width * scale_up), int(agent_img.height * scale_up)))
            self._agent_img = agent_img
            self._agent_frames += 1
        else:
            agent_img = self._agent_img

        self._cum_rew += rew
        info["rew"] = self._cum_rew

        text_img = self._get_action_info_img()
        info_img = self._get_info_img(info)
        agent_img = self._get_concat_v(agent_img, info_img)
        img_to_show = self._get_concat_h(real_game_img, agent_img)
        img_to_show = self._get_concat_v(img_to_show, text_img)

        self._timesteps += 1

        return np.array(img_to_show)

    def _get_action_info_img(self):
        text_img = Image.new('RGB', (self._width, 50))
        d = ImageDraw.Draw(text_img)
        font = ImageFont.load_default()
        buttons = ["A", "B", "C", "X", "Y", "Z", "UP", "DOWN", "LEFT", "RIGHT"]

        def vec_to_str(vec):
            s = ""
            for a in vec[:6]:
                if a:
                    s += " x"
                else:
                    s += "  "

            for a in vec[6:7]:
                if a:
                    s += " x "
                else:
                    s += "   "

            for a in vec[7:]:
                if a:
                    s += "  x  "
                else:
                    s += "     "
            return s

        txt = f'    {" ".join(buttons)}\n' \
              f'P1:{vec_to_str([k in self._p1 and self._p1[k] for k in buttons])}\n' \
              f'P2:{vec_to_str([k in self._p2 and self._p2[k] for k in buttons])}'
        d.text((0, 0), txt, fill=(255, 0, 0), font=font)
        return text_img

    def _get_info_img(self, info):
        text_img = Image.new('RGB', (120, 120))
        if 'health' in info:
            info["P1_health"] = info["health"]
            info["P2_health"] = info["enemy_health"]
            del info['health']
            del info['enemy_health']

            info["P1_wins"] = info["wins"]
            info["P2_wins"] = info["enemy_rounds_won"]
            del info['wins']
            del info['enemy_rounds_won']

            info["steps"] = self._timesteps
            info["agent steps"] = self._agent_frames

        d = ImageDraw.Draw(text_img)
        font = ImageFont.load_default()
        txt = "\n".join([k + ": " + str(v) for k, v in info.items()])
        d.text((0, 0), txt, fill=(255, 0, 0), font=font)
        return text_img

    def keys_to_act(self, keys):
        inputs_1 = {
            'BUTTON': 'Z' in keys,
            'A': 'F' in keys,
            'B': 'G' in keys,
            'C': 'H' in keys,
            'X': 'R' in keys,
            'Y': 'T' in keys,
            'Z': 'Y' in keys,
            'L': 'Z' in keys,
            'R': 'C' in keys,
            'UP': 'W' in keys,
            'DOWN': 'S' in keys,
            'LEFT': 'A' in keys,
            'RIGHT': 'D' in keys,
            'MODE': 'TAB' in keys,
            'SELECT': 'Q' in keys,
            'RESET': 'ENTER' in keys,
            'START': 'E' in keys,
        }
        inputs_2 = {
            'BUTTON': 'Z' in keys,
            'A': 'NUM_1' in keys,
            'B': 'NUM_2' in keys,
            'C': 'NUM_3' in keys,
            'X': 'NUM_4' in keys,
            'Y': 'NUM_5' in keys,
            'Z': 'NUM_6' in keys,
            'L': 'Z' in keys,
            'R': 'C' in keys,
            'UP': 'UP' in keys,
            'DOWN': 'DOWN' in keys,
            'LEFT': 'LEFT' in keys,
            'RIGHT': 'RIGHT' in keys,
            'MODE': 'TAB' in keys,
            'SELECT': 'NUM_7' in keys,
            'RESET': 'ENTER' in keys,
            'START': 'NUM_9' in keys,
        }

        self._p1 = inputs_1
        self._p2 = inputs_2

        acc_vec = [inputs_1[b] for b in self._buttons]
        if self._players == 2:
            acc_vec += [inputs_2[b] for b in self._buttons]
        return acc_vec

    @staticmethod
    def _get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    @staticmethod
    def _get_concat_v(im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='MortalKombatII-Genesis')
    parser.add_argument('--state', default='Jax_vs_SubZero_2p')
    parser.add_argument('--players', default=2)
    parser.add_argument('--scenario', default=None)
    args = parser.parse_args()

    ia = RetroInteractive(game=args.game, state=args.state,
                          scenario=args.scenario, players=args.players)
    ia.run()


if __name__ == '__main__':
    main()