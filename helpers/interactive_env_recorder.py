import ctypes
import abc
import time

import pyglet
from pyglet import gl
from pyglet.window import key as keycodes
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont
import tempfile
import cv2
import moviepy.editor as mpe
import soundfile as sf


class RealTimeVideoPlayer(abc.ABC):
    """
    Base class for making gym environments interactive for human use
    """
    def __init__(self, sync=True, tps=60, width=1000, sound_buffer_len=12, show_on_screen=True):
        if not show_on_screen:
            tps = 1000

        if show_on_screen:
            self._image, _ = self.get_image_and_sound([])
            assert len(self._image.shape) == 3 and self._image.shape[2] == 3, 'must be an RGB image'
            image_height, image_width = self._image.shape[:2]

            aspect_ratio = image_width / image_height
            win_width = width
            win_height = int(win_width / aspect_ratio)

            win = pyglet.window.Window(width=win_width, height=win_height)

            self._key_handler = pyglet.window.key.KeyStateHandler()
            win.push_handlers(self._key_handler)
            win.on_close = self.on_close

            gl.glEnable(gl.GL_TEXTURE_2D)
            self._texture_id = gl.GLuint(0)
            gl.glGenTextures(1, ctypes.byref(self._texture_id))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, image_width, image_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

            self._win = win

        self._key_previous_states = {}

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

        self._sound_buffer = np.array([])
        self._rate = self.get_audio_rate()
        self._steps = 0
        self._sound_buffer_len = sound_buffer_len

        self._show_on_screen = show_on_screen
        self._continue = True

    def _update(self, dt):
        # cap the number of frames rendered so we don't just spend forever trying to catch up on frames
        # if rendering is slow
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        # catch up the simulation to the current time
        self._current_time += dt
        while self._sim_time < self._current_time and self._continue:
            self._sim_time += 1 / self._tps

            if self._show_on_screen:

                keys_clicked = set()
                keys_pressed = set()
                for key_code, pressed in self._key_handler.items():
                    if pressed:
                        keys_pressed.add(key_code)

                    if not self._key_previous_states.get(key_code, False) and pressed:
                        keys_clicked.add(key_code)
                    self._key_previous_states[key_code] = pressed

                if keycodes.ESCAPE in keys_pressed:
                    self.on_close()

                # assume that for async environments, we just want to repeat keys for as long as they are held
                inputs = keys_pressed
                if self._sync:
                    inputs = keys_clicked

            keys = []
            if self._show_on_screen:
                for keycode in inputs:
                    for name in dir(keycodes):
                        if getattr(keycodes, name) == keycode:
                            keys.append(name)

            self._image, sounds = self.get_image_and_sound(keys)
            self._steps += 1

            if self._show_on_screen:
                if self._steps % self._sound_buffer_len == 0:
                    sd.play(self._sound_buffer / 10000, int(self._rate))
                    self._sound_buffer = np.array(sounds)
                else:
                    self._sound_buffer = np.append(self._sound_buffer, sounds)

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

    @abc.abstractmethod
    def on_close(self):
        pass

    @abc.abstractmethod
    def get_image_and_sound(self, keys):
        pass

    @abc.abstractmethod
    def get_audio_rate(self):
        pass

    def run(self):
        """
        Run the interactive window until the user quits
        """
        # pyglet.app.run() has issues like https://bitbucket.org/pyglet/pyglet/issues/199/attempting-to-resize-or-close-pyglet
        # and also involves inverting your code to run inside the pyglet framework
        # avoid both by using a while loop
        prev_frame_time = time.time()
        while self._continue:
            if self._show_on_screen:
                self._win.switch_to()
                self._win.dispatch_events()
            now = time.time()
            self._update(now - prev_frame_time)
            prev_frame_time = now
            if self._show_on_screen:
                self._draw()
                self._win.flip()


class InteractiveEnvRecorder(RealTimeVideoPlayer):
    def __init__(self, env,
                 p1, p2,
                 p1_frameskip, p2_frameskip,
                 p1_env, p2_env,
                 width=1200,
                 record_output_path=None,
                 record_n_frames_after_done=0,
                 close_after_done=False,
                 resize_video=1,
                 show_on_screen=True):
        self._env = env
        self._p1 = p1
        self._p2 = p2
        self._p1_frameskip = p1_frameskip
        self._p2_frameskip = p2_frameskip
        self._p1_env = p1_env
        self._p2_env = p2_env
        self._p1_frame = None
        self._p2_frame = None
        self._p1_act = None
        self._p2_act = None

        self._buttons = env.buttons
        self._players = env.action_space.shape[0] // 12  # Probably Genesis-specific but who cares
        self._width = width
        self._steps = 0

        if record_output_path:
            assert close_after_done

        self._record_output_path = record_output_path
        self._n_frames_to_close_left = record_n_frames_after_done
        self._close_after_done = close_after_done
        self._should_be_closed = False
        self._resize = resize_video

        self._images = []
        self._sounds = np.array([])

        self._env.reset()

        super().__init__(sync=False, tps=60, width=width, show_on_screen=show_on_screen)

    def get_audio_rate(self):
        return self._env.em.get_audio_rate() * 0.85  # for eliminating clipping in real time sound generation

    def on_close(self):
        if self._record_output_path:
            # Ugly as hell but I don't care, it works :>
            audio_rate = self._env.em.get_audio_rate()
            resized_shape = (int(self._images[0].shape[1] * self._resize), int(self._images[0].shape[0] * self._resize))
            with tempfile.TemporaryDirectory(dir="/tmp") as temp:
                fps = len(self._images) / (len(self._sounds) / audio_rate)
                writer = cv2.VideoWriter(temp + '/tmp_vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, resized_shape)

                for img in self._images:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, resized_shape, interpolation=cv2.INTER_AREA)
                    writer.write(img)

                sf.write(temp + '/tmp_sound.flac', self._sounds / (max(self._sounds) * 2), int(audio_rate))

                cv2.destroyAllWindows()
                writer.release()

                my_clip = mpe.VideoFileClip(temp + '/tmp_vid.mp4')
                audio_background = mpe.AudioFileClip(temp + '/tmp_sound.flac')
                final_clip = my_clip.set_audio(audio_background)
                final_clip.write_videofile(self._record_output_path, logger=None)

        self._env.close()
        self._continue = False

    def get_image_and_sound(self, keys):
        main_frame = self._env.render(mode="rgb_array")
        act = self._keys_to_act(keys)

        if self._steps % self._p1_frameskip == 0:
            self._p1_frame = self._p1_env.observation(main_frame)
            if self._p1 != "human":
                self._p1_act = self._p1.predict(self._p1_frame)[0]
                act[:12] = self._p1_act
        else:
            if self._p1 != "human":
                act[:12] = self._p1_act

        if self._steps % self._p2_frameskip == 0:
            self._p2_frame = self._p2_env.observation(main_frame)
            if self._p2 != "human":
                self._p2_act = self._p2.predict(self._p2_frame)[0]
                act[12:] = self._p2_act
        else:
            if self._p2 != "human":
                act[12:] = self._p2_act

        # Make step
        _, _, done, info = self._env.step(act)

        # Render img
        main_img = Image.fromarray(main_frame)
        p1_img = self._get_player_img(self._p1_frame)
        p2_img = self._get_player_img(self._p2_frame)

        img = self._get_concat_h(p1_img, main_img)
        img = self._get_concat_h(img, p2_img)

        action_info_img = self._get_action_info_img(act)
        info_img = self._get_info_img(info)
        info_img = self._get_concat_h(action_info_img, info_img)
        img = self._get_concat_v(img, info_img)

        # Final output
        self._steps += 1
        image, sound = np.array(img), self._env.em.get_audio()[:, 0]

        # Save for recording if needed
        if self._record_output_path:
            self._images.append(image)
            self._sounds = np.append(self._sounds, sound)

        # Decide whether to close
        if self._close_after_done and done:
            self._should_be_closed = True

        if self._should_be_closed:
            if self._n_frames_to_close_left == 0:
                self.on_close()
            self._n_frames_to_close_left -= 1

        return image, sound

    def _get_action_info_img(self, act):
        text_img = Image.new('RGB', (275, 50))
        d = ImageDraw.Draw(text_img)
        font = ImageFont.load_default()
        buttons = ["A", "B", "C", "X", "Y", "Z", "UP", "DOWN", "LEFT", "RIGHT", "START", "MODE"]

        # I know its ugly
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

            for a in vec[7:9]:
                if a:
                    s += "  x  "
                else:
                    s += "     "

            for a in vec[9:]:
                if a:
                    s += "   x  "
                else:
                    s += "      "

            return s

        txt = f'    {" ".join(buttons)}\n' \
              f'P1:{vec_to_str([act[self._buttons.index(k)] for k in buttons])}\n' \
              f'P2:{vec_to_str([act[self._buttons.index(k) + 12] for k in buttons])}'
        d.text((0, 0), txt, fill=(255, 0, 0), font=font)
        return text_img

    @staticmethod
    def _get_player_img(frame):
        mode = "RGB"
        if len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = frame.reshape(frame.shape[:2])
            mode = "L"
        return Image.fromarray(frame, mode)

    @staticmethod
    def _get_info_img(info):
        text_img = Image.new('RGB', (120, 100))
        d = ImageDraw.Draw(text_img)
        font = ImageFont.load_default()
        txt = "\n".join([k + ": " + str(v) for k, v in info.items()])
        d.text((0, 0), txt, fill=(255, 0, 0), font=font)
        return text_img

    def _keys_to_act(self, keys):
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
            'MODE': 'Q' in keys,
            'SELECT': 'TAB' in keys,
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
            'MODE': 'NUM_7' in keys,
            'SELECT': 'TAB' in keys,
            'RESET': 'ENTER' in keys,
            'START': 'NUM_9' in keys,
        }

        return np.array([inputs_1[b] for b in self._buttons] + [inputs_2[b] for b in self._buttons])

    @staticmethod
    def _get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    @staticmethod
    def _get_concat_v(im1, im2):
        dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst
