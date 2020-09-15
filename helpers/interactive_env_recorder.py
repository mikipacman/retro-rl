import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import cv2
import moviepy.editor as mpe
import soundfile as sf

import pygame


class PygameInteractiveEnvRecorder():
    inputs = {
        'BUTTON': 'Z',
        'A': 'F',
        'B': 'G',
        'C': 'H',
        'X': 'R',
        'Y': 'T',
        'Z': 'Y',
        'L': 'Z',
        'R': 'C',
        'UP': 'W',
        'DOWN': 'S',
        'LEFT': 'A',
        'RIGHT': 'D',
        'MODE': 'Q',
        'SELECT': 'TAB',
        'RESET': 'ENTER',
        'START': 'E',
    }

    def __init__(self, fps, env, p1, p2, win_size=(640 * 2, 480 * 2), render=True,
                 render_n_frames_after_done=0, record_output_path=None):
        self.fps = fps
        self.env = env
        self.buttons = env.buttons
        self.win_size = (int(win_size[0]), int(win_size[1]))
        self.render = render
        self.render_n_frames_after_done = render_n_frames_after_done

        assert type(p1) == type(p2) == dict
        assert set(p1.keys()) == set(p2.keys()) == {'policy', 'frameskip', 'env'}
        self.p1 = p1
        self.p2 = p2
        self.p1_actions = self.p2_actions = [0] * 12

        env.reset()
        self.sound_rate = env.em.get_audio_rate()
        self.sound_buffer = np.array([[0, 0]], dtype=np.dtype('int16'))
        self.sound_buffer_len = 12

        if render:
            pygame.init()
            pygame.display.set_caption("Interactive Env")
            self.win = pygame.display.set_mode(win_size, 0, 32)
            self.clock = pygame.time.Clock()

        self.record_output_path = record_output_path
        self.images = []
        self.sounds = np.array([[0, 0]])

        self.step_count = 0

    def run(self):
        done = False

        obs = self.env.reset()
        sound = self.env.em.get_audio()
        actions = self._get_actions(obs)
        im = self._get_full_image(obs, {}, actions)
        self._render_frame_and_sound(im, sound)

        while not done or self.render_n_frames_after_done > 0:
            self.step_count += 1

            obs, rew, done, info = self.env.step(actions)
            sound = self.env.em.get_audio()
            actions = self._get_actions(obs)
            im = self._get_full_image(obs, info, actions)
            self._render_frame_and_sound(im, sound)

            if done:
                self.render_n_frames_after_done -= 1

        self.on_close()

    def _render_frame_and_sound(self, frame, sound):
        self.images.append(frame)
        self.sounds = np.append(self.sounds, sound[:, 0])

        if self.render:
            # image
            frame = np.transpose(frame, axes=(1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            self.win.blit(surf, (0, 0))
            pygame.display.update()

            # sound
            if self.step_count % self.sound_buffer_len == 0:
                pygame.mixer.Sound(array=self.sound_buffer).play()
                self.sound_buffer = np.array(sound, dtype=np.dtype('int16'))
            else:
                self.sound_buffer = np.append(self.sound_buffer, sound[::2, :], axis=0)

            self.clock.tick(self.fps)

    def _get_keyboard_actions(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        def get_buttons_vec(d):
            return [keys[ord(d[k].lower())]for k in self.buttons]

        return get_buttons_vec(self.inputs) + [0] * 12

    def _get_actions(self, obs):
        keyboard_actions = self._get_keyboard_actions()

        if self.step_count % self.p1["frameskip"] == 0:
            self.p1_obs = self.p1["env"].observation(obs)

        if self.p1["policy"] != "human" and self.step_count % self.p1["frameskip"] == 0:
            self.p1_actions = self.p1["policy"].predict(self.p1_obs)[0]
        keyboard_actions[:12] = self.p1_actions

        if self.step_count % self.p2["frameskip"] == 0:
            self.p2_obs = self.p2["env"].observation(obs)

        if self.p2["policy"] != "human" and self.step_count % self.p2["frameskip"] == 0:
            self.p2_actions = self.p2["policy"].predict(self.p2_obs)[0]
        keyboard_actions[12:] = self.p2_actions

        return keyboard_actions

    def _get_full_image(self, obs, info, act):
        main_img = Image.fromarray(obs)
        p1_img = self._get_player_img(self.p1_obs)
        p2_img = self._get_player_img(self.p2_obs)

        img = self._get_concat_h(p1_img, main_img)
        img = self._get_concat_h(img, p2_img)

        action_info_img = self._get_action_info_img(act)
        info_img = self._get_info_img(info)
        info_img = self._get_concat_h(action_info_img, info_img)
        img = self._get_concat_v(img, info_img)

        return cv2.resize(np.array(img), self.win_size)

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
              f'P1:{vec_to_str([act[self.buttons.index(k)] for k in buttons])}\n' \
              f'P2:{vec_to_str([act[self.buttons.index(k) + 12] for k in buttons])}'
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

    def on_close(self):
        if self.record_output_path:
            # Ugly as hell but I don't care, it works :>
            audio_rate = self.env.em.get_audio_rate()
            shape = self.images[0].shape[:2]
            shape = shape[::-1]
            with tempfile.TemporaryDirectory(dir="/tmp") as temp:
                writer = cv2.VideoWriter(temp + '/tmp_vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                         self.fps, frameSize=shape)

                for img in self.images:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
                    writer.write(img)

                sf.write(temp + '/tmp_sound.flac', self.sounds / (max(self.sounds) * 2), int(audio_rate))

                cv2.destroyAllWindows()
                writer.release()

                my_clip = mpe.VideoFileClip(temp + '/tmp_vid.mp4')
                audio_background = mpe.AudioFileClip(temp + '/tmp_sound.flac')
                final_clip = my_clip.set_audio(audio_background)
                final_clip.write_videofile(self.record_output_path, logger=None)
