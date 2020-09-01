import retro
import time
import numpy as np
from PIL import Image
import cv2
import soundfile as sf
import tempfile
import moviepy.editor as mpe


def render_video(env, step_function, output, max_length):
    obs = env.reset()
    sound = np.array([])
    frames = []

    for _ in range(max_length):
        obs, rew, done, info = env.step(step_function(obs))
        sound = np.append(sound, env.em.get_audio()[:, 0])
        frames.append(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        if done:
            obs = env.reset()
            break

    rate = env.em.get_audio_rate()
    env.close()

    with tempfile.TemporaryDirectory(dir="/tmp") as temp:
        fps = len(frames) / (len(sound) / rate)
        writer = cv2.VideoWriter(temp + '/tmp_vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                                 fps, (obs.shape[1], obs.shape[0]))

        for frame in frames:
            writer.write(frame)

        sf.write(temp + '/tmp_sound.flac', sound / (max(sound) * 2), int(rate))

        cv2.destroyAllWindows()
        writer.release()

        my_clip = mpe.VideoFileClip(temp + '/tmp_vid.mp4')
        audio_background = mpe.AudioFileClip(temp + '/tmp_sound.flac')
        final_clip = my_clip.set_audio(audio_background)
        final_clip.write_videofile(output, logger=None)

