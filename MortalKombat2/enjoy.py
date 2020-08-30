from MortalKombat2.Env import make_mk2
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.atari_wrappers import WarpFrame
import time
from MortalKombat2.Env import FrameskipWithRealGameTracker, MaxEpLenWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env():
    env = make_mk2(state="Scorpion_vs_Raiden_1p", players=1)
    env = WarpFrame(env, 48, 48)
    env = FrameskipWithRealGameTracker(env, skip=10)
    return Monitor(env)


env = make_env()
model = PPO.load("saves/mk2_raiden_easy_scorpio/rl_model_7359985_steps.zip", env)

obs = env.reset()
done = False
while not done:
    a = model.predict(obs)
    obs, rew, done, info = env.step(a[0])
    env.render()
    time.sleep(1/ 6)

print(info)