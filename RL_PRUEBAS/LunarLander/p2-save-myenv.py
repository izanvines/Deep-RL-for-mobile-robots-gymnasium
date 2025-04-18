import gym
from stable_baselines3 import PPO, A2C
import os

models_dir = "models_Lunar_1012_3/A2C"
logdir = "logs_Lunar_1012"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

#env = gym.make("Rover")
env = gym.make("LunarLander-v2", render_mode="human")
env.reset()


model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 5000
for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C_Lunar_1012_3")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()