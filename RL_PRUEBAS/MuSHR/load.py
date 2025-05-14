import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime

# --- SEMILLA ---
SEED = 42  
np.random.seed(SEED)
torch.manual_seed(SEED)

# === CONFIGURACIÓN ===
models_dir = "/home/izan/Escritorio/graficas"
model_path = f"{models_dir}/models_AMR_200425_3/PPO/5300000"
normalize_path = f"{models_dir}/models_AMR_200425_3/vec_normalize.pkl"

# === CREAR Y ENVOLVER ENTORNO ===
def make_env():
    return gymnasium.make("MuSHREnv-v0", render_mode="human")

env = make_vec_env(make_env, n_envs=1, seed=SEED)
env = VecNormalize.load(normalize_path, env)

env.training = False
env.norm_reward = False

model = PPO.load(model_path, env=env)

episodes = 10
plt.figure()

for ep in range(episodes):
    obs = env.reset()
    done = False
    episode_positions = []

    raw_obs = env.get_original_obs()
    target_x = raw_obs[0][6]
    target_y = raw_obs[0][7]

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        raw_obs = env.get_original_obs()
        x = raw_obs[0][3]
        y = raw_obs[0][4]
        episode_positions.append((x, y))

    # Graficar
    xs, ys = zip(*episode_positions)
    plt.scatter(xs, ys, s=5, label=f"Episodio {ep+1}")
    plt.plot(target_x, target_y, 'ro', markersize=15)

plt.title("Trayectorias trazadas por el robot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis("equal")
plt.legend(loc='best', fontsize='small')

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"/home/izan/Escritorio/trayectorias_objetivos_{timestamp}.png"
plt.savefig(filename)
plt.close()

env.close()
