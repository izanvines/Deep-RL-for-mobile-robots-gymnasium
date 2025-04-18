import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import torch

# --- SEMILLA ---
SEED = 42  # o el número que prefieras
np.random.seed(SEED)
torch.manual_seed(SEED)

# === CONFIGURACIÓN ===
models_dir = "/media/izan/usb-rl/MuSHR/models_AMR_150425_2/PPO"
model_path = f"{models_dir}/3505000"  # Usa el modelo que quieras
normalize_path = "/media/izan/usb-rl/MuSHR/models_AMR_150425_2/vec_normalize.pkl"

# === CREAR Y ENVOLVER ENTORNO ===
def make_env():
    #return gymnasium.make("MuSHREnv-v0")
    return gymnasium.make("MuSHREnv-v0", render_mode="human") #-> con entorno gráfico

env = make_vec_env(make_env, n_envs=1, seed=SEED) #Por defeecto usa gymnasium -> ya se ha añadido el entorno a gymnasium para poder usarlo (menos warnings)

# IMPORTANTE: desactivar el update de stats para reproducir correctamente
env.training = False
env.norm_reward = False

model = PPO.load(model_path, env=env)

episodes = 100

for ep in range(episodes):
    # Resetear el entorno y obtener la observación
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)  # El modelo ahora toma la observación (obs)
        obs, reward, done, info = env.step(action)

env.close()
