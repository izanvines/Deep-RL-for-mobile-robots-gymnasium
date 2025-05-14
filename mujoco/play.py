from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import pygame
import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch

# --- SEMILLA ---
SEED = 42  
np.random.seed(SEED)
torch.manual_seed(SEED)

pygame.init()
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("Control MuSHR - Teclado")

# === CREAR Y ENVOLVER ENTORNO ===
def make_env():
    return gymnasium.make("MuSHREnv-v0", render_mode="human") 

env = make_vec_env(make_env, n_envs=1, seed=SEED) 

vec_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=5.0) # Normalización
obs = vec_env.reset()

steering = 0.0
speed = 0.0
step_size = 0.05

last_print_time = time.time()
print_interval = 2.0 

cumulative_reward = 0.0

print("Controles: ← → para girar | ↑ ↓ para acelerar/frenar | Cierra la ventana para salir")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        steering += step_size
    if keys[pygame.K_RIGHT]:
        steering -= step_size
    if keys[pygame.K_UP]:
        speed += step_size
    if keys[pygame.K_DOWN]:
        speed -= step_size

    steering = np.clip(steering, -0.38, 0.38)
    speed = np.clip(speed, -0.3, 0.3)

    action = np.array([[steering, speed]])
    obs, total_reward, terminated, info = env.step(action)

    cumulative_reward += total_reward[0]

    # === MOSTRAR ÁNGULO CADA 2 SEGUNDOS ===
    current_time = time.time()
    if current_time - last_print_time > print_interval:
        
        last_print_time = current_time
        
        print("===================== RECOMPENSA STEP =========================")
        reward_dist = info[0].get("reward_dist", 0.0)
        reward_ctrl = info[0].get("reward_ctrl", 0.0)
        reward_time = info[0].get("reward_time", 0.0)
        reward_heading = info[0].get("reward_heading", 0.0)

        print(f"Recompensas parciales:")
        print(f"  Distancia     : {reward_dist:.3f}")
        print(f"  Control       : {reward_ctrl:.3f}")
        print(f"  Temporal      : {reward_time:.3f}")
        print(f"  Heading       : {reward_heading:.3f}")
        print(f"  Recompensa paso actual: {total_reward[0]:.3f}")
        print(f"Recompensas acumulada en el epiosdio : {cumulative_reward:.3f}")
        print("===============================================================")

    if terminated:
        print("Episodio terminado")
        obs = env.reset()
        cumulative_reward = 0.0  # Reiniciar recompensa acumulada

    time.sleep(0.05)

env.close()
pygame.quit()