from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import pygame
import gymnasium
from stable_baselines3.common.env_util import make_vec_env
import torch

# --- SEMILLA ---
SEED = 42  # o el número que prefieras
np.random.seed(SEED)
torch.manual_seed(SEED)

pygame.init()
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("Control MuSHR - Teclado")

# === CREAR Y ENVOLVER ENTORNO ===
def make_env():
    #return gymnasium.make("MuSHREnv-v0")
    return gymnasium.make("MuSHREnv-v0", render_mode="human") #-> con entorno gráfico

env = make_vec_env(make_env, n_envs=1, seed=SEED) #Por defeecto usa gymnasium -> ya se ha añadido el entorno a gymnasium para poder usarlo (menos warnings)
obs = env.reset()

steering = 0.0
speed = 0.0
step_size = 0.05

last_print_time = time.time()
print_interval = 2.0  # segundos

print("Controles: ← → para girar | ↑ ↓ para acelerar/frenar | Cierra la ventana para salir")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        steering -= step_size
    if keys[pygame.K_RIGHT]:
        steering += step_size
    if keys[pygame.K_UP]:
        speed += step_size
    if keys[pygame.K_DOWN]:
        speed -= step_size

    steering = np.clip(steering, -0.38, 0.38)
    speed = np.clip(speed, -0.3, 0.3)

    action = np.array([[steering, speed]])
    obs, total_reward, terminated, info = env.step(action)

    # === MOSTRAR ÁNGULO CADA 2 SEGUNDOS ===
    current_time = time.time()
    if current_time - last_print_time > print_interval:
        qpos = env.envs[0].unwrapped.sim.data.qpos
        quat = qpos[3:7]  # cuaternión [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy usa [x, y, z, w]
        euler = r.as_euler("xyz", degrees=True)
        yaw = euler[2]

        # === Ángulo hacia el objetivo ===
        buddy_pos = env.envs[0].unwrapped.get_body_com("buddy")[:2]
        target_pos = env.envs[0].unwrapped.get_body_com("target")[:2]
        vec_to_target = target_pos - buddy_pos
        target_angle = np.degrees(np.arctan2(vec_to_target[1], vec_to_target[0]))

        print(f"Ángulo del robot (heading/yaw): {yaw:.2f}° | Objetivo en dirección: {target_angle:.2f}°")
        last_print_time = current_time
        
        # Mostrar por terminal cada una de las recompensas


    if terminated:
        print("Episodio terminado")
        obs = env.reset()

    time.sleep(0.05)

env.close()
pygame.quit()