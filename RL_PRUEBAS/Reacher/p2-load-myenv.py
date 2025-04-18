import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Crear el entorno y envolverlo en un DummyVecEnv
env = gym.make("Reacher", render_mode="human")
env = DummyVecEnv([lambda: env])  # Envolver el entorno en un VecEnv

# Cargar el modelo
models_dir = "models_Reacher_140125_1/PPO"
model_path = f"{models_dir}/200000"

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
