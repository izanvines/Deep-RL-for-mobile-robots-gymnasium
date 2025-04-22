import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv
from gymnasium.spaces import Box

from scipy.spatial.transform import Rotation as R


class MuSHREnv(MuJocoPyEnv, utils.EzPickle):
        
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20, #No entiendo porque solo funciona con 20
    }

    
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        # Inicializión observaciones
        self.last_action = np.zeros(2, dtype=np.float64)  
        self.last_yaw = 0.0
        self.last_heading_error = 0.0

        self.cumulative_reward = 0.0  # Inicializar antes del bucle while True

        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

        MuJocoPyEnv.__init__(
            self, "one_car.xml", 5, observation_space=observation_space, **kwargs
        )     

    def step(self, a):

        terminated = False
        truncated = False

        vec_to_target = (self.get_body_com("target") - self.get_body_com("buddy"))[:2] # XY
        distance_to_target = np.linalg.norm(vec_to_target)

        # --- CALCULO DEL HEADING AL OBJETIVO ---
        #[x, y, z, w, x, y, z] (posición + orientación)
        quat = self.sim.data.qpos[3:7]
        yaw = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz", degrees=False)[2]

        target_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
        
        # ===== Normalización y pesos =====
        weight_dist = 5.0
        weight_ctrl = 1.0
        weight_time = 0.5
        weight_heading = 2.0
        reward_goal = 2000.0

        #--------------------------------- CALCULO DE LA RECOMPENSA --------------------------------- 
        
        # --- Recompensa basada en distancia ---
        reward_dist = weight_dist * np.exp(-0.3*distance_to_target)
        
        # --- Penalización por tiempo ---
        reward_time_penalty = -weight_time

        # --- Penalización por acción ---
        delta_steering = a[0] - self.last_action[0]
        reward_ctrl = weight_ctrl * -(delta_steering ** 2)

        # --- Recompensa por heading ---
        heading_error = (target_angle - yaw + np.pi) % (2 * np.pi) - np.pi #error angular entre pi y -pi NORMALIZADO
        reward_heading = weight_heading * (1 - 2 * abs(heading_error) / np.pi) #Error entre -1 y 1

        # === Recompensa final ===
        reward = reward_dist + reward_ctrl + reward_time_penalty + reward_heading
      
        # --- Recompensa si alcanza el objetivo ---
        min_distance = 0.25
        
        if distance_to_target < min_distance:
            reward += reward_goal
            terminated = True
       
       
        #---------------------------------------------------------------------------------------------
        #=== ACCIÓN SOBRE EL ENTORNO ===
        self.do_simulation(a, self.frame_skip) 

        if self.render_mode == "human":
            self.render()

        self.last_yaw = yaw
        self.last_heading_error = heading_error

        ob = self._get_obs()

        self.last_action = a 

        self.cumulative_reward += reward

        
        """
        print("===================== RECOMPENSA STEP =========================")
        print(f"Recompensas parciales:")
        print(f"  Distancia     : {reward_dist:.3f}")
        print(f"  Control       : {reward_ctrl:.3f}")
        print(f"  Temporal      : {reward_time_penalty:.3f}")
        print(f"  Heading       : {reward_heading:.3f}")
        print(f"  Recompensa paso actual: {reward:.3f}")
        print(f"Recompensas acumulada en el epiosdio : {self.cumulative_reward:.3f}")
        print("===============================================================")
        """
        

        return (
            ob,
            reward,
            terminated,
            truncated,
            dict(
                reward_dist=reward_dist,
                reward_ctrl=reward_ctrl,
                reward_time=reward_time_penalty,
                reward_heading=reward_heading,
            ),
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0
        
    def reset_model(self): 
       
        qpos = self.init_qpos

        # Generar una posición aleatoria dentro del rango de distancia especificado
        min_distance = 1.5
        max_distance = 4.0

        while True:
            # Generar una posición aleatoria dentro del cuadrado que engloba el círculo
            random_pos = self.np_random.uniform(low=-max_distance, high=max_distance, size=2)
            distance = np.linalg.norm(random_pos)
            
            # Aceptar solo posiciones dentro del anillo (entre los radios mínimo y máximo)
            if min_distance <= distance <= max_distance:
                break

        self.goal = random_pos

        qpos[-2:] = self.goal
               
        #Todas las velocidades a 0, partimos del reposo y el objetivo no se mueve
        qvel = self.init_qvel 

        self.set_state(qpos, qvel)

        self.last_action = np.zeros(2, dtype=np.float64)
        self.last_yaw = 0.0
        self.last_heading_error = 0.0

        self.cumulative_reward = 0.0  # Reiniciar recompensa acumulada

        return self._get_obs()

    def _get_obs(self):
        # Posición relativa entre el rover y el objetivo (2 -> XY)
        rel_pos = (self.get_body_com("buddy") - self.get_body_com("target"))[:2]

        # Posición absoluta del rover (2 -> XY)
        pos_buddy = (self.get_body_com("buddy"))[:2]  

        # Posición del objetivo (2 -> XY)
        pos_target = (self.get_body_com("target"))[:2]

        return np.concatenate(
            [
                rel_pos,                 # Vector al objetivo - error_pos (2 -> XY)
                [self.last_heading_error], # Error_orienta (1)
                pos_buddy,               # Posición del rover (2 -> XY)
                [self.last_yaw],           # Orientación del rover (1 -> YAW)
                pos_target,              # Posición del objetivo (2 -> XY)
                self.last_action         # Acción previa (2)
            ]
        )

