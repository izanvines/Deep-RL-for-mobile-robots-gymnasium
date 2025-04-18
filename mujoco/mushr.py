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
        self.last_action = np.zeros(2, dtype=np.float64)  # Inicializa acción previa
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "one_car.xml", 5, observation_space=observation_space, **kwargs
        )     

    def step(self, a):
        vec = self.get_body_com("buddy") - self.get_body_com("target")
        distance_to_target = np.linalg.norm(vec)

        # --- CALCULO DEL HEADING AL OBJETIVO ---
        quat = self.sim.data.qpos[3:7]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w] para scipy
        yaw = r.as_euler("xyz", degrees=True)[2]

        buddy_pos = self.get_body_com("buddy")[:2]
        target_pos = self.get_body_com("target")[:2]
        vec_to_target = target_pos - buddy_pos
        target_angle = np.degrees(np.arctan2(vec_to_target[1], vec_to_target[0]))


        # ===== Normalización y pesos =====
        weight_dist = 1.0
        weight_ctrl = 0.2
        weight_time = 1
        weight_heading = 1.0
        reward_goal = 100.0

        #--------------------------------- CALCULO DE LA RECOMPENSA --------------------------------- 
        
        # --- Recompensa basada en distancia ---
        reward_dist = weight_dist * np.exp(-distance_to_target)
        
        # --- Penalización por tiempo ---
        reward_time_penalty = -weight_time

        # --- Penalización por acción ---
        reward_ctrl = weight_ctrl * -np.square(a).sum()

        # --- Recompensa por heading ---
        heading_error = (target_angle - yaw + 180) % 360 - 180
        reward_heading = weight_heading * (1- abs(heading_error)/180.0)

        # === Recompensa final ===
        reward = reward_dist + reward_ctrl + reward_time_penalty + reward_heading
      
        # --- Recompensa si alcanza el objetivo ---
        min_distance = 0.25
        terminated = False
        if distance_to_target < min_distance:
            reward += reward_goal
            terminated = True
       
       
        #---------------------------------------------------------------------------------------------
        #=== ACCIÓN SOBRE EL ENTORNO ===
        self.do_simulation(a, self.frame_skip) 

        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        self.last_action = a  # Guarda acción actual como la "última"

        return (
            ob,
            reward,
            terminated,
            False,
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

        return self._get_obs()

    def _get_obs(self):
        # Posición relativa entre el rover y el objetivo (3 xyz)
        rel_pos = self.get_body_com("buddy") - self.get_body_com("target")  
        
        # Distancia al objetivo (1 dist)
        dist_to_target = [np.linalg.norm(self.get_body_com("buddy") - self.get_body_com("target"))]
        
        # Posición absoluta del rover (3 xyz)
        pos_buddy = self.get_body_com("buddy")  
        
        # Orientación del rover (1 angle)
        orientation_buddy = [np.arctan2(self.get_body_com("buddy")[1], self.get_body_com("buddy")[0])]
        
        # Posición del objetivo (3 xyz)
        pos_target = self.get_body_com("target")
        
        return np.concatenate(
            [
                rel_pos,           # Vector al objetivo (3)
                dist_to_target,    # Distancia al objetivo (1)
                pos_buddy,         # Posición del rover (3)
                orientation_buddy, # Orientación del rover (1)
                pos_target,        # Posición del objetivo (3)
                self.last_action   # Acción previa (2)
            ]
        )

