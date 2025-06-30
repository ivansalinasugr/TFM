#!/usr/bin/python3
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import numpy as np
import math
from math import atan2
from cv_bridge import CvBridge
import cv2
import torch
import argparse
import time
import os
import random
from stable_baselines3.common.callbacks import BaseCallback
import signal
import sys
from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
import wandb
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from gazebo_msgs.msg import EntityState, ModelStates
import matplotlib
import imageio
from typing import Callable
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# for debug
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

tamanio = 9

# Convertir ángulo a cuaternión
def quaternion_from_yaw(yaw):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return (0.0, 0.0, qz, qw)

# Convertir el mapa de ocupación en imagen (para visualización)
def occupancy_map_to_image(occupancy_map, escala=44.4444, colormap_name='bwr'):
    # Crear imagen RGB directamente con valores en [-1, 1]
    cmap = matplotlib.colormaps[colormap_name]
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    rgba = cmap(norm(occupancy_map))  # incluye alpha
    rgb_image = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Escalar imagen
    rgb_image_scaled = cv2.resize(rgb_image, (800, 800), interpolation=cv2.INTER_NEAREST)

    # Dibujar texto solo donde el valor sea distinto de 0
    nonzero_indices = np.argwhere(occupancy_map != 0)

    for i, j in nonzero_indices:
        texto = f"{occupancy_map[i, j]:.2f}"
        pos = (round(j * escala + 4), round(i * escala + escala - 15))
        cv2.putText(rgb_image_scaled, texto, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return rgb_image_scaled

# Convertir el mapa del lidar en imagen (para visualización)
def lidar_map_to_image(lidar_map):
    gray = (((lidar_map + 1) / 2) * 255).astype(np.uint8)
    rgb_image = np.stack([gray] * 3, axis=-1)
    return cv2.resize(rgb_image, (800, 800), interpolation=cv2.INTER_NEAREST)

# Callback personalizado para registrar métricas en WandB y guardar videos.
class WandbCallbackWithVideo(BaseCallback):
    def __init__(self, env, save_path, model_save_path, save_freq_episodes = 2, verbose=1):
        super(WandbCallbackWithVideo, self).__init__(verbose)
        self.env = env
        self.save_path = save_path
        self.model_save_path = model_save_path
        self.save_freq_episodes = save_freq_episodes
        self.episode_rewards = []
        self.current_episode_rewards = []
        self.video_frames = []
        self.actions = []
        self.time = 0
        self.num_episodios_metricas = 0
        self.current_info = []
        self.metrics = {"tasa_exito": 0, "tasa_colision": 0, "tasa_timeout": 0, "distancia_media_minima_personas": 0}

        # Para visualizar entorno, mapa de ocupación y lidar
        sep = 32
        self.h_sep = np.full((sep, 800 + sep + 800, 3), 255, dtype=np.uint8)  # blanco entre ocupación y lidar
        self.v_sep = np.full((800, sep, 3), 255, dtype=np.uint8)  # gris vertical

        # Crear directorio de videos si no existe
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        action = [self.env.linear_velocity, self.env.angular_velocity]
        self.current_episode_rewards.append(reward)
        self.actions.append(action)
        self.current_info.append(self.locals["infos"][0])

        # Renderizar el entorno y agregar el frame redimensionado al video
        frame, occup_map_vx, occup_map_vy, lidar_history = self.env.render(mode='rgb_array')

        combined_top = np.hstack((frame, self.v_sep, lidar_map_to_image(lidar_history)))
        combined_bottom = np.hstack((occupancy_map_to_image(occup_map_vx), self.v_sep, occupancy_map_to_image(occup_map_vy)))
        combined_frame = np.vstack((combined_top, self.h_sep, combined_bottom))

        # Guardar imagen en buffer
        self.video_frames.append(combined_frame)

        self.time += 1

        # Verificar si el episodio ha terminado
        dones = self.locals["dones"]
        if dones[0]:  # Si el primer agente terminó
            self._on_episode_end()

        return True

    def _on_episode_end(self) -> None:
        # Registrar recompensa total del episodio
        episode_reward = np.sum(self.current_episode_rewards)
        self.episode_rewards.append(episode_reward)

        # Guardar video del episodio
        video_path = os.path.join(self.save_path, f"episode_{len(self.episode_rewards)}.mp4")
        self._save_video(self.video_frames, video_path)

        # Obtener métricas de entrenamiento del modelo
        training_metrics = {}
        if self.model._n_updates > 0:
            training_metrics = self.model.logger.name_to_value.copy()  # Copiar los logs actuales

        # Extraer todas las velocidades lineales y angulares de self.actions
        linear_actions = [action[0] for action in self.actions]  
        angular_actions = [action[1] for action in self.actions]  

        if self.num_timesteps > 2000: # mayor que el valor de learning starts del modelo
            self.num_episodios_metricas += 1
            for info in self.current_info:
                if info["success"]:
                    self.metrics["tasa_exito"] += 1
                    break
                elif info["colision"]:
                    self.metrics["tasa_colision"] += 1
                    break
                elif info["timeout"]:
                    self.metrics["tasa_timeout"] += 1
                    break

            self.metrics["distancia_media_minima_personas"] += min(d["min_dist_human"] for d in self.current_info)

            metrics_processed = {}
            for k, v in self.metrics.items():
                if k == ("distancia_media_minima_personas"):
                    metrics_processed[k] = v / self.num_episodios_metricas
                else:
                    metrics_processed[k] = (v / self.num_episodios_metricas) * 100

        # Loggear en WandB
        log_data = {
            "episode_reward": episode_reward,
            "rewards_mean": np.mean(self.episode_rewards),
            "total_timesteps": self.num_timesteps,
            "time": self.time,
            "linear_hist": wandb.Histogram(linear_actions),
            "angular_hist": wandb.Histogram(angular_actions),
            "video": wandb.Video(video_path, format="mp4"),
        }

        if self.num_timesteps > 2000: # mayor que el valor de learning starts del modelo
            log_data.update(metrics_processed)
            log_data.update(training_metrics)

        wandb.log(log_data)

        # Guardar el modelo cada N episodios
        if len(self.episode_rewards) % self.save_freq_episodes == 0:
            self.model.save(self.model_save_path)
            base_path, _ = os.path.splitext(self.model_save_path)
            replay_path = base_path + "-replay_buffer.pkl"
            self.model.save_replay_buffer(replay_path)
            if self.verbose:
                print(f"Modelo guardado después de {len(self.episode_rewards)} episodios en {model_path}")
                print(f"Replay buffer guardado en {replay_path}")

        # Reiniciar buffers
        self.current_episode_rewards = []
        self.video_frames = []
        self.actions = []  # Resetear acciones
        self.current_info = []
        self.time = 0

    def _save_video(self, frames, path):
        if len(frames) == 0:
            return  # Evitar guardar un video vacío

        # Asegurarse de que los frames estén en formato uint8 y sin alpha
        frames_uint8 = [frame.astype(np.uint8) for frame in frames]
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_uint8]
        imageio.mimsave(path, frames_rgb, fps=30)

# Entorno de Gazebo + Gymnasium
class GazeboEnv(gym.Env):

    def __init__(self, seed=None):
        self._init_ros2()
        self._initialize_robot_properties()
        self._define_action_and_observation_spaces()
        self._initialize_publishers()
        self._initialize_suscribers()

        self.seed(seed)

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
    
    # Inicializar la semilla
    def seed(self, seed=None):
        self._seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]


    # Continuar física del simulador
    def _unpause_physics(self, time_sleep=0.2):
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/unpause: service not available, waiting again')

        future = self.unpause.call_async(Empty.Request())

        # Esperar a que se complete
        while not future.done():
            time.sleep(0.01)

        if future.result() is None:
            self.node.get_logger().error("Error al hacer unpause de la física.")
            raise RuntimeError("/unpause service call failed.")

        time.sleep(time_sleep)

    # Pausar física del simulador
    def _pause_physics(self, time_sleep=0.2):
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/pause: service not available, waiting again')

        future = self.pause.call_async(Empty.Request())

        # Esperar a que se complete
        while not future.done():
            time.sleep(0.01)

        if future.result() is None:
            self.node.get_logger().error("Error al pausar la física.")
            raise RuntimeError("/pause service call failed.")

        time.sleep(time_sleep)

    # Iniciar parámetros de ROS2
    def _init_ros2(self):
        # Iniciar ROS 2 en hilo separado
        rclpy.init()
        self.node = Node("gazebo_gym_env")
        self.bridge = CvBridge()

        # Sincronización
        self.odom_data = None
        self.lidar_data = None
        self.image_data = None
        self.actors_data = {}
        self.odom_lock = threading.Lock()
        self.lidar_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.model_lock = threading.Lock()

    # Definir espacio de acciones y observaciones
    def _define_action_and_observation_spaces(self):
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "goal_distance": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # Distancia normalizada
            "goal_angle": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # Ángulo relativo
            "time": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # Tiempo
            "map": spaces.Box(low=-1.0, high=1.0, shape=(3, 18, 18), dtype=np.float32), # Mapa de ocupación
            "last_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })

    # Inicializar propiedades del robot y parámetros de recompensa
    def _initialize_robot_properties(self):
        gym.Env.__init__(self)

        # Robot velocity and angular speed limits
        self.max_linear_speed = 0.26  # Maximum linear velocity (m/s)
        self.max_angular_speed = 0.5  # Maximum angular velocity (rad/s)
        self.min_linear_speed = 0.0  # Maximum linear velocity (m/s)
        self.robot_radius = 0.15
        self.goal_radius = 0.3
        self.num_actors = 5

        self.unpause = self.node.create_client(Empty, "/unpause_physics")
        self.pause = self.node.create_client(Empty, "/pause_physics")

        self.config = {
            "MAX_STEPS": 750,
            "CLOSENESS_REWARD": 1.0,  # Recompensa por acercarse al objetivo
            "TARGET_REWARD": 10.0,    # Recompensa grande por alcanzar el objetivo
            "TIMEOUT_PENALTY": -10.0, # Penalización por no llegar al objetivo en el tiempo previsto
            "COLLISION_PENALTY": -10.0,  # Penalización fuerte por chocar
            "LIDAR_PENALTY": -0.1, # Penalización por acercarse a obstáculos

            # Nuevos parámetros para comportamiento social
            "HUMAN_PROXIMITY_PENALTY": -0.1/self.num_actors, # Penalización por invadir espacio personal (por metro de invasión)
            "PERSONAL_SPACE_RADIUS": 0.8,    # Metros, radio del espacio personal del humano
            "SOCIAL_ZONE_RADIUS": 2.0,       # Metros, zona donde el robot debe ser más cauteloso
            "MAX_SPEED_NEAR_HUMANS": 0.15,   # m/s, velocidad lineal máxima permitida cerca de humanos
            "SPEEDING_NEAR_HUMANS_PENALTY": -0.5/self.num_actors, # Penalización por exceder MAX_SPEED_NEAR_HUMANS
            "SMOOTHNESS_PENALTY_FACTOR": -0.01 # Factor de penalización por movimientos bruscos
        }

    # Obtener la configuración de recompensas
    def _get_reward_params(self):
        return self.config

    # Inicializar publicadores de ROS
    def _initialize_publishers(self):
        self.publisher_cmd_vel = self.node.create_publisher(Twist, '/cmd_vel', 10)

    # Inicializar subcriptores de ROS
    def _initialize_suscribers(self):
        self.node.create_subscription(Odometry, "/odom", self._odom_callback, 1)
        self.node.create_subscription(LaserScan, "/scan", self._lidar_callback, 1)
        self.node.create_subscription(Image, "/camera_sensor/image_raw", self._image_sensor_callback, 1)
        self.node.create_subscription(ModelStates, "/model_states", self._model_states_callback, 1)
    
    # Callback de odometría
    def _odom_callback(self, msg):
        with self.odom_lock:
            pose = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            q0, q1, q2, q3 = orientation.x, orientation.y, orientation.z, orientation.w
            self.odom_data = [pose.x, pose.y, -atan2(2 * (q0 * q1 + q2 * q3), (q0**2 - q1**2 - q2**2 + q3**2))]

    # Callback del lidar
    def _lidar_callback(self, msg):
        with self.lidar_lock:
            # Convierte los datos a array de float32
            lidar_full = np.array(msg.ranges, dtype=np.float32)

            # Selecciona los 180° frontales: desde -90° (270°) hasta +90° (90°)
            front_half = np.concatenate((
                lidar_full[270:],
                lidar_full[:90]
            ))

            self.lidar_data = front_half
    
    # Callback de la cámara aérea
    def _image_sensor_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        with self.image_lock:
            self.image_data = rgb_img

    # Callback de los humanos simulados
    def _model_states_callback(self, msg):
        with self.model_lock:
            for name, pose in zip(msg.name, msg.pose):
                if re.fullmatch(r"actor\d+", name):
                    actor_position = pose.position
                    actor_orientation = pose.orientation

                    q0, q1, q2, q3 = (
                        actor_orientation.x,
                        actor_orientation.y,
                        actor_orientation.z,
                        actor_orientation.w
                    )

                    yaw = -atan2(2 * (q0 * q1 + q2 * q3), (q0**2 - q1**2 - q2**2 + q3**2))
                    corrected_yaw = (yaw + np.pi/2 + np.pi) % (2*np.pi) - np.pi

                    # Actualiza los datos del actor en el diccionario
                    self.actors_data[name] = [actor_position.x, actor_position.y, corrected_yaw]

    # Esperar a que haya datos recibidos
    def _wait_for_data(self, timeout=5.0):
        with self.odom_lock:
            self.odom_data = None
        
        with self.lidar_lock:
            self.lidar_data = None
        
        with self.image_lock:
            self.image_data = None
        
        with self.model_lock:
            self.actors_data = {}
        
        self._unpause_physics(0.5)
        start_time = time.time()

        while (
            self.lidar_data is None or 
            self.odom_data is None or 
            self.image_data is None or
            len(self.actors_data) < self.num_actors
        ):
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout esperando datos de sensores.")
            
            missing = []
            if self.lidar_data is None: missing.append("lidar")
            if self.odom_data is None: missing.append("odom")
            if self.image_data is None: missing.append("image")
            if len(self.actors_data) < self.num_actors:
                missing.append(f"actors (esperados: {self.num_actors}, recibidos: {len(self.actors_data)})")

            self.node.get_logger().warn(f"Esperando: {', '.join(missing)}")
            time.sleep(0.1)
        
        self._pause_physics()

    # Normalizar orientación
    def _normalize_orientation(self, theta):
        return theta / np.pi
    
    # Normalizar rango del lidar
    def _normalize_lidar_ranges(self, lidar_ranges):
        # Definir los valores mínimo y máximo del sensor
        range_min = 0.12
        range_max = 3.5

        # Reemplazar valores infinitos por range_max
        ranges = np.clip(lidar_ranges, range_min, range_max)
        ranges_normalized = (2 * (ranges - range_min) / (range_max - range_min) - 1).astype(np.float32)

        return ranges_normalized

    # Normalizar el tiempo
    def _normalize_time(self):
        return 2 * (self.time / self.config["MAX_STEPS"]) - 1

    # Normalizar la distancia
    def _normalize_distance(self, distance):
        max_distance = 12.727
        distance = distance - self.goal_radius
        return 2 * (distance / max_distance) - 1
    
    # Calcular distancia euclidiana 2D entre dos poses
    def _calculate_distance_2d(self, pos1, pos2):
        """Calcula la distancia Euclidiana 2D entre dos poses (x,y,theta)."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    # Paso de simulación
    def step(self, action):
        current_linear_vel = self.min_linear_speed + (self.max_linear_speed - self.min_linear_speed) * (action[0] + 1) / 2
        current_angular_vel = self.max_angular_speed * action[1]

        self._update_robot_state(action)

        self._publish_cmd_vel()

        self._unpause_physics(0.1) # Duración de cada paso de simulación (se ejecuta la acción durante 0.1 seg)

        self._pause_physics()

        self.occupancy_map_vx = np.zeros((18, 18), dtype=np.float32)
        self.occupancy_map_vy = np.zeros((18, 18), dtype=np.float32)

        # Asignar personas al mapa de ocupación
        self._assign_to_local_velocity_maps(self.actors_data, self.odom_data)

        # Calcular la distancia al objetivo
        distance_to_goal_x, distance_to_goal_y = self.odom_data[0] - self.goal[0], self.odom_data[1] - self.goal[1]
        distance_to_the_goal = math.sqrt(distance_to_goal_x ** 2 + distance_to_goal_y ** 2)
        angulo_objetivo = self._calcular_angulo_hacia_objetivo(self.odom_data, self.goal)

        # Calcular recompensa
        reward = self._calcular_recompensa(distance_to_the_goal, current_linear_vel, current_angular_vel)

        self.prev_distance = distance_to_the_goal
        self.prev_linear_vel = current_linear_vel
        self.prev_angular_vel = current_angular_vel

        # Procesar datos del lidar
        self._process_lidar()

        self.time += 1
        
        map_obs = np.stack((self.occupancy_map_vx, self.occupancy_map_vy, self.lidar_history), axis=2)
        map_obs = map_obs.transpose(2, 0, 1)

        # Generar siguiente estado
        self.next_state = {
            "goal_distance": np.array([self._normalize_distance(distance_to_the_goal)], dtype=np.float32),
            "goal_angle": np.array([self._normalize_orientation(angulo_objetivo)], dtype=np.float32),
            "time": np.array([self._normalize_time()], dtype=np.float32),
            "map": map_obs,
            "last_velocity": np.array(action, dtype=np.float32)
        }

        return self.next_state, reward, self.done, False, self.info

    # Actualizar posición del robot
    def _update_robot_state(self, action):
        # Determinar las velocidades objetivo
        target_linear_velocity = self.min_linear_speed + (self.max_linear_speed - self.min_linear_speed) * (action[0] + 1) / 2
        target_angular_velocity = self.max_angular_speed * action[1]

        self.linear_velocity = target_linear_velocity
        self.angular_velocity = target_angular_velocity

    # Publicar comando de velocidad del robot (para que se mueva)
    def _publish_cmd_vel(self):
        twist_msg = Twist()
        
        twist_msg.linear.x = self.linear_velocity
        twist_msg.angular.z = self.angular_velocity

        self.publisher_cmd_vel.publish(twist_msg)

    # Calcular ángulo entre el robot y el objetivo
    def _calcular_angulo_hacia_objetivo(self, body_pose, objetivo_pos):
        # Extraer coordenadas
        x_robot, y_robot, orientacion_robot = body_pose
        x_objetivo, y_objetivo = objetivo_pos

        # Calcular el ángulo absoluto hacia el objetivo
        theta_objetivo = -math.atan2(y_objetivo - y_robot, x_objetivo - x_robot)

        # Diferencia de ángulos
        delta_theta = theta_objetivo - orientacion_robot

        # Normalizar a rango [-π, π]
        delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi

        return delta_theta

    # Calcular recompensa
    def _calcular_recompensa(self, distance_to_the_goal, current_linear_vel, current_angular_vel):
        reward_target = 0.0
        reward_obstacle = 0.0
        reward_social = 0.0
        reward_smoothness = 0.0

        # Recompensas y penalizaciones de navegación
        if distance_to_the_goal < self.goal_radius:
            reward_target = self.config["TARGET_REWARD"] + 5 * (self.config["MAX_STEPS"] - self.time) / self.config["MAX_STEPS"]
            self.done = True
            self.info["success"] = True
        elif self.time >= self.config["MAX_STEPS"]:
            reward_target = self.config["TIMEOUT_PENALTY"]
            self.done = True
            self.info["timeout"] = True
        elif min(self.lidar_data) <= 0.15:
            reward_target = self.config["COLLISION_PENALTY"]
            self.done = True
            self.info["colision"] = True
        else:
            reward_target = self.config["CLOSENESS_REWARD"] * (self.prev_distance - distance_to_the_goal)

            min_lidar_dist = min(self.lidar_data)
            if min_lidar_dist <= 3 * self.robot_radius:
                reward_obstacle = (3 * self.robot_radius - min_lidar_dist) * self.config["LIDAR_PENALTY"]

        # Penalizaciones sociales 
        if not self.done:
            robot_pos = self.odom_data

            for _, human_data in self.actors_data.items():
                human_pos = human_data

                dist_to_human = self._calculate_distance_2d(robot_pos, human_pos) - 0.25  # radio humano
                angulo_relativo = self._calcular_angulo_hacia_objetivo(robot_pos, human_pos[:2]) # para saber si está enfrente del robot

                if dist_to_human < self.info["min_dist_human"]:
                    self.info["min_dist_human"] = dist_to_human

                # 1. Penalización por invasión de espacio personal
                if dist_to_human < self.config["PERSONAL_SPACE_RADIUS"] and abs(angulo_relativo) <= math.pi / 2:
                    invasion_depth = self.config["PERSONAL_SPACE_RADIUS"] - dist_to_human
                    reward_social += invasion_depth * self.config["HUMAN_PROXIMITY_PENALTY"]

                # 2. Penalización por exceso de velocidad cerca de humanos
                if dist_to_human < self.config["SOCIAL_ZONE_RADIUS"] and abs(angulo_relativo) <= math.pi / 2:
                    if abs(current_linear_vel) > self.config["MAX_SPEED_NEAR_HUMANS"]:
                        reward_social += self.config["SPEEDING_NEAR_HUMANS_PENALTY"] * (
                            abs(current_linear_vel) - self.config["MAX_SPEED_NEAR_HUMANS"]
                        )

            # 3. Penalización por movimientos bruscos (jerk)
            linear_jerk = abs(current_linear_vel - self.prev_linear_vel)
            angular_jerk = abs(current_angular_vel - self.prev_angular_vel)
            reward_smoothness = (linear_jerk + 0.5 * angular_jerk) * self.config["SMOOTHNESS_PENALTY_FACTOR"]

        total_reward = reward_target + reward_obstacle + reward_social + reward_smoothness
        return total_reward

    # Devuelve true si no se solapan dos posiciones con una cierta distancia
    def _regions_overlap(self, pos1, pos2, min_dist=0.5):
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Calcular distancia euclidiana
        distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Verificar si están a más de min_dist metros
        return distancia > min_dist
    
    # Genera una bola roja pequeña en la posición del objetivo (para visual)
    def _spawn_objective_ball(self, x, y, z, time_sleep=0.2):
        client = self.node.create_client(SpawnEntity, '/spawn_entity')

        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Esperando servicio de spawn en Gazebo...')

        request = SpawnEntity.Request()
        request.name = "objective_ball"
        request.xml = open("/TFM/SDFs/objective.sdf", 'r').read()
        request.initial_pose.position.x = x
        request.initial_pose.position.y = y
        request.initial_pose.position.z = z

        future = client.call_async(request)

        while not future.done():
            time.sleep(0.01)

        if future.result() is not None:
            self.node.get_logger().info("Bola objetivo creada exitosamente en Gazebo")
        else:
            self.node.get_logger().error("Fallo al crear la bola objetivo")

        time.sleep(time_sleep)

    # Borra la bola roja pequeña
    def _delete_objective_ball(self, time_sleep=0.2):
        client = self.node.create_client(DeleteEntity, '/delete_entity')

        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Esperando servicio /delete_entity en Gazebo...')

        request = DeleteEntity.Request()
        request.name = "objective_ball"

        future = client.call_async(request)

        while not future.done():
            time.sleep(0.01)

        if future.result() is not None:
            self.node.get_logger().info("Bola objetivo eliminada de Gazebo.")
        else:
            self.node.get_logger().error("Fallo al eliminar la bola objetivo.")

        time.sleep(time_sleep)

    # Generar obstáculo fijo en forma de cubo
    def _spawn_obstacle(self, name, x, y, time_sleep=0.2):
        client = self.node.create_client(SpawnEntity, '/spawn_entity')

        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Esperando servicio de spawn en Gazebo...')

        request = SpawnEntity.Request()
        request.name = name
        request.xml = open("/TFM/SDFs/cube.sdf", 'r').read()
        request.initial_pose.position.x = x
        request.initial_pose.position.y = y
        request.initial_pose.position.z = 0.1

        future = client.call_async(request)

        while not future.done():
            time.sleep(0.01)

        if future.result() is not None:
            self.node.get_logger().info(f"Obstáculo {name} creado exitosamente en Gazebo")
        else:
            self.node.get_logger().error(f"Fallo al crear el obstáculo {name}")

        time.sleep(time_sleep)

    # Eliminar obstáculo fijo
    def _delete_obstacle(self, name, time_sleep=0.2):
        client = self.node.create_client(DeleteEntity, '/delete_entity')

        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Esperando servicio /delete_entity en Gazebo...')

        request = DeleteEntity.Request()
        request.name = name

        future = client.call_async(request)

        while not future.done():
            time.sleep(0.01)

        if future.result() is not None:
            self.node.get_logger().info(f"Obstáculo {name} eliminado de Gazebo.")
        else:
            self.node.get_logger().error(f"Fallo al eliminar el obstáculo {name}.")

        time.sleep(time_sleep)

    # Resetea la posición del robot a la pasada por parámetros
    def _reset_model_position(self, name, position=(0.0, 0.0, 0.0), orient = 0):
        set_entity_state_client = self.node.create_client(SetEntityState, '/set_entity_state')
        while not set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/set_entity_state: esperando al servicio...')

        state_msg = EntityState()
        state_msg.name = name
        state_msg.pose.position.x = position[0]
        state_msg.pose.position.y = position[1]
        state_msg.pose.position.z = position[2]
        state_msg.pose.orientation.x = 0.0
        state_msg.pose.orientation.y = 0.0
        state_msg.pose.orientation.z = math.sin(orient / 2.0)
        state_msg.pose.orientation.w = math.cos(orient / 2.0)

        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0

        state_msg.reference_frame = "world"

        future = set_entity_state_client.call_async(SetEntityState.Request(state=state_msg))

        while not future.done():
            time.sleep(0.01)

        if future.result() is not None and future.result().success:
            pass
        else:
            self.node.get_logger().error("Error al resetear posición de robot.")
            raise RuntimeError("/set_entity_state service call failed.")

    # Procesa los valores de lidar para transformarlos en el mapa del lidar
    def _process_lidar(self):
        # Normalizar LIDAR (asumido en [-1, 1])
        data = self._normalize_lidar_ranges(self.lidar_data)

        # Reorganizar en bloques de 10 (18 sectores)
        blocks = data.reshape(18, 10)

        # Calcular min y avg por bloque
        min_vals = np.min(blocks, axis=1)
        avg_vals = np.mean(blocks, axis=1)

        # Desplazar historial (observaciones pasadas)
        self.lidar_history[2:, :] = self.lidar_history[:-2, :]

        # Insertar observación actual
        self.lidar_history[0, :] = min_vals
        self.lidar_history[1, :] = avg_vals

    # Asigna las velocidades de cada humano en el mapa de ocupación
    def _assign_to_local_velocity_maps(self, actors_data, robot_pose):
        # Parámetros
        GRID_SIZE = 0.25
        MAP_SIZE = 18
        MAX_VEL = 0.3

        x_r, y_r, theta_r = robot_pose  # Posición del robot

        for actor_id, human_pose in actors_data.items():
            x_h, y_h = human_pose[0], human_pose[1]

            # Posición relativa al robot
            dx = x_h - x_r
            dy = y_h - y_r

            # Transformación al marco del robot
            rel_x = dx * np.cos(theta_r) - dy * np.sin(theta_r)
            rel_y = dx * np.sin(theta_r) + dy * np.cos(theta_r)

            # Índices en el mapa local
            ix = int(np.ceil(MAP_SIZE - 1 - (rel_x / GRID_SIZE)))
            iy = int(np.floor(-rel_y / GRID_SIZE) + (MAP_SIZE // 2))

            # Cálculo de velocidad
            if self.prev_human_pos[actor_id] is None:
                vx, vy = 0.0, 0.0
            else:
                prev_x, prev_y = self.prev_human_pos[actor_id]
                vx = (x_h - prev_x) / 0.1
                vy = (y_h - prev_y) / 0.1

            norm_vx = np.clip(vx / MAX_VEL, -1.0, 1.0)
            norm_vy = np.clip(vy / MAX_VEL, -1.0, 1.0)

            # Guardar nueva posición
            self.prev_human_pos[actor_id] = (x_h, y_h)

            # Guardar en el mapa si está dentro de los límites
            if 0 <= ix < MAP_SIZE and 0 <= iy < MAP_SIZE:
                if self.occupancy_map_vx[ix, iy] == 0.0:
                    self.occupancy_map_vx[ix, iy] = norm_vx

                if self.occupancy_map_vy[ix, iy] == 0.0:
                    self.occupancy_map_vy[ix, iy] = norm_vy

    # Reset del entrenamiento
    def reset(self, seed=None, options=None):
        self.node.get_logger().info("Reseting environment...")
        NUM_OBSTACLES = 5
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self._publish_cmd_vel()

        self._delete_objective_ball()

        # Eliminar obstáculos previos
        for i in range(1, NUM_OBSTACLES+1):
            self._delete_obstacle(f"cube_{i}")

        while True:
            robot_x = random.uniform(-3.5, 3.5)
            robot_y = random.uniform(-3.5, 3.5)
            robot_orient = random.uniform(0, 2 * math.pi)
            # Comprobamos que no se superponga con ningún actor
            actors_ok = all(
                self._regions_overlap((robot_x, robot_y), (x, y), min_dist=1.5)
                for x, y, _ in self.actors_data.values()
            )
            if actors_ok:
                break
        self._reset_model_position("waffle", position=(robot_x, robot_y, 0.0), orient=robot_orient)

        self._unpause_physics(1.5)
        self._pause_physics()

        # Generar goal con restricción de distancia mínima
        while True:
            goal_x = random.uniform(-3.5, 3.5)
            goal_y = random.uniform(-3.5, 3.5)
            if (
                self._regions_overlap((robot_x, robot_y), (goal_x, goal_y), 3.0) # robot
                and all(self._regions_overlap((goal_x, goal_y), (x, y), min_dist=1.0) for x, y, _ in self.actors_data.values())
            ):
                break

        self.goal = [goal_x, goal_y]
        self._spawn_objective_ball(self.goal[0], self.goal[1], 0.01)

        # Generar obstáculos
        self.obstacles = []
        for i in range(1, NUM_OBSTACLES+1):
            while True:
                obs_x = random.uniform(-3.5, 3.5)
                obs_y = random.uniform(-3.5, 3.5)
                if (
                    all(self._regions_overlap((obs_x, obs_y), obs, min_dist=1.0) for obs in self.obstacles) #obs
                    and self._regions_overlap((obs_x, obs_y), (robot_x, robot_y), 1.5) #robot
                    and self._regions_overlap((obs_x, obs_y), (goal_x, goal_y), 0.75) # goal
                    and all(self._regions_overlap((obs_x, obs_y), (x, y), min_dist=1.0) for x, y, _ in self.actors_data.values())
                ):
                    self.obstacles.append((obs_x, obs_y))
                    break
        
        # Spawnear obstáculos
        for i, (obs_x, obs_y) in enumerate(self.obstacles, 1):
            self.node.get_logger().info(f"Spawning obstacle cube_{i} at ({obs_x}, {obs_y})...")
            self._spawn_obstacle(f"cube_{i}", obs_x, obs_y)    

        self._wait_for_data()

        self.occupancy_map_vx = np.zeros((18, 18), dtype=np.float32)
        self.occupancy_map_vy = np.zeros((18, 18), dtype=np.float32)
        self.prev_human_pos = {actor_id: None for actor_id in self.actors_data.keys()}

        # Asignar personas al mapa de ocupación
        self._assign_to_local_velocity_maps(self.actors_data, self.odom_data)

        distance_to_goal_x, distance_to_goal_y = self.odom_data[0] - self.goal[0], self.odom_data[1] - self.goal[1]
        distance_to_the_goal = math.sqrt(distance_to_goal_x ** 2 + distance_to_goal_y ** 2)
        self.prev_distance = distance_to_the_goal
        angulo_objetivo = self._calcular_angulo_hacia_objetivo(self.odom_data, self.goal)

        self.time = 0
        self.done = False
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0

        self.lidar_history = np.ones((18, 18), dtype=np.float32)

        # Procesar mapa de lidar
        self._process_lidar()

        map_obs = np.stack((self.occupancy_map_vx, self.occupancy_map_vy, self.lidar_history), axis=2)
        map_obs = map_obs.transpose(2, 0, 1)

        # Generar siguiente estado
        self.next_state = {
            "goal_distance": np.array([self._normalize_distance(distance_to_the_goal)], dtype=np.float32),
            "goal_angle": np.array([self._normalize_orientation(angulo_objetivo)], dtype=np.float32),
            "time": np.array([self._normalize_time()], dtype=np.float32),
            "map": map_obs,
            "last_velocity": np.array([-1.0, self.prev_angular_vel], dtype=np.float32)
        }

        self.info = {"success": False, "colision": False, "timeout": False , "min_dist_human": np.inf}

        return self.next_state, self.info
    
    # Renderiza el entorno de simulación
    def render(self, mode='rgb_array'):
        if mode not in ['human', 'rgb_array']:
            raise ValueError(f"Modo de renderización no soportado: {mode}")

        if mode == 'human':
            # Mostrar la imagen usando OpenCV
            return None
        elif mode == 'rgb_array':
            return self.image_data, self.occupancy_map_vx, self.occupancy_map_vy, self.lidar_history

# Función para manejar las señales de cierre del programa
def signal_handler(sig, frame):
    print("Señal recibida: cerrando el programa de forma segura...")
    cleanup()  # Llama a la función de limpieza antes de salir
    sys.exit(0)

# Registrar las señales de interrupción y terminación
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Kill signal

def cleanup():
    if run is not None:
        run.finish()  # Finalizar la sesión de WandB
    # Liberar memoria de GPU en PyTorch
    torch.cuda.empty_cache()
    print("Limpieza completada.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
    args = parser.parse_args()

    env = GazeboEnv(seed=args.seed)
    check_env(env)

    # Para usar GPU:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Crear el callback
    save_video_path = "./videos"

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    load = True # True para cargar modelo y False para empezar desde cero

    model_path = ""

    def linear_schedule(initial_value: float, end_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return end_value + progress_remaining * (initial_value - end_value)
        return func
    
    # Tasas de aprendizaje iniciales y finales
    initial_lr = 3e-4 
    final_lr = 1e-4

    lr_schedule = linear_schedule(initial_lr, final_lr)

    if load:
        model_path = "TFM/model_policy/model_trained_policy.zip"
        model = SAC.load(model_path, env = env, device = device, learning_starts = 2000, learning_rate = lr_schedule)
    else:
        model_path = "TFM/model_policy/model_trained_policy.zip"
        model = SAC("MultiInputPolicy", env, verbose=1, seed = args.seed, device = device, learning_starts = 3000)
    
    # Crear el callback para registrar los episodios y métricas
    callback = WandbCallbackWithVideo(env=env, save_path=save_video_path, model_save_path=model_path)

    params = model.get_parameters()
    reward_params = env._get_reward_params()

    # Combinar ambos en un solo diccionario para pasarlo como configuración
    config_params = {**reward_params, **params}

    wandb.login(key="")

    # Configurar WandB
    run = wandb.init(
        project="Social Navigation RL",  # Nombre del proyecto en WandB
        config=config_params,
        sync_tensorboard=True,
        monitor_gym=True
    )

    try:
        model.learn(
            total_timesteps=150000, # Número de pasos a ejecutar
            callback=[callback]
        )
        model.save(model_path)

    except Exception as e:
        error_msg = f"❌ Error en el entrenamiento: {e}"
        print(error_msg)

    finally:
        # Garantizar que se finaliza la ejecución de la sesión de WandB y otros recursos
        cleanup()