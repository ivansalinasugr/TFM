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
import re
import json
import copy

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

# Calcular LDLJ
def compute_ldlj(velocities, dt):
    velocities = np.array(velocities)
    
    if len(velocities) < 5:
        return 0.0  # Muy pocos puntos para estimar derivadas de orden 3
    
    # Derivadas numéricas
    acceleration = np.gradient(velocities, dt)
    jerk = np.gradient(acceleration, dt)

    # Integral aproximada del jerk^2
    jerk_squared = jerk ** 2
    integral_jerk_squared = np.sum(jerk_squared) * dt

    duration = len(velocities) * dt
    peak_velocity = np.max(np.abs(velocities))

    if peak_velocity == 0 or duration == 0:
        return 0.0

    epsilon = 1e-12
    ratio = integral_jerk_squared / (peak_velocity ** 2 * duration ** 5)
    ldlj = np.log(max(ratio, epsilon))

    return ldlj

# Calcular índice de colisión 
def compute_ci_series(positions, human_positions_series, sigma_px=0.45, sigma_py=0.45):
    ci_values = []

    for robot_pos, humans in zip(positions, human_positions_series):
        x_r, y_r = robot_pos

        if not humans:  # No hay humanos cerca
            ci = 0.0
        else:
            ci = max(
                np.exp(-(((x_r - x_h)**2) / (2 * sigma_px**2) + ((y_r - y_h)**2) / (2 * sigma_py**2)))
                for (x_h, y_h) in humans
            )
        ci_values.append(ci)

    ci_mean = np.mean(ci_values)
    ci_std = np.std(ci_values)
    ci_max = np.max(ci_values)

    return ci_mean, ci_std, ci_max, ci_values

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
            "MAX_STEPS": 1000,
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

        # Lista de objetivos posibles
        self.lista_objetivos = [
            (-4.0, 2.5),
            (-3.0, 3.5), 
            (-0.5, 4.0),
            (1.0, 2.0),
            (3.5, 2.0),
            (3.5, 3.5),
            (-0.5, -0.5),
            (2.0, -1.5),
            (4.0, -3.0),
            (1.0, -3.0),
            (-2.5, -3.5),
            (-4.0, -2.0)
        ]

        # Lista de poses iniciales
        self.lista_pose_inicial = [
            (-3.0, 3.0, 0.0),
            (3.5, 2.5, math.pi),
            (0.0, 0.0, 0.0),
            (3.5, -2.5, math.pi),
            (-3.5, -3.5, 0.0)
        ]

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

        # Guardar posiciones para calcular métricas de evaluación
        self.info["robot_position"] = self.odom_data[:-1]
        self.info["lista_pos_humans"] = self._get_human_positions_in_front()

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

    # Obtener las posiciones de las personas que están delante del robot (que son las que nos importan)
    def _get_human_positions_in_front(self):
        robot_pos = self.odom_data  # (x, y, theta)
        humans_in_front = []

        for _, human_data in self.actors_data.items():
            human_pos = human_data[:2]  # (x, y)
            angulo_relativo = self._calcular_angulo_hacia_objetivo(robot_pos, human_pos)

            if abs(angulo_relativo) <= math.pi / 2:
                humans_in_front.append(human_pos)

        return humans_in_front

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
        elif min(self.lidar_data) <= 0.15 and self.time > 5:
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
            self.node.get_logger().error("Error al resetear posición del robot.")
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
        NUM_OBSTACLES = 0 # Número de obstáculos fijos a generar
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self._publish_cmd_vel()

        self._delete_objective_ball()

        while True:
            # Elegimos aleatoriamente una pose inicial de la lista
            pose = random.choice(self.lista_pose_inicial)
            robot_x, robot_y, robot_orient = pose

            # Comprobamos que no se superponga con ningún actor
            actors_ok = all(
                self._regions_overlap((robot_x, robot_y), (x, y), min_dist=1.5)
                for x, y, _ in self.actors_data.values()
            )
            
            if actors_ok:
                break

        self._reset_model_position("waffle", position=(robot_x, robot_y, 0.0), orient=robot_orient)

        self._unpause_physics(2.0)
        self._pause_physics()

        # Elegimos aleatoriamente un objetivo de la lista
        while True:
            goal = random.choice(self.lista_objetivos)
            goal_x, goal_y = goal

            # Comprobamos condiciones de distancia
            if (
                self._regions_overlap((robot_x, robot_y), (goal_x, goal_y), 3.0) and
                all(self._regions_overlap((goal_x, goal_y), (x, y), min_dist=1.0) for x, y, _ in self.actors_data.values())
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

        self.info = {"success": False, "colision": False , "robot_position": self.odom_data[:-1], "goal_initial_distance": distance_to_the_goal, "lista_pos_humans": self._get_human_positions_in_front()}

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

# Clase para calcular las métricas de evaluación
class TestEvaluator():
    def __init__(self, env, save_path):
        self.env = env
        self.save_path = save_path
        self.episode_log_path = "episode_logs.jsonl"

        # Borrar el archivo si existe para empezar desde cero
        with open(self.episode_log_path, "w") as f:
            pass 

        self.episode_rewards = []
        self.current_episode_rewards = []
        self.video_frames = []
        self.actions = []
        self.time = 0
        self.num_episodios_metricas = 0
        self.current_info = []
        self.metrics = {"tasa_exito": 0, "tasa_colision": 0, "average_time": 0, "average_distance": 0, "colision_index_mean": 0, "ldlj_lineal": 0, "ldlj_angular": 0}

        # Para visualizar entorno, mapa de ocupación y lidar
        sep = 32
        self.h_sep = np.full((sep, 800 + sep + 800, 3), 255, dtype=np.uint8)  # blanco entre ocupación y lidar
        self.v_sep = np.full((800, sep, 3), 255, dtype=np.uint8)  # gris vertical

        # Crear directorio de videos si no existe
        os.makedirs(save_path, exist_ok=True)

    def step(self, obs, reward, done, info) -> bool:
        action = [self.env.linear_velocity, self.env.angular_velocity]
        self.current_episode_rewards.append(reward)
        self.actions.append(action)
        self.current_info.append(info)

        # Renderizar el entorno y agregar el frame redimensionado al video
        frame, occup_map_vx, occup_map_vy, lidar_history = self.env.render(mode='rgb_array')

        combined_top = np.hstack((frame, self.v_sep, lidar_map_to_image(lidar_history)))
        combined_bottom = np.hstack((occupancy_map_to_image(occup_map_vx), self.v_sep, occupancy_map_to_image(occup_map_vy)))
        combined_frame = np.vstack((combined_top, self.h_sep, combined_bottom))

        # Guardar imagen en buffer
        self.video_frames.append(combined_frame)

        self.time += 1

        return True

    def episode_end(self) -> None:
        # Registrar recompensa total del episodio
        episode_reward = np.sum(self.current_episode_rewards)
        self.episode_rewards.append(episode_reward)

        # Guardar video del episodio
        video_path = os.path.join(self.save_path, f"episode_{len(self.episode_rewards)}.mp4")
        self._save_video(self.video_frames, video_path)

        # Extraer todas las velocidades lineales y angulares de self.actions
        linear_actions = [action[0] for action in self.actions]  
        angular_actions = [action[1] for action in self.actions]  

        dt = 0.1  # Ajusta según tu timestep real
        ldlj_linear = compute_ldlj(linear_actions, dt)
        ldlj_angular = compute_ldlj(angular_actions, dt)

        # Sumar al acumulado de métricas
        self.metrics["ldlj_lineal"] += ldlj_linear
        self.metrics["ldlj_angular"] += ldlj_angular

        positions = [x["robot_position"] for x in self.current_info]
        path_length = sum(np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i])) for i in range(len(positions)-1))
        euclidean_dist = self.current_info[0]["goal_initial_distance"]
        eficiencia = euclidean_dist / (path_length + 1e-6)

        self.num_episodios_metricas += 1
        for info in self.current_info:
            if info["success"]:
                self.metrics["tasa_exito"] += 1
                self.metrics["average_time"] += self.time * dt
                self.metrics["average_distance"] += eficiencia
                break
            elif info["colision"]:
                self.metrics["tasa_colision"] += 1
                break

        human_positions = [x["lista_pos_humans"] for x in self.current_info]
        ci_mean, ci_std, ci_max, ci_values = compute_ci_series(positions, human_positions)
        self.metrics["colision_index_mean"] += ci_mean

        metrics_processed = {}
        for k, v in self.metrics.items():
            if k in ["distancia_media_minima_personas", "ldlj_lineal", "ldlj_angular", "colision_index_mean"]:
                metrics_processed[k] = v / self.num_episodios_metricas
            elif k in ["average_time", "average_distance"]:
                tasa_exito = self.metrics.get("tasa_exito", 0)
                if tasa_exito != 0:
                    metrics_processed[k] = v / tasa_exito
                else:
                    metrics_processed[k] = 0
            else:
                metrics_processed[k] = (v / self.num_episodios_metricas) * 100

        # Loggear en WandB
        log_data = {
            "episode_reward": episode_reward,
            "rewards_mean": np.mean(self.episode_rewards),
            "time": self.time,
            "ci_max": ci_max,
            "linear_hist": wandb.Histogram(linear_actions),
            "angular_hist": wandb.Histogram(angular_actions),
            "video": wandb.Video(video_path, format="mp4"),
            **metrics_processed
        }

        wandb.log(log_data)

        # Guardar datos del episodio en archivo JSONL
        episode_data = {
            "episode_number": len(self.episode_rewards),
            "reward_total": episode_reward,
            "actions": self.actions,
            "info": self.current_info,
            "ldlj_linear": ldlj_linear,
            "ldlj_angular": ldlj_angular,
            "path_length": path_length,
            "initial_distance": euclidean_dist,
            "efficiency": eficiencia,
            "positions": positions,
            "human_positions": human_positions,
            "ci_mean": ci_mean,
            "ci_std": ci_std,
            "ci_max": ci_max,
            "time": self.time
        }

        # Guardar como línea JSON
        with open(self.episode_log_path, "a") as f:
            json.dump(episode_data, f, indent=4)
            f.write("\n\n")


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

    # Carpeta para guardar los vídeos de cada ejecución
    save_video_path = "./videos"

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    model_path = "TFM/model_policy/model_trained_policy.zip"
    model = SAC.load(model_path, env = env, device = device)
    
    # Crear clase para evaluación de métricas
    evaluator = TestEvaluator(env=env, save_path=save_video_path)

    params = model.get_parameters()
    reward_params = env._get_reward_params()

    # Combinar ambos en un solo diccionario para pasarlo como configuración a wandb
    config_params = {**reward_params, **params}

    wandb.login(key="") # poned vuestra clave

    # Configurar WandB
    run = wandb.init(
        project="Social Navigation RL",  # Nombre del proyecto en WandB
        config=config_params,
        sync_tensorboard=True,
        monitor_gym=True
    )

    # Número de episodios a testear
    num_episodes = 100

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)

                evaluator.step(obs, reward, done, copy.deepcopy(info))

            evaluator.episode_end()

    except Exception as e:
        error_msg = f"❌ Error en el entrenamiento: {e}"
        print(error_msg)

    finally:
        # Garantizar que se finaliza la ejecución de la sesión de WandB y otros recursos
        cleanup()