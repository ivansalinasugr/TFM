# Navegación autónoma social mediante aprendizaje por refuerzo profundo

Trabajo Fin de Máster – Máster en Ciencia de Datos e Ingeniería de Computadores

Universidad de Granada

Autor: **Iván Salinas López**

---

## Descripción

Este proyecto implementa un sistema de navegación autónoma para robots móviles en entornos sociales, utilizando **aprendizaje por refuerzo profundo (Deep Reinforcement Learning, DRL)**. El objetivo es que un robot **TurtleBot3** aprenda a navegar evitando colisiones tanto con obstáculos estáticos como con personas en movimiento, alcanzando un objetivo dentro de un entorno simulado, realista y dinámico.

La simulación se desarrolla con **ROS 2 Humble**, **Gazebo Classic** y el modelo **TurtleBot3 Waffle**. Además, se integra un plugin (`gazebo_sfm_plugin`) que simula actores humanos con comportamiento social y que ha sido modificado para permitir **colisiones físicas entre el robot y los actores**, lo cual es esencial para la correcta simulación.

---

## Requisitos

* **Ubuntu 22.04**
* **ROS 2 Humble**
* **Gazebo Simulator (Classic)**
* **TurtleBot3** (`waffle`)
* **Plugin `gazebo_sfm_plugin` modificado**:
  [https://github.com/robotics-upo/gazebo\_sfm\_plugin/tree/galactic](https://github.com/robotics-upo/gazebo_sfm_plugin/tree/galactic)

> ⚠️ El plugin debe ser recompilado (`make`) tras reemplazar los archivos `PedestrianSFMPlugin.cpp` y `PedestrianSFMPlugin.h` por los incluidos en este repositorio. Estos permiten detectar colisiones entre el robot y los actores humanos.

---

## Instalación

1. Instalar ROS 2 Humble, Gazebo Classic y TurtleBot3 según su documentación oficial.
2. Clonar el plugin `gazebo_sfm_plugin`, sustituir los archivos `.cpp` y `.h` por los proporcionados, recompilar y copiar a los plugins de Gazebo:

   ```bash
   cd gazebo_sfm_plugin
   make
   sudo cp libPedestrianSFMPlugin.so /usr/lib/x86_64-linux-gnu/gazebo-11/plugins/
   ```
3. Instalar las dependencias de Python:

   ```bash
   pip install -r requirements.txt
   ```

---

## Archivos de simulación

* `objective.sdf`: cilindro rojo que representa (sólo por estética visual) el **objetivo** que el robot debe alcanzar.
* `cube.sdf`: cubos estáticos que actúan como **obstáculos fijos** en el entorno.
* `walls.sdf`: contiene únicamente las **paredes del entorno**.
* `empty_world_base_train.world`: archivo principal del mundo de Gazebo, para entrenamiento, sobre el cual se insertan dinámicamente los actores.
* `empty_world_base_test.world`: archivo principal del mundo de Gazebo, para evaluación (es decir, con mesas, sillas, sofás), sobre el cual se insertan dinámicamente los actores.
* `empty_world.world`: archivo resultante del mundo de Gazebo, donde se han insertado los actores, ya sea para entrenamiento o para evaluación.

---

## Ejecución

### Parámetros importantes

* `num_actors`: número de humanos simulados.
* `test`: indica si el entorno es para entrenamiento (`False`) o evaluación (`True`). Para el entorno de entrenamiento solo hay paredes (y obstáculos fijos creados dinámicamente durante el entrenamiento) mientras que para el entorno de evaluación, hay paredes, mesas, sillas y sofás.

### 1. Entrenamiento

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle

ros2 launch turtlebot3_gazebo empty_world.launch.py num_actors:=4 test:=False

python3 entrenamiento.py --num_timesteps 100000
```

> La política entrenada se almacena en un archivo `.zip`.

### 2. Evaluación

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle

ros2 launch turtlebot3_gazebo empty_world.launch.py num_actors:=4 test:=True

python3 evaluacion.py --num_episodes 20
```

> Los resultados de evaluación se guardan automáticamente en un archivo `.jsonl`.

---

## Plugin de actores humanos

Este proyecto utiliza el plugin `gazebo_sfm_plugin` para simular personas que siguen un modelo de comportamiento social. El script `generate_world.py` se ejecuta automáticamente al lanzar el entorno con `ros2 launch`, y se encarga de crear el mundo (`empty_world.world`) para insertar a las personas simuladas.

Ejemplo:

```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py num_actors:=5 test:=False
```

Este ejemplo añadiría 5 actores, junto con sus trayectorias de movimiento, al entorno de simulación.

---

## Dependencias de Python (v3.10)

* `cv_bridge==3.2.1`
* `gazebo_msgs==3.9.0`
* `geometry_msgs==4.8.0`
* `gymnasium==1.1.1`
* `imageio==2.37.0`
* `matplotlib==3.5.1`
* `nav2_msgs==1.1.18`
* `nav_msgs==4.8.0`
* `numpy==1.21.5`
* `rclpy==3.3.16`
* `requests==2.32.4`
* `sensor_msgs==4.8.0`
* `stable_baselines3==2.6.0`
* `std_srvs==4.8.0`
* `torch==2.6.0`
* `wandb==0.20.1`

Instalación:

```bash
pip install -r requirements.txt
```

---

## Licencia

Este proyecto se distribuye bajo licencia **MIT**, salvo donde se indique lo contrario (por ejemplo, el plugin `gazebo_sfm_plugin`, cuya licencia original debe ser consultada en su repositorio).

---

## Contacto

Iván Salinas López
\[[ivansalinas@correo.ugr.es](mailto:ivansalinas@correo.ugr.es)]
Universidad de Granada