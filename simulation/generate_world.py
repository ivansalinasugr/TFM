#!/usr/bin/env python3
import argparse
import random
import math
import xml.etree.ElementTree as ET
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
use_walk = True


TRAIN_BASE_WORLD_PATH = "/home/ivan/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/empty_world_base_train.world"
TEST_BASE_WORLD_PATH = "/home/ivan/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/empty_world_base_test.world"

OUTPUT_WORLD_PATH = "/home/ivan/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/empty_world.world"

# Zonas iniciales del robot
ROBOT_INITIAL_POSES = [
    (-3.0, 3.0),
    (3.5, 2.5),
    (0.0, 0.0),
    (3.5, -2.5),
    (-3.5, -3.5)
]

# Zonas donde existen obstáculos fijos
FORBIDDEN_ZONES = [
    (-4.5, -1.25, -2.5, 2.5),       # Zona A
    (-1.25, 1.0, -4.5, -3.0),       # Zona B
    (1, 3.5, -4.5, -3.25),    # Zona C
    (3, 4.5, -4.5, 4.5),         # Zona D
    (0.25, 3, 3.5, 4.5)       # Zona E
]

ALL_WAYPOINTS = []

def is_in_forbidden_zone(x, y):
    for xmin, xmax, ymin, ymax in FORBIDDEN_ZONES:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False

def is_valid_position(x, y, other_actor_positions):
    if not (-4.0 <= x <= 4.0 and -4.0 <= y <= 4.0):
        return False
    if is_in_forbidden_zone(x, y):
        return False
    for rx, ry in ROBOT_INITIAL_POSES:
        if math.hypot(x - rx, y - ry) < 1.0:
            return False
    for ax, ay in other_actor_positions:
        if math.hypot(x - ax, y - ay) < 1.5:
            return False
    return True


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    elif not level:
        elem.tail = "\n"

def generate_human_like_waypoints(start=None, min_steps=2, max_steps=5):
    global ALL_WAYPOINTS
    n = random.randint(min_steps, max_steps)
    waypoints = []

    if start is None:
        x = random.uniform(-3.0, 3.0)
        y = random.uniform(-3.0, 3.0)
    else:
        x, y = start

    if not is_in_forbidden_zone(x, y):
        wp = (round(x, 2), round(y, 2), 1.25)
        if all(math.hypot(x - ox, y - oy) >= 0.5 for ox, oy, _ in ALL_WAYPOINTS):
            waypoints.append(wp)
            ALL_WAYPOINTS.append(wp)

    for _ in range(n - 1):
        for _ in range(100):  # Máximo 100 intentos por waypoint
            angle = random.uniform(-math.pi/2, math.pi/2)
            distance = random.uniform(2.0, 4.0)
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)
            new_x = x + dx
            new_y = y + dy

            if (
                -4.0 <= new_x <= 4.0 and
                -4.0 <= new_y <= 4.0 and
                not is_in_forbidden_zone(new_x, new_y) and
                all(math.hypot(new_x - ox, new_y - oy) >= 0.5 for ox, oy, _ in ALL_WAYPOINTS)
            ):
                x, y = new_x, new_y
                wp = (round(x, 2), round(y, 2), 1.25)
                waypoints.append(wp)
                ALL_WAYPOINTS.append(wp)
                break
        else:
            print("No se pudo encontrar un waypoint válido suficientemente alejado.")

    return waypoints



def create_actor_element(index, other_actor_positions):
    global use_walk
    for _ in range(100):
        x_init = round(random.uniform(-4.0, 4.0), 2)
        y_init = round(random.uniform(-4.0, 4.0), 2)
        if is_valid_position(x_init, y_init, other_actor_positions):
            break
    else:
        raise ValueError(f"No se encontró una posición válida para el actor {index}")

    other_actor_positions.append((x_init, y_init))
    z_init = 1.1
    print(f"Posición actor {index}: (", x_init, ",", y_init, ")")

    actor = ET.Element("actor", name=f"actor{index}")
    ET.SubElement(actor, "pose").text = f"{x_init} {y_init} {z_init} 0 0 0"

    animation = "walk" if use_walk else "moonwalk"
    use_walk = not use_walk  # Alterna para la próxima vez la vestimenta

    skin = ET.SubElement(actor, "skin")
    ET.SubElement(skin, "filename").text = f"{animation}.dae"
    ET.SubElement(skin, "scale").text = "1.0"

    anim = ET.SubElement(actor, "animation", name="walking")
    ET.SubElement(anim, "filename").text = "walk.dae"
    ET.SubElement(anim, "scale").text = "1.0"
    ET.SubElement(anim, "interpolate_x").text = "true"

    plugin = ET.SubElement(actor, "plugin", name=f"actor{index}_plugin", filename="libPedestrianSFMPlugin.so")
    ET.SubElement(plugin, "velocity").text = "0.2"
    ET.SubElement(plugin, "radius").text = "0.25"
    ET.SubElement(plugin, "animation_factor").text = "5.1"
    ET.SubElement(plugin, "people_distance").text = "5.0"
    ET.SubElement(plugin, "goal_weight").text = "10.0"
    ET.SubElement(plugin, "obstacle_weight").text = "60.0"
    ET.SubElement(plugin, "social_weight").text = "0.0"
    ET.SubElement(plugin, "group_gaze_weight").text = "0.0"
    ET.SubElement(plugin, "group_coh_weight").text = "0.0"
    ET.SubElement(plugin, "group_rep_weight").text = "0.0"

    ignore = ET.SubElement(plugin, "ignore_obstacles")
    for model in ["ground_plane", "aerial_camera", f"actor{index}_collision_box"]:
        ET.SubElement(ignore, "model").text = model

    traj = ET.SubElement(plugin, "trajectory")
    ET.SubElement(traj, "cyclic").text = "true"
    for x, y, z in generate_human_like_waypoints(start = (x_init, y_init)):
        ET.SubElement(traj, "waypoint").text = f"{x} {y} {z}"

    return actor

def insert_actors_to_world(base_world_path, output_path, actor_count):
    ET.register_namespace('', "http://sdformat.org/schemas/root.xsd")
    tree = ET.parse(base_world_path)
    root = tree.getroot()
    world = root.find("world")

    actor_positions = []
    for i in range(1, actor_count + 1):
        actor_element = create_actor_element(i, actor_positions)
        world.append(actor_element)

    indent(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Mundo generado con {actor_count} actores en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generador de mundos con actores aleatorios")
    parser.add_argument('--actors', type=int, default=5, help="Número de actores a insertar")
    parser.add_argument('--test', type=int, default=0, help="Tipo de entorno (train 0 o test 1)")
    args = parser.parse_args()

    if args.test == 0:
        insert_actors_to_world(TRAIN_BASE_WORLD_PATH, OUTPUT_WORLD_PATH, args.actors)
    else:
        insert_actors_to_world(TEST_BASE_WORLD_PATH, OUTPUT_WORLD_PATH, args.actors)

if __name__ == "__main__":
    main()
