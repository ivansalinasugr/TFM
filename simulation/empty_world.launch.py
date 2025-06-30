import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # Directorios
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Configuraciones
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-3.5')
    y_pose = LaunchConfiguration('y_pose', default='3.5')
    num_actors = LaunchConfiguration('num_actors')
    test = LaunchConfiguration('test')

    # Ruta del mundo generado dinámicamente
    generated_world_path = os.path.join(turtlebot3_gazebo_dir, 'worlds', 'empty_world.world')

    # Paso 1: Ejecutar script Python que genera el mundo con actores
    generate_world_cmd = ExecuteProcess(
        cmd=['python3', '/TFM/simulation/generate_world.py', '--actors', num_actors, '--test', test],
        output='screen'
    )

    # Paso 2: Lanzar gzserver con el mundo generado
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': generated_world_path, 'pause': 'true'}.items()
    )

    # Paso 3: Lanzar gzclient (interfaz gráfica)
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # Paso 4: Publicador de estado del robot
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_dir, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Paso 5: Spawnear el TurtleBot3
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_dir, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    # Paso 6: Spawnear los muros
    spawn_walls_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'walls',
            '-file', '/TFM/SDFs/walls.sdf',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # Paso 7: Spawnear cámara aérea (para visualizar en wandb)
    spawn_camera_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'aerial_camera',
            '-file', '/home/ivan/.gazebo/models/aerial_camera_2/model.sdf',
            '-x', '0',
            '-y', '0',
            '-z', '0'
        ],
        output='screen'
    )

    # Ensamblar todo
    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('num_actors', default_value='1'))
    ld.add_action(DeclareLaunchArgument('test', default_value='0'))
    ld.add_action(generate_world_cmd)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(spawn_walls_cmd)
    ld.add_action(spawn_camera_cmd)

    return ld