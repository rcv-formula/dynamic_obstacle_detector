from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('obstacle_context')
    use_sim_time = LaunchConfiguration('use_sim_time')

    config_file = os.path.join(
        pkg_dir,
        'config',
        'obstacle_cluster_context.yaml'
    )

    node = Node(
        package='obstacle_context',
        executable='obstacle_cluster_context_node',
        name='obstacle_cluster_context_node',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time}
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use /clock from simulation or rosbag playback'
        ),
        node
    ])
