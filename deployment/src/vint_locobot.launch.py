from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='twist_mux',
            executable='twist_mux',
            name='twist_mux',
            parameters=['/deployment/config/twist_mux.yaml'],
            output='screen'
        ),
    ])
