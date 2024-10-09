from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # コマンドライン引数の宣言
        DeclareLaunchArgument(
            'config_file',
            default_value='/home/racecar/vint_release/gnm-v2/deployment/config/cmd_vel_mux.yaml',
            description='Path to the velocity multiplexer config file'
        ),

        # 自作ロボットの立ち上げ
        Node(
            package='om_modbus_master',
            executable='om_modbus_master_node',
            name='om_modbus_master',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', '/home/ryuddi/ROS2/om_modbus_master_V201/src/om_modbus_master/sample/BLV_R/twist_to_motor.py'],
            output='screen'
        ),

        # カメラの立ち上げ
        Node(
            package='gstreamer_camera',
            executable='gstreamer_camera_node',
            name='gstreamer_camera',
            output='screen'
        ),

        # Joyconの立ち上げ
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            output='screen'
        ),

        # Velocity Multiplexer の起動
        Node(
            package='yocs_cmd_vel_mux',
            executable='cmd_vel_mux',
            name='cmd_vel_mux',
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
            output='screen'
        ),
    ])
