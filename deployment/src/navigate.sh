#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# 最初のペインを2つに分割（上下）
tmux split-window -v -t $session_name:0.0
# 上のペインを3つに分割（左右）
tmux split-window -h -t $session_name:0.0
tmux split-window -h -t $session_name:0.0
# 下のペインを2つに分割（左右）
tmux split-window -h -t $session_name:0.3
# 左下のペインを上下に分割
tmux split-window -v -t $session_name:0.3
# 右下のペインを上下に分割
tmux split-window -v -t $session_name:0.4
# レイアウトを調整
tmux select-layout -t $session_name:0 tiled

# Run the ros2 launch command in the first pane
tmux select-pane -t 0
tmux send-keys "ros2 launch om_modbus_master om_modbus_master_launch.py" Enter

# Run the twist_to_motor.py script in the second pane
tmux select-pane -t 1
tmux send-keys "python3 /home/ryuddi/ROS2/om_modbus_master_V201/src/om_modbus_master/sample/BLV_R/twist_to_motor.py" Enter

# Run the gstreamer_camera_node in the third pane
tmux select-pane -t 2
tmux send-keys "ros2 run gstreamer_camera gstreamer_camera_node" Enter

# Run the teleop-launch.py script in the fourth pane
tmux select-pane -t 3
tmux send-keys "ros2 launch teleop_twist_joy teleop-launch.py joy_vel:=/cmd_vel_mux/input/teleop" Enter

# Run the navigate.py script with command line args in the fifth pane
tmux select-pane -t 6
tmux send-keys "conda activate vint_deployment_2" Enter
tmux send-keys "python navigate.py $(printf '%q ' "$@")" Enter

# Run the twist_mux script in the sixth pane
tmux select-pane -t 5
tmux send-keys "ros2 run twist_mux_custom twist_mux_custom /ssd/source/navigation/visualnav-transformer/deployment/config/twist_mux.yaml" Enter

# Run the pd_controller.py script in the seventh pane
tmux select-pane -t 4
tmux send-keys "conda activate vint_deployment_2" Enter
tmux send-keys "python pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
