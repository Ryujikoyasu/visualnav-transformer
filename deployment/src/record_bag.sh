#!/bin/bash

# Create a timestamp for the filename
timestamp=$(date +"%Y_%m%d_%H%M")
filename="rosbag_${timestamp}"

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# ウィンドウを6つの同じサイズのペインに分割
# 最初のペインを2つに分割（上下）
tmux split-window -v -t $session_name:0.0
# 上のペインを3つに分割（左右）
tmux split-window -h -t $session_name:0.0
tmux split-window -h -t $session_name:0.0
# 下のペインを3つに分割（左右）
tmux split-window -h -t $session_name:0.3
tmux split-window -h -t $session_name:0.3
# レイアウトを調整して等面積にする
tmux select-layout -t $session_name:0 tiled


# robot setup
tmux select-pane -t 0
tmux send-keys "ros2 launch om_modbus_master om_modbus_master_launch.py" Enter

tmux select-pane -t 1
tmux send-keys "python3 /home/ryuddi/ROS2/om_modbus_master_V201/src/om_modbus_master/sample/BLV_R/twist_to_motor.py" Enter

# camera and joycon setup
tmux select-pane -t 2
tmux send-keys "ros2 run gstreamer_camera gstreamer_camera_node" Enter

tmux select-pane -t 3
tmux send-keys "conda activate vint_deployment_2" Enter
tmux send-keys "python joy_teleop.py" Enter

# twist_mux node setup
tmux select-pane -t 4
tmux send-keys "ros2 run twist_mux_custom twist_mux_custom /ssd/source/navigation/visualnav-transformer/deployment/config/twist_mux.yaml" Enter

# bag recording
tmux select-pane -t 5
tmux send-keys "cd ../topomaps/bags && ros2 bag record /image_raw -o ${filename}" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
