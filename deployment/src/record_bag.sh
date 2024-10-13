#!/bin/bash

# Create a timestamp for the filename
timestamp=$(date +"%Y_%m%d_%H%M")
filename="rosbag_${timestamp}"

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# ウィンドウを5つの同じサイズのペインに分割
tmux selectp -t 0    # 最初のペインを選択
tmux splitw -h -p 66 # 水平方向に3分の2で分割
tmux selectp -t 0    # 左側のペインを選択
tmux splitw -h -p 50 # 左側を更に半分に分割
tmux selectp -t 2    # 右側のペインを選択
tmux splitw -v -p 66 # 垂直方向に3分の2で分割
tmux selectp -t 2    # 上のペインを選択
tmux splitw -v -p 50 # 上のペインを更に半分に分割
tmux selectp -t 0    # 最初のペインに戻る

# robot setup
tmux select-pane -t 0
tmux send-keys "ros2 launch om_modbus_master om_modbus_master_launch.py" Enter

tmux select-pane -t 1
tmux send-keys "python3 /home/ryuddi/ROS2/om_modbus_master_V201/src/om_modbus_master/sample/BLV_R/twist_to_motor.py" Enter

# camera and joycon setup
tmux select-pane -t 2
tmux send-keys "ros2 run gstreamer_camera gstreamer_camera_node" Enter

tmux select-pane -t 3
tmux send-keys "ros2 launch teleop_twist_joy teleop-launch.py joy_vel:=/cmd_vel_mux/input/teleop" Enter

# twist_mux node setup
tmux select-pane -t 4
tmux send-keys "ros2 run twist_mux_custom twist_mux_custom /ssd/source/navigation/visualnav-transformer/deployment/config/twist_mux.yaml" Enter

# bag recording
tmux select-pane -t 5
tmux send-keys "cd ../topomaps/bags && ros2 bag record /image_raw -o ${filename}" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
