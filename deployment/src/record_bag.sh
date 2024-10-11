#!/bin/bash

# Create a timestamp for the filename
timestamp=$(date +"%Y_%m%d_%H%M")
filename="rosbag_${timestamp}"

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

# robot setup
tmux select-pane -t 0
tmux send-keys "ros2 launch om_modbus_master om_modbus_master_launch.py" Enter
tmux send-keys "python3 /home/ryuddi/ROS2/om_modbus_master_V201/src/om_modbus_master/sample/BLV_R/twist_to_motor.py" Enter

# camera and joycon setup
tmux select-pane -t 1
tmux send-keys "ros2 run gstreamer_camera gstreamer_camera_node" Enter
tmux send-keys "ros2 launch teleop_twist_joy teleop-launch.py joy_vel:=/cmd_vel_mux/input/teleop" Enter

# twist_mux node setup
tmux select-pane -t 2
tmux send-keys "ros2 run twist_mux twist_mux --ros-args -p config_file:=/deployment/config/twist_mux.yaml" Enter

# bag recording
tmux select-pane -t 3
tmux send-keys "cd ../topomaps/bags && ros2 bag record /image_raw -o ${filename}" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name