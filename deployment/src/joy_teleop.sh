#!/bin/bash

# Create a new tmux session
session_name="teleop_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into two panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

# Run the ros2 launch command in the first pane
tmux select-pane -t 0
tmux send-keys "ros2 launch om_modbus_master om_modbus_master_launch.py" Enter
tmux send-keys "python3 /home/ryuddi/ROS2/om_modbus_master_V201/src/om_modbus_master/sample/BLV_R/twist_to_motor.py" Enter
tmux send-keys "ros2 run gstreamer_camera gstreamer_camera_node" Enter

# Run the joy_teleop.py script in the second pane
tmux select-pane -t 1
tmux send-keys "ros2 launch teleop_twist_joy teleop-launch.py joy_vel:=/cmd_vel_mux/input/teleop" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name