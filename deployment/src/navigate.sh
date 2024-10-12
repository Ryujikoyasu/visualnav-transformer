#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into six panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 66 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 3    # select the new, third (3) pane
tmux splitw -v -p 66 # split it into two halves
tmux selectp -t 3    # select the new, third (3) pane
tmux splitw -v -p 50 # split it into two halves

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
tmux select-pane -t 4
tmux send-keys "conda activate vint_deployment_py310" Enter
tmux send-keys "python navigate.py $@" Enter

# Run the twist_mux script in the sixth pane
tmux select-pane -t 5
tmux send-keys "ros2 run twist_mux twist_mux --ros-args -p config_file:=/deployment/config/twist_mux.yaml" Enter

# Run the pd_controller.py script in the seventh pane
tmux select-pane -t 6
tmux send-keys "conda activate vint_deployment_py310" Enter
tmux send-keys "python pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
