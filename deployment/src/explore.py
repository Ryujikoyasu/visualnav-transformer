import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)

# CONSTANTS
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# GLOBALS
context_queue = []
context_size = None  

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class Exploration(Node):
    def __init__(self, args):
        super().__init__("EXPLORATION")
        self.args = args
        self.context_queue = []
        self.context_size = None
        self.model = None
        self.noise_scheduler = None

        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self.callback_obs, 1)
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)

        self.load_model()
        self.timer = self.create_timer(1.0 / RATE, self.timer_callback)

    def load_model(self):
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[self.args.model]["config_path"]
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)

        self.context_size = model_params["context_size"]

        ckpth_path = model_paths[self.args.model]["ckpt_path"]
        if os.path.exists(ckpth_path):
            self.get_logger().info(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        
        self.model = load_model(ckpth_path, model_params, device)
        self.model = self.model.to(device)
        self.model.eval()

        num_diffusion_iters = model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def callback_obs(self, msg):
        obs_img = msg_to_pil(msg)
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)

    def timer_callback(self):
        if len(self.context_queue) > self.context_size:
            obs_images = transform_images(self.context_queue, self.model.image_size, center_crop=False)
            obs_images = obs_images.to(device)
            fake_goal = torch.randn((1, 3, *self.model.image_size)).to(device)
            mask = torch.ones(1).long().to(device)

            with torch.no_grad():
                obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
                
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
                
                noisy_action = torch.randn(
                    (self.args.num_samples, self.model.len_traj_pred, 2), device=device)
                naction = noisy_action

                self.noise_scheduler.set_timesteps(self.model.num_diffusion_iters)

                start_time = time.time()
                for k in self.noise_scheduler.timesteps[:]:
                    noise_pred = self.model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                self.get_logger().info(f"Time elapsed: {time.time() - start_time}")

            naction = to_numpy(get_action(naction))
            
            sampled_actions_msg = Float32MultiArray()
            sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten())).tolist()
            self.sampled_actions_pub.publish(sampled_actions_msg)

            naction = naction[0]

            chosen_waypoint = naction[self.args.waypoint]

            if self.model.normalize:
                chosen_waypoint *= (MAX_V / RATE)
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint.tolist()
            self.waypoint_pub.publish(waypoint_msg)
            self.get_logger().info("Published waypoint")

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using device: {device}")

    exploration = Exploration(args)
    
    try:
        rclpy.spin(exploration)
    except KeyboardInterrupt:
        pass
    finally:
        exploration.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()