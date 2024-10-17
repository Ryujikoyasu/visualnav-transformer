# python navigate.py --model <model_name> -—dir <topomap_dir>: 
# This python script starts a node that reads in image observations from the /image_raw topic, 
# inputs the observations and the map into the model, 
# and publishes actions to the /waypoint topic.

import matplotlib.pyplot as plt
import sys, os
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
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class Navigation(Node):
    def __init__(self, args):
        super().__init__("NAVIGATION")
        self.args = args
        self.context_queue = []
        self.context_size = None
        self.model = None
        self.noise_scheduler = None
        self.closest_node = 0
        self.goal_node = args.goal_node if args.goal_node != -1 else None
        self.reached_goal = False
        self.topomap = []

        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self.callback_obs, 1)
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)
        self.goal_pub = self.create_publisher(
            Bool, "/topoplan/reached_goal", 1)

        self.load_model()
        self.load_topomap()
        self.timer = self.create_timer(1.0 / RATE, self.timer_callback)

    def load_model(self):
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[self.args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]

        ckpth_path = model_paths[self.args.model]["ckpt_path"]
        if os.path.exists(ckpth_path):
            self.get_logger().info(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        
        self.model = load_model(ckpth_path, self.model_params, device)
        self.model = self.model.to(device)
        self.model.eval()

        if self.model_params["model_type"] == "nomad":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )

    def load_topomap(self):
        topomap_filenames = sorted(os.listdir(os.path.join(
            TOPOMAP_IMAGES_DIR, self.args.dir)), key=lambda x: int(x.split(".")[0]))
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{self.args.dir}"
        num_nodes = len(os.listdir(topomap_dir))
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            self.topomap.append(PILImage.open(image_path))
        
        if self.goal_node is None:
            self.goal_node = len(self.topomap) - 1

    def callback_obs(self, msg):
        obs_img = msg_to_pil(msg)
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)

    def timer_callback(self):
        if len(self.context_queue) > self.model_params["context_size"]:
            chosen_waypoint = np.zeros(4)
            if self.model_params["model_type"] == "nomad":
                chosen_waypoint = self.process_nomad()
            else:
                chosen_waypoint = self.process_other()

            if self.model_params["normalize"]:
                chosen_waypoint[:2] *= (MAX_V / RATE)  
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint.tolist()
            self.waypoint_pub.publish(waypoint_msg)

            self.reached_goal = self.closest_node == self.goal_node
            goal_msg = Bool()
            goal_msg.data = bool(self.reached_goal)  # ここを変更
            self.goal_pub.publish(goal_msg)

            if self.reached_goal:
                self.get_logger().info("Reached goal! Stopping...")

    def process_nomad(self):
        obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1) 
        obs_images = obs_images.to(device)
        mask = torch.zeros(1).long().to(device)  

        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        goal_image = [transform_images(g_img, self.model_params["image_size"], center_crop=False).to(device) for g_img in self.topomap[start:end + 1]]
        goal_image = torch.concat(goal_image, dim=0)

        obsgoal_cond = self.model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
        dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dists = to_numpy(dists.flatten())
        min_idx = np.argmin(dists)
        self.closest_node = min_idx + start
        self.get_logger().info(f"Closest node: {self.closest_node}")
        sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

        with torch.no_grad():
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
            
            noisy_action = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=device)
            naction = noisy_action

            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])

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
        self.get_logger().info("Published sampled actions")
        self.sampled_actions_pub.publish(sampled_actions_msg)
        naction = naction[0] 
        return naction[self.args.waypoint]

    def process_other(self):
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        batch_obs_imgs = []
        batch_goal_data = []
        for sg_img in self.topomap[start: end + 1]:
            transf_obs_img = transform_images(self.context_queue, self.model_params["image_size"])
            goal_data = transform_images(sg_img, self.model_params["image_size"])
            batch_obs_imgs.append(transf_obs_img)
            batch_goal_data.append(goal_data)
            
        batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
        batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

        distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
        distances = to_numpy(distances)
        waypoints = to_numpy(waypoints)
        min_dist_idx = np.argmin(distances)
        if distances[min_dist_idx] > self.args.close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][self.args.waypoint]
            self.closest_node = start + min_dist_idx
        else:
            chosen_waypoint = waypoints[min(
                min_dist_idx + 1, len(waypoints) - 1)][self.args.waypoint]
            self.closest_node = min(start + min_dist_idx + 1, self.goal_node)
        return chosen_waypoint

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
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
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    
    try:
        rclpy.spin(navigation)
    except KeyboardInterrupt:
        pass
    finally:
        navigation.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
