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
from vint_train.models.nomad.nomad_adapter import NoMaDAdapter
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
ROBOT_CONFIG_PATH = "../config/robot.yaml"
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

class ExplorationAdapter(Node):
    def __init__(self, args):
        super().__init__("EXPLORATION_ADAPTER")
        self.args = args
        self.context_queue = []
        self.context_size = None
        self.model = None
        self.noise_scheduler = None
        self.adapters = {}  # 複数のアダプターを保持する辞書

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
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]

        # ベースモデルの読み込み
        base_model = load_model(model_paths[self.args.model]["ckpt_path"], self.model_params, device)
        
        # アダプターモデルの初期化
        self.model = NoMaDAdapter(
            base_model=base_model,
            adapter_bottleneck_dim=self.model_params["adapter_bottleneck_dim"]
        ).to(device)

        # 複数のアダプターの読み込み
        for task_name, adapter_path in model_paths[self.args.model]["adapter_paths"].items():
            if os.path.exists(adapter_path):
                self.get_logger().info(f"Loading adapter for task {task_name} from {adapter_path}")
                adapter_state = torch.load(adapter_path)
                self.adapters[task_name] = adapter_state
            else:
                raise FileNotFoundError(f"Adapter weights not found at {adapter_path}")

        # デフォルトのアダプターを読み込む
        if self.args.task in self.adapters:
            self.model.load_adapter(self.adapters[self.args.task])
            self.get_logger().info(f"Loaded adapter for task: {self.args.task}")
        else:
            raise ValueError(f"No adapter found for task: {self.args.task}")

        self.model.eval()

        num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
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
            obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1) 
            obs_images = obs_images.to(device)
            fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(device)

            with torch.no_grad():
                # アダプターモデルを使用して予測
                noisy_action = torch.randn(
                    (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=device)
                naction = noisy_action

                self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])

                start_time = time.time()
                for k in self.noise_scheduler.timesteps[:]:
                    noise_pred = self.model(obs_images, fake_goal, naction, k)
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

            if self.model_params["normalize"]:
                chosen_waypoint *= (MAX_V / RATE)
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint.tolist()
            self.waypoint_pub.publish(waypoint_msg)
            self.get_logger().info("Published waypoint")

    def switch_adapter(self, task_name: str):
        """タスクに応じてアダプターを切り替える"""
        if task_name in self.adapters:
            self.model.load_adapter(self.adapters[task_name])
            self.get_logger().info(f"Switched to adapter for task: {task_name}")
        else:
            self.get_logger().warning(f"No adapter found for task: {task_name}")

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(
        description="Code to run NoMaD Adapter EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad_adapter",
        type=str,
        help="model name (hint: check ../config/models.yaml)",
    )
    parser.add_argument(
        "--task",
        "-t",
        default="task1",
        type=str,
        help="task name for adapter selection",
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

    exploration = ExplorationAdapter(args)
    
    try:
        rclpy.spin(exploration)
    except KeyboardInterrupt:
        pass
    finally:
        exploration.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
