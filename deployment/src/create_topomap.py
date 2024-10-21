import os
import sys
import numpy as np
import cv2

import argparse
from utils import msg_to_pil 
import time
from PIL import Image as PILImage

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
import shutil

IMAGE_TOPIC = "/image_raw"
TOPOMAP_IMAGES_DIR = "../topomaps/images"

class CreateTopomap(Node):
    def __init__(self, args):
        super().__init__("CREATE_TOPOMAP")
        self.obs_img = None
        self.args = args
        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self.callback_obs, 1)
        self.subgoals_pub = self.create_publisher(Image, "/subgoals", 1)
        self.joy_sub = self.create_subscription(Joy, "joy", self.callback_joy, 1)
        
        self.topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            print(f"{self.topomap_name_dir} already exists. Removing previous images...")
            self.remove_files_in_dir(self.topomap_name_dir)
        
        self.i = 0
        self.start_time = float("inf")
        self.timer = self.create_timer(args.dt, self.timer_callback)

    def remove_files_in_dir(self, dir_path: str):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    def callback_obs(self, msg: Image):
        # ROSのImage メッセージからnumpy配列に変換
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # BGRからRGBに変換
        rgb_img = cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB)
        self.obs_img = PILImage.fromarray(rgb_img)

    def callback_joy(self, msg: Joy):
        if msg.buttons[0]:
            self.get_logger().info("Shutdown requested")
            rclpy.shutdown()

    def timer_callback(self):
        if self.obs_img is not None:
            # 既にRGB形式なので、そのまま保存
            self.obs_img.save(os.path.join(self.topomap_name_dir, f"{self.i}.png"))
            print("published image", self.i)
            self.i += 1
            self.start_time = time.time()
            self.obs_img = None
        if time.time() - self.start_time > 2 * self.args.dt:
            print(f"Topic {IMAGE_TOPIC} not publishing anymore. Shutting down...")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 1.0)",
    )
    args = parser.parse_args()

    create_topomap = CreateTopomap(args)
    print("Registered with master node. Waiting for images...")
    
    try:
        rclpy.spin(create_topomap)
    except KeyboardInterrupt:
        pass
    finally:
        create_topomap.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
