#!/usr/bin/env python3

# Adapterのためのデータセット作成．画像とTwist.msgを同期して保存．

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import os
import time
from message_filters import ApproximateTimeSynchronizer, Subscriber

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # 保存先ディレクトリの設定
        self.save_dir = self.declare_parameter('save_dir', 
            os.path.expanduser('~/nomad_dataset/raw_data')).value
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.cv_bridge = CvBridge()
        
        # Subscriberの設定
        self.image_sub = Subscriber(self, Image, '/img_raw')
        self.twist_sub = Subscriber(self, Twist, '/cmd_vel_mux/input/teleop')
        
        # 同期設定（許容時間差0.1秒）
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.twist_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)
        
        self.get_logger().info('Data collector started')
        
    def sync_callback(self, img_msg, twist_msg):
        timestamp = int(time.time() * 1000)  # ミリ秒単位のタイムスタンプ
        
        # 画像の保存
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
            img_path = os.path.join(self.save_dir, f'{timestamp}_image.jpg')
            cv2.imwrite(img_path, cv_image)
            
            # Twistメッセージの保存
            twist_path = os.path.join(self.save_dir, f'{timestamp}_twist.txt')
            with open(twist_path, 'w') as f:
                f.write(f'{twist_msg.linear.x},{twist_msg.linear.y},{twist_msg.linear.z},'
                       f'{twist_msg.angular.x},{twist_msg.angular.y},{twist_msg.angular.z}')
                
        except Exception as e:
            self.get_logger().error(f'Error in callback: {e}')

def main():
    rclpy.init()
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 