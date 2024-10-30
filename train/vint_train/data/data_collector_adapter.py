import argparse
import os
import rclpy
import cv2
from datetime import datetime

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # コマンドライン引数でも指定できるように
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_dir', type=str, 
                           default='/ssd/source/navigation/visualnav-transformer/train/vint_train/data/nomad_adapter_dataset/raw_data',
                           help='データの保存先ディレクトリ')
        args, _ = parser.parse_known_args()
        
        # パラメータの設定
        self.save_dir = self.declare_parameter(
            'save_dir', 
            os.path.expanduser(args.save_dir)
        ).value
        
        # ディレクトリの作成
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'twists'), exist_ok=True)
        
        self.cv_bridge = CvBridge()
        self.frame_count = 0
        
        # Subscriberの設定
        self.image_sub = Subscriber(self, Image, '/image_raw')
        self.twist_sub = Subscriber(self, Twist, '/cmd_vel_mux/input/teleop')
        
        # 同期設定
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.twist_sub],
            queue_size=10,
            slop=0.1,
            allow_headerless=True
        )
        self.sync.registerCallback(self.sync_callback)
        
        self.get_logger().info('Data collector initialized')

    def sync_callback(self, image_msg, twist_msg):
        try:
            # 画像の保存
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            image_filename = os.path.join(self.save_dir, 'images', f'{timestamp}.jpg')
            cv2.imwrite(image_filename, cv_image)
            
            # Twistデータの保存
            twist_filename = os.path.join(self.save_dir, 'twists', f'{timestamp}.txt')
            with open(twist_filename, 'w') as f:
                f.write(f'{twist_msg.linear.x},{twist_msg.linear.y},{twist_msg.linear.z},'
                       f'{twist_msg.angular.x},{twist_msg.angular.y},{twist_msg.angular.z}')
            
            self.frame_count += 1
            if self.frame_count % 10 == 0:  # 10フレームごとにログを出力
                self.get_logger().info(f'Saved {self.frame_count} frames')
                
        except Exception as e:
            self.get_logger().error(f'Error in sync_callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
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