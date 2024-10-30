import argparse
import os

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer
from rclpy.subscription import Subscriber

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # コマンドライン引数でも指定できるように
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_dir', type=str, 
                           default='~/nomad_dataset/raw_data',
                           help='データの保存先ディレクトリ')
        args, _ = parser.parse_known_args()
        
        # パラメータの設定
        self.save_dir = self.declare_parameter(
            'save_dir', 
            os.path.expanduser(args.save_dir)
        ).value
        
        # ディレクトリの作成
        os.makedirs(self.save_dir, exist_ok=True) 
        
        self.cv_bridge = CvBridge()
        
        # Subscriberの設定
        self.image_sub = Subscriber(self, Image, '/img_raw')
        self.twist_sub = Subscriber(self, Twist, '/cmd_vel_mux/input/teleop')
        
        # 同期設定 - allow_headerlessをTrueに設定
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.twist_sub],
            queue_size=10,
            slop=0.1,
            allow_headerless=True
        )
        self.sync.registerCallback(self.sync_callback)