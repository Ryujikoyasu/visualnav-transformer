import argparse
import os
import rclpy
import cv2
import pygame
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
                           default='/ssd/source/navigation/asset/nomad_adapter_dataset/raw_data',
                           help='データの保存先ディレクトリ')
        args, _ = parser.parse_known_args()
        
        # ベースディレクトリを保存
        self.base_save_dir = args.save_dir
        
        # 現在時刻をディレクトリ名に含める
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(self.base_save_dir, f'traj_{timestamp}')
        
        # ディレクトリの作成
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'twists'), exist_ok=True)
        
        self.cv_bridge = CvBridge()
        self.frame_count = 0
        
        # データ収集の状態管理を追加
        self.last_save_time = None
        self.min_save_interval = 1.0 / 4.0  # 4Hzでデータを保存
        
        # Subscriberの設定
        self.image_sub = Subscriber(self, Image, '/image_raw')
        self.twist_sub = Subscriber(self, Twist, '/cmd_vel_mux/input/teleop')
        
        # 同期設定（緩めの設定に変更）
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.twist_sub],
            queue_size=5,
            slop=0.1,  # 100msまでの時間差を許容
            allow_headerless=True
        )
        self.sync.registerCallback(self.sync_callback)
        
        self.get_logger().info(f'Data collector initialized. Saving to {self.save_dir}')
        
        # Pygameの初期化
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.joystick.init()
        
        # Joy-Conの初期化
        self.joystick = None
        for i in range(pygame.joystick.get_count()):
            joy = pygame.joystick.Joystick(i)
            joy.init()
            if "Joy-Con" in joy.get_name():
                self.joystick = joy
                self.get_logger().info(f'Joy-Con が見つかりました: {joy.get_name()}')
                break
        
        if self.joystick is None:
            self.get_logger().error('Joy-Con が見つかりません')
            raise RuntimeError('Joy-Con が見つかりません')

        # ボタン設定
        self.button_a = 1
        self.button_b = 2
        
        # 状態管理
        self.is_recording = False
        self.last_button_state = None  # チャタリング防止用
        
        # タイマーの追加（ボタン状態チェック用）
        self.create_timer(0.1, self.check_buttons)
        
    def check_buttons(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                current_state = 'A' if event.button == self.button_a else 'B' if event.button == self.button_b else None
                
                if current_state and current_state != self.last_button_state:
                    if current_state == 'A' and self.last_button_state == 'B':
                        self.stop_recording()
                    elif current_state == 'B' and self.last_button_state == 'A':
                        self.start_recording()
                    
                    self.last_button_state = current_state

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.get_logger().info('データの記録を開始します')
            
            # 新しいディレクトリの作成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_dir = os.path.join(self.base_save_dir, f'traj_{timestamp}')
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'twists'), exist_ok=True)
            
            self.frame_count = 0
            self.last_save_time = None

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.get_logger().info(f'データの記録を終了します。保存フレーム数: {self.frame_count}')

    def sync_callback(self, image_msg, twist_msg):
        if not self.is_recording:
            return
            
        try:
            current_time = datetime.now()
            
            # 前回の保存から十分な時間が経過していない場合はスキップ
            if self.last_save_time is not None:
                time_diff = (current_time - self.last_save_time).total_seconds()
                if time_diff < self.min_save_interval:
                    return
            
            # 画像の保存
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            timestamp = current_time.strftime('%Y%m%d_%H%M%S_%f')
            image_filename = os.path.join(self.save_dir, 'images', f'{timestamp}.jpg')
            cv2.imwrite(image_filename, cv_image)
            
            # Twistデータの保存
            twist_filename = os.path.join(self.save_dir, 'twists', f'{timestamp}.txt')
            with open(twist_filename, 'w') as f:
                f.write(f'{twist_msg.linear.x},{twist_msg.linear.y},{twist_msg.linear.z},'
                       f'{twist_msg.angular.x},{twist_msg.angular.y},{twist_msg.angular.z}')
            
            self.last_save_time = current_time
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
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()