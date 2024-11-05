import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import socket
import pickle
import struct

class ImageReceiverNode(Node):
    def __init__(self):
        super().__init__('image_receiver_node')
        
        # パブリッシャーの設定
        self.publisher = self.create_publisher(Image, 'image_raw', 10)
        
        # OpenCV-ROS bridge
        self.bridge = CvBridge()
        
        # ソケット通信の設定
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_ip = '0.0.0.0'
        port = 9999
        socket_address = (host_ip, port)
        self.server_socket.bind(socket_address)
        self.server_socket.listen(5)
        
        self.get_logger().info("Waiting for connection...")
        self.client_socket, self.addr = self.server_socket.accept()
        self.get_logger().info(f"Connected to: {self.addr}")
        
        # 受信用のデータバッファ
        self.data = b""
        self.payload_size = struct.calcsize("L")
        
        # タイマーコールバックの設定
        self.create_timer(0.01, self.timer_callback)  # 100Hz for smooth operation

    def timer_callback(self):
        try:
            # ペイロードサイズを受信
            while len(self.data) < self.payload_size:
                self.data += self.client_socket.recv(4096)
            
            packed_msg_size = self.data[:self.payload_size]
            self.data = self.data[self.payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            
            # 画像データを受信
            while len(self.data) < msg_size:
                self.data += self.client_socket.recv(4096)
            
            frame_data = self.data[:msg_size]
            self.data = self.data[msg_size:]
            
            # 画像データをデシリアライズ
            frame = pickle.loads(frame_data)
            
            # ROSメッセージに変換してパブリッシュ
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")

    def __del__(self):
        self.client_socket.close()
        self.server_socket.close()

def main(args=None):
    rclpy.init(args=args)
    node = ImageReceiverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()