#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame
import sys

class JoyTeleop(Node):
    def __init__(self):
        super().__init__('joy_teleop')
        
        # パブリッシャーの設定
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel_mux/input/teleop',
            10
        )
        
        # Pygameの初期化
        pygame.init()
        pygame.joystick.init()
        
        # ジョイスティックの接続確認
        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.get_logger().info('ジョイスティックが接続されました')
        except pygame.error:
            self.get_logger().error('ジョイスティックが見つかりません')
            sys.exit(1)
            
        # 制御パラメータ
        self.linear_speed = 0.5  # 直進速度の最大値
        self.angular_speed = 1.0  # 回転速度の最大値
        
        # タイマーの設定（20Hz）
        self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        # イベントの処理
        pygame.event.pump()
        
        # Twistメッセージの作成
        msg = Twist()
        
        # 左スティックの値を取得
        msg.linear.x = -self.joystick.get_axis(1) * self.linear_speed  # 前後移動
        msg.angular.z = -self.joystick.get_axis(0) * self.angular_speed  # 回転
        
        # 速度指令値をパブリッシュ
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = JoyTeleop()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()
