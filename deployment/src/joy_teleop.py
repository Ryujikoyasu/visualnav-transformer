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
            
            # ジョイスティックの情報を表示
            self.get_logger().info(f'ジョイスティック名: {self.joystick.get_name()}')
            self.get_logger().info(f'軸の数: {self.joystick.get_numaxes()}')
            self.get_logger().info(f'ボタンの数: {self.joystick.get_numbuttons()}')
            self.get_logger().info(f'HAT の数: {self.joystick.get_numhats()}')
        except pygame.error:
            self.get_logger().error('ジョイスティックが見つかりません')
            sys.exit(1)
            
        # 制御パラメータ
        self.linear_speed = 0.5  # 直進速度の最大値
        self.angular_speed = 1.0  # 回転速度の最大値
        
        # 現在のHAT値を保存
        self.current_hat = (0, 0)
        
        # タイマーの設定（20Hz）
        self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        # イベントの処理
        for event in pygame.event.get():
            if event.type == pygame.JOYHATMOTION:
                self.current_hat = event.value
                self.get_logger().info(f'HAT値: {event.value}')
        
        # Twistメッセージの作成
        msg = Twist()
        
        # HATの値を速度指令に変換
        # current_hat は (x, y) のタプルで、
        # x: -1（左）、0（中立）、1（右）
        # y: -1（下）、0（中立）、1（上）
        msg.linear.x = float(self.current_hat[1]) * self.linear_speed
        msg.angular.z = -float(self.current_hat[0]) * self.angular_speed
        
        # 速度指令値をパブリッシュ
        self.publisher.publish(msg)
        
        # 実際にパブリッシュされた値をログ出力（値が0でない場合のみ）
        if abs(msg.linear.x) > 0.0 or abs(msg.angular.z) > 0.0:
            self.get_logger().info(f'パブリッシュ: linear.x = {msg.linear.x:.2f}, angular.z = {msg.angular.z:.2f}')

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
