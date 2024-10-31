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
        
        # タイマーの設定（20Hz）
        self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        # イベントの処理
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                self.get_logger().info(f'ボタン {event.button} が押されました')
            elif event.type == pygame.JOYBUTTONUP:
                self.get_logger().info(f'ボタン {event.button} が離されました')
            elif event.type == pygame.JOYAXISMOTION:
                self.get_logger().info(f'軸 {event.axis} の値: {event.value}')
            elif event.type == pygame.JOYHATMOTION:
                self.get_logger().info(f'HAT {event.hat} の値: {event.value}')
        
        # Twistメッセージの作成
        msg = Twist()
        
        # 全ての軸の値を確認
        for i in range(self.joystick.get_numaxes()):
            axis_value = self.joystick.get_axis(i)
            if abs(axis_value) > 0.1:
                self.get_logger().info(f'軸 {i} の現在値: {axis_value}')

        # 全てのボタンの状態を確認
        for i in range(self.joystick.get_numbuttons()):
            if self.joystick.get_button(i):
                self.get_logger().info(f'ボタン {i} が押されています')

        # 全てのHATの状態を確認
        for i in range(self.joystick.get_numhats()):
            hat_value = self.joystick.get_hat(i)
            if hat_value != (0, 0):
                self.get_logger().info(f'HAT {i} の値: {hat_value}')
                # HATの値を速度指令に変換
                msg.linear.x = hat_value[1] * self.linear_speed  # 上下方向
                msg.angular.z = -hat_value[0] * self.angular_speed  # 左右方向
        
        # 速度指令値をパブリッシュ
        self.publisher.publish(msg)
        
        # 実際にパブリッシュされた値をログ出力
        if abs(msg.linear.x) > 0.0 or abs(msg.angular.z) > 0.0:
            self.get_logger().info(f'パブリッシュ: linear.x = {msg.linear.x}, angular.z = {msg.angular.z}')

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
