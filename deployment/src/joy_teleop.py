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
            
            # 全ての軸の初期値を表示
            for i in range(self.joystick.get_numaxes()):
                self.get_logger().info(f'軸 {i} の初期値: {self.joystick.get_axis(i)}')
        except pygame.error:
            self.get_logger().error('ジョイスティックが見つかりません')
            sys.exit(1)
            
        # 制御パラメータ
        self.linear_speed = 0.5  # 直進速度の最大値
        self.angular_speed = 1.0  # 回転速度の最大値
        
        # デッドゾーンの設定
        self.deadzone = 0.15
        
        # タイマーの設定（20Hz）
        self.create_timer(0.05, self.timer_callback)

    def apply_deadzone(self, value, deadzone):
        if abs(value) < deadzone:
            return 0.0
        return value

    def timer_callback(self):
        # イベントの処理
        for event in pygame.event.get():
            # 全てのイベントタイプを出力
            self.get_logger().info(f'イベントタイプ: {event.type}')
            
            if event.type == pygame.JOYAXISMOTION:
                self.get_logger().info(f'軸の動き - 軸: {event.axis}, 値: {event.value:.3f}')
            elif event.type == pygame.JOYBUTTONDOWN:
                self.get_logger().info(f'ボタン押下 - ボタン: {event.button}')
            elif event.type == pygame.JOYBUTTONUP:
                self.get_logger().info(f'ボタン解放 - ボタン: {event.button}')
            elif event.type == pygame.JOYHATMOTION:
                self.get_logger().info(f'HAT変化 - HAT: {event.hat}, 値: {event.value}')
            elif event.type == pygame.JOYDEVICEADDED:
                self.get_logger().info('ジョイスティックが接続されました')
            elif event.type == pygame.JOYDEVICEREMOVED:
                self.get_logger().info('ジョイスティックが切断されました')
        
        # 現在の全ての軸の値を取得
        axes_values = [self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())]
        # 値が0でない軸のみ表示
        for i, value in enumerate(axes_values):
            if abs(value) > self.deadzone:
                self.get_logger().info(f'軸 {i} の現在値: {value:.3f}')
        
        # Twistメッセージの作成
        msg = Twist()
        
        # 全ての軸を試してみる
        for i in range(self.joystick.get_numaxes()):
            value = self.apply_deadzone(self.joystick.get_axis(i), self.deadzone)
            if abs(value) > self.deadzone:
                # 各軸の値を試験的に使用
                if i in [0, 2]:  # 水平方向の可能性がある軸
                    msg.angular.z = -value * self.angular_speed
                elif i in [1, 3]:  # 垂直方向の可能性がある軸
                    msg.linear.x = -value * self.linear_speed
        
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
