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
        except pygame.error:
            self.get_logger().error('ジョイスティックが見つかりません')
            sys.exit(1)
            
        # 制御パラメータ
        self.linear_speed = 0.5  # 直進速度の最大値
        self.angular_speed = 1.0  # 回転速度の最大値
        
        # Joy-Con (L)の軸の設定
        self.STICK_HORIZONTAL = 0  # 左スティックの水平方向
        self.STICK_VERTICAL = 1    # 左スティックの垂直方向
        
        # デッドゾーンの設定
        self.deadzone = 0.15
        
        # タイマーの設定（20Hz）
        self.create_timer(0.05, self.timer_callback)
        
        # 前回の軸の値を保存
        self.prev_axes = [0.0] * self.joystick.get_numaxes()

    def apply_deadzone(self, value, deadzone):
        if abs(value) < deadzone:
            return 0.0
        return value

    def timer_callback(self):
        # イベントの処理
        pygame.event.pump()
        
        # 全ての軸の値をチェック
        for i in range(self.joystick.get_numaxes()):
            current_value = self.joystick.get_axis(i)
            # 値が変化した場合のみ出力
            if abs(current_value - self.prev_axes[i]) > 0.01:
                self.get_logger().info(f'軸 {i} の値: {current_value:.3f}')
                self.prev_axes[i] = current_value
        
        # Twistメッセージの作成
        msg = Twist()
        
        # スティックの値を取得
        horizontal = self.apply_deadzone(self.joystick.get_axis(self.STICK_HORIZONTAL), self.deadzone)
        vertical = self.apply_deadzone(self.joystick.get_axis(self.STICK_VERTICAL), self.deadzone)
        
        # 速度指令値の計算
        msg.linear.x = -vertical * self.linear_speed
        msg.angular.z = -horizontal * self.angular_speed
        
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
