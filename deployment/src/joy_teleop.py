#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame
import sys
import os

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
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # ディスプレイが不要なことを指定
        pygame.init()
        pygame.joystick.init()
        
        # 利用可能なジョイスティックの表示
        self.get_logger().info(f'検出されたジョイスティック数: {pygame.joystick.get_count()}')
        
        # Joy-Con (L)を探す
        self.joystick = None
        for i in range(pygame.joystick.get_count()):
            joy = pygame.joystick.Joystick(i)
            joy.init()
            self.get_logger().info(f'ジョイスティック {i}: {joy.get_name()}')
            if "Joy-Con" in joy.get_name():
                self.joystick = joy
                self.get_logger().info(f'Joy-Con が見つかりました: {joy.get_name()}')
                self.get_logger().info(f'軸の数: {joy.get_numaxes()}')
                break
        
        if self.joystick is None:
            self.get_logger().error('Joy-Con が見つかりません')
            sys.exit(1)
            
        # 制御パラメータ
        self.linear_speed = 0.5  # 直進速度の最大値
        self.angular_speed = 1.0  # 回転速度の最大値
        
        # スティックの状態
        self.stick_x = 0.0
        self.stick_y = 0.0
        
        # デッドゾーン
        self.deadzone = 0.15
        
        # タイマーの設定（60Hz）
        self.create_timer(1.0/60.0, self.timer_callback)
        
        self.get_logger().info("初期化完了")

    def apply_deadzone(self, value, deadzone):
        if abs(value) < deadzone:
            return 0.0
        return value

    def timer_callback(self):
        # イベントの処理
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.get_logger().info(f'軸の動き: axis={event.axis}, value={event.value:.3f}')
                if event.axis == 0:  # X軸
                    self.stick_x = event.value
                elif event.axis == 1:  # Y軸
                    self.stick_y = event.value
        
        # 現在の軸の値を直接読み取り
        try:
            self.stick_x = self.joystick.get_axis(0)
            self.stick_y = self.joystick.get_axis(1)
        except pygame.error as e:
            self.get_logger().error(f'ジョイスティック読み取りエラー: {e}')
            return
        
        # デッドゾーンの適用
        self.stick_x = self.apply_deadzone(self.stick_x, self.deadzone)
        self.stick_y = self.apply_deadzone(self.stick_y, self.deadzone)
        
        # Twistメッセージの作成
        msg = Twist()
        msg.linear.x = -self.stick_y * self.linear_speed
        msg.angular.z = -self.stick_x * self.angular_speed
        
        # 速度指令値をパブリッシュ
        self.publisher.publish(msg)
        
        # 実際の値をログ出力（値が0でない場合のみ）
        if abs(self.stick_x) > self.deadzone or abs(self.stick_y) > self.deadzone:
            self.get_logger().info(f'スティック: X={self.stick_x:.3f}, Y={self.stick_y:.3f}')
            self.get_logger().info(f'パブリッシュ: linear.x={msg.linear.x:.3f}, angular.z={msg.angular.z:.3f}')

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
