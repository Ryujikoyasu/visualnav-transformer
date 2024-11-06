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
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.joystick.init()
        
        # Joy-Con (L)を探す
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
            sys.exit(1)
            
        # 制御パラメータ
        self.linear_speed = 0.6  # 直進速度（固定）
        self.angular_speed_slow = -0.6  # L1押下時の回転速度
        self.angular_speed_fast = -0.9  # L2押下時の回転速度
        
        # ボタン設定
        self.slow_button = 15  # L1 button (低速回転)
        self.fast_button = 14  # L2 button (高速回転)
        
        # 状態管理
        self.current_hat = (0, 0)
        self.slow_enabled = False  # L1が押されているか
        self.fast_enabled = False  # L2が押されているか
        
        # タイマーの設定（10Hz）
        self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info("初期化完了")

    def timer_callback(self):
        # イベントの処理
        for event in pygame.event.get():
            if event.type == pygame.JOYHATMOTION:
                self.current_hat = event.value
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == self.slow_button:
                    self.slow_enabled = True
                elif event.button == self.fast_button:
                    self.fast_enabled = True
            elif event.type == pygame.JOYBUTTONUP:
                if event.button == self.slow_button:
                    self.slow_enabled = False
                elif event.button == self.fast_button:
                    self.fast_enabled = False
        
        # Twistメッセージの作成
        msg = Twist()
        x, y = self.current_hat
        
        # ボタンの状態に応じて角速度を設定
        if self.slow_enabled:
            angular_speed = self.angular_speed_slow
        elif self.fast_enabled:
            angular_speed = self.angular_speed_fast
        else:
            angular_speed = 0.0
        
        # 速度指令値の設定
        msg.linear.x = -x * self.linear_speed
        msg.angular.z = y * angular_speed
        
        # 速度指令値をパブリッシュ
        self.publisher.publish(msg)
        
        # 実際の値をログ出力（値が0でない場合のみ）
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.get_logger().info(
                f'HAT: {self.current_hat}, L1: {self.slow_enabled}, L2: {self.fast_enabled}, '
                f'Vel: ({msg.linear.x:.2f}, {msg.angular.z:.2f})'
            )

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
