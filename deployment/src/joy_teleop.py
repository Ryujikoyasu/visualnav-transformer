#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame
import sys
import os
import math
from time import time

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
        self.linear_speed = 0.5  # 直進速度の最大値
        self.angular_speed = 1.0  # 回転速度の最大値
        
        # 状態管理
        self.current_hat = (0, 0)  # 現在のHAT値
        self.target_vel = [0.0, 0.0]  # 目標速度 [linear.x, angular.z]
        self.current_vel = [0.0, 0.0]  # 現在の速度
        
        # 補間パラメータ
        self.interpolation_rate = 5.0  # 補間速度（値が大きいほど素早く変化）
        self.last_time = time()
        
        # タイマーの設定（60Hz）
        self.create_timer(1.0/60.0, self.timer_callback)
        
        self.get_logger().info("初期化完了")

    def interpolate_velocity(self, current, target, dt):
        """現在の速度から目標速度に向けて滑らかに補間"""
        diff = target - current
        step = self.interpolation_rate * diff * dt
        return current + step

    def timer_callback(self):
        # 時間差分の計算
        current_time = time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # イベントの処理
        for event in pygame.event.get():
            if event.type == pygame.JOYHATMOTION:
                self.current_hat = event.value
                # HAT値から目標速度を設定
                x, y = self.current_hat
                # 斜め入力の場合は正規化
                if x != 0 and y != 0:
                    norm = math.sqrt(2)
                    self.target_vel = [y/norm * self.linear_speed, -x/norm * self.angular_speed]
                else:
                    self.target_vel = [y * self.linear_speed, -x * self.angular_speed]
        
        # 速度の補間
        self.current_vel[0] = self.interpolate_velocity(
            self.current_vel[0], self.target_vel[0], dt)
        self.current_vel[1] = self.interpolate_velocity(
            self.current_vel[1], self.target_vel[1], dt)
        
        # Twistメッセージの作成と送信
        msg = Twist()
        msg.linear.x = self.current_vel[0]
        msg.angular.z = self.current_vel[1]
        self.publisher.publish(msg)
        
        # 実際の値をログ出力（値が0でない場合のみ）
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.get_logger().info(
                f'HAT: {self.current_hat}, '
                f'Target: ({self.target_vel[0]:.2f}, {self.target_vel[1]:.2f}), '
                f'Current: ({self.current_vel[0]:.2f}, {self.current_vel[1]:.2f})'
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
