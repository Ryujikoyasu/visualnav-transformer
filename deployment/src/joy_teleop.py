#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import hid
import sys

class JoyTeleop(Node):
    def __init__(self):
        super().__init__('joy_teleop')
        self.publisher = self.create_publisher(Twist, '/cmd_vel_mux/input/teleop', 10)
        self.linear_speed = 0.5
        self.angular_speed = 1.0
        self.deadzone = 0.15
        self.get_logger().info("初期化完了")

        # Joy-Con (L) の情報を指定
        self.vendor_id = 0x057E  # Nintendo
        self.product_id = 0x2006  # Joy-Con (L)

        try:
            self.device = hid.device()
            self.device.open(self.vendor_id, self.product_id)
            self.device.set_nonblocking(1)
            self.get_logger().info("Joy-Con (L) が接続されました")
        except IOError:
            self.get_logger().error("Joy-Con (L) を接続できません")
            sys.exit(1)

        self.timer = self.create_timer(0.05, self.timer_callback)

    def apply_deadzone(self, value, deadzone):
        if abs(value) < deadzone:
            return 0.0
        return value

    def timer_callback(self):
        try:
            reports = self.device.read(64)
            if reports:
                # Joy-Con の入力フォーマットに基づいて解析
                # 詳細な解析にはJoy-Conのプロトコル理解が必要
                # ここでは簡単に軸の値を取得する例を示します
                # 具体的なバイトオフセットはJoy-Conの仕様によります
                ax = reports[1] - 128  # 仮のオフセット
                ay = reports[2] - 128  # 仮のオフセット

                stick_x = self.apply_deadzone(ax / 128.0, self.deadzone)
                stick_y = self.apply_deadzone(ay / 128.0, self.deadzone)

                msg = Twist()
                msg.linear.x = -stick_y * self.linear_speed
                msg.angular.z = -stick_x * self.angular_speed

                self.publisher.publish(msg)

                if abs(msg.linear.x) > 0.0 or abs(msg.angular.z) > 0.0:
                    self.get_logger().info(f'パブリッシュ: linear.x={msg.linear.x:.3f}, angular.z={msg.angular.z:.3f}')
        except Exception as e:
            self.get_logger().error(f'入力読み取りエラー: {e}')

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

if __name__ == '__main__':
    main()
