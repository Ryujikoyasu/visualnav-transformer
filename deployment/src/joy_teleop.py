#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from evdev import InputDevice, categorize, ecodes, list_devices
import sys
import math
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
        
        # 利用可能なデバイスをすべて表示
        self.get_logger().info("利用可能なデバイス:")
        for path in list_devices():
            try:
                device = InputDevice(path)
                self.get_logger().info(f"  {path}: {device.name}")
            except Exception as e:
                self.get_logger().error(f"デバイス {path} のオープンに失敗: {e}")
        
        # Joy-Con (L)を探す
        self.device = None
        for path in list_devices():
            try:
                device = InputDevice(path)
                if "Joy-Con" in device.name:
                    self.device = device
                    self.get_logger().info(f'Joy-Con が見つかりました: {device.name} at {path}')
                    # デバイスの機能を表示
                    caps = device.capabilities(verbose=True)
                    self.get_logger().info("デバイスの機能:")
                    for key, value in caps.items():
                        self.get_logger().info(f"  {key}: {value}")
                    break
            except Exception as e:
                self.get_logger().error(f"デバイス {path} のオープンに失敗: {e}")
        
        if self.device is None:
            self.get_logger().error('Joy-Con が見つかりません')
            sys.exit(1)
            
        # デバイスのパーミッションをチェック
        device_path = self.device.path
        stat_info = os.stat(device_path)
        self.get_logger().info(f"デバイスのパーミッション: {oct(stat_info.st_mode)}")
        self.get_logger().info(f"デバイスの所有者: {stat_info.st_uid}")
        self.get_logger().info(f"現在のユーザーID: {os.getuid()}")
            
        # 制御パラメータ
        self.linear_speed = 0.5  # 直進速度の最大値
        self.angular_speed = 1.0  # 回転速度の最大値
        
        # スティックの状態
        self.stick_x = 0
        self.stick_y = 0
        
        # デッドゾーン
        self.deadzone = 500
        
        # タイマーの設定（20Hz）
        self.create_timer(0.05, self.timer_callback)
        
        self.get_logger().info("初期化完了")

    def normalize_stick(self, value):
        # スティックの値を-1から1の範囲に正規化
        MAX_VALUE = 2048
        normalized = value / MAX_VALUE
        
        # デッドゾーンの適用
        if abs(normalized) < self.deadzone / MAX_VALUE:
            return 0.0
        return normalized

    def timer_callback(self):
        try:
            # イベントの処理
            events = self.device.read()
            for event in events:
                self.get_logger().info(f'イベント検出: type={event.type}, code={event.code}, value={event.value}')
                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_X:  # 左スティック X軸
                        self.stick_x = event.value - 2048  # 中心を0に
                        self.get_logger().info(f'スティックX: {self.stick_x}')
                    elif event.code == ecodes.ABS_Y:  # 左スティック Y軸
                        self.stick_y = event.value - 2048  # 中心を0に
                        self.get_logger().info(f'スティックY: {self.stick_y}')
        except BlockingIOError:
            pass  # イベントがない場合は無視
        except Exception as e:
            self.get_logger().error(f'イベント読み取りエラー: {e}')
        
        # Twistメッセージの作成
        msg = Twist()
        
        # スティックの値を速度指令に変換
        msg.linear.x = -self.normalize_stick(self.stick_y) * self.linear_speed
        msg.angular.z = -self.normalize_stick(self.stick_x) * self.angular_speed
        
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

if __name__ == '__main__':
    main()
