# ros2_gps_map/ros2_gps_map/gps_bridge_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
import json
import websockets
import asyncio
import threading

class GPSBridgeNode(Node):
    def __init__(self):
        super().__init__('gps_bridge_node')
        
        # クラス変数の初期化
        self.clients = set()
        self.current_position = None
        self.goal_position = None
        
        # サブスクリプションの設定
        self.gps_subscription = self.create_subscription(
            NavSatFix,
            '/gps/fix',
            self.gps_callback,
            10
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # WebSocketサーバーの起動
        self.server = None
        threading.Thread(target=self.run_websocket_server, daemon=True).start()
        
    async def start_server(self):
        self.server = await websockets.serve(self.ws_handler, "localhost", 8765)
        await self.server.wait_closed()
    
    def run_websocket_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_server())
        loop.run_forever()

    async def ws_handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            while True:
                await websocket.recv()  # クライアントからのメッセージを待機
        except websockets.exceptions.ConnectionClosed:
            self.clients.remove(websocket)

    def gps_callback(self, msg):
        self.current_position = {
            'type': 'current_position',
            'latitude': msg.latitude,
            'longitude': msg.longitude
        }
        self.broadcast_position()

    def goal_callback(self, msg):
        # 注: このコールバックは簡略化されています。実際のアプリケーションでは
        # 座標変換が必要かもしれません
        self.goal_position = {
            'type': 'goal_position',
            'latitude': msg.pose.position.x,  # 適切な変換が必要
            'longitude': msg.pose.position.y  # 適切な変換が必要
        }
        self.broadcast_position()

    def broadcast_position(self):
        if not self.clients:
            return
            
        data = {
            'current': self.current_position,
            'goal': self.goal_position
        }
        
        asyncio.run(self.send_to_all(json.dumps(data)))

    async def send_to_all(self, message):
        if self.clients:
            await asyncio.wait([
                client.send(message)
                for client in self.clients
            ])

def main(args=None):
    rclpy.init(args=args)
    node = GPSBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

        
if __name__ == '__main__':
    main()