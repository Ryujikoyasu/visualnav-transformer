# ros2_gps_map/ros2_gps_map/gps_bridge_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
import json
import websockets
import asyncio

class GPSBridgeNode(Node):
    # クラス変数として定義
    clients = set()
    current_position = None
    goal_position = None

    def __init__(self):
        super().__init__('gps_bridge_node')
        self.gps_subscription = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # WebSocketサーバーの設定
        self.websocket_server = None
        self.create_timer(1.0, self.start_websocket_server)

    async def start_websocket_server(self):
        if self.websocket_server is None:
            self.websocket_server = await websockets.serve(self.ws_handler, "localhost", 8765)
            self.get_logger().info("WebSocket server started")

    async def ws_handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            while True:
                await websocket.recv()
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
        
        asyncio.create_task(self.send_to_all(json.dumps(data)))

    async def send_to_all(self, message):
        if self.clients:
            await asyncio.gather(*[client.send(message) for client in self.clients])

async def main(args=None):
    rclpy.init(args=args)
    node = GPSBridgeNode()
    
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        await asyncio.sleep(0.1)
    
    rclpy.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
