import rclpy
from rclpy.node import Node

class ROSData:
    def __init__(self, timeout: float = 3.0, queue_size: int = 1, name: str = "", node: Node = None):
        self.timeout = timeout
        self.last_time_received = float("-inf")
        self.queue_size = queue_size
        self.data = None
        self.name = name
        self.phantom = False
        self.node = node
        
        if self.node is None:
            # 警告: ノードが提供されていない場合、時間の取得が正確でない可能性があります
            import warnings
            warnings.warn("No ROS 2 node provided to ROSData. Time calculations may be inaccurate.")
    
    def get(self):
        return self.data
    
    def set(self, data): 
        current_time = self.get_current_time()
        time_waited = current_time - self.last_time_received
        if self.queue_size == 1:
            self.data = data
        else:
            if self.data is None or time_waited > self.timeout: # reset queue if timeout
                self.data = []
            if len(self.data) == self.queue_size:
                self.data.pop(0)
            self.data.append(data)
        self.last_time_received = current_time
        
    def is_valid(self, verbose: bool = False):
        current_time = self.get_current_time()
        time_waited = current_time - self.last_time_received
        valid = time_waited < self.timeout
        if self.queue_size > 1:
            valid = valid and len(self.data) == self.queue_size
        if verbose and not valid:
            print(f"Not receiving {self.name} data for {time_waited} seconds (timeout: {self.timeout} seconds)")
        return valid

    def get_current_time(self):
        if self.node:
            return self.node.get_clock().now().seconds_nanoseconds()[0]
        else:
            # フォールバック: システム時間を使用（正確でない可能性があります）
            import time
            return time.time()