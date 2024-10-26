let ws = null;

export const connectWebSocket = (onMessage) => {
  ws = new WebSocket('ws://localhost:8765');
  
  ws.onopen = () => {
    console.log('Connected to ROS2 bridge');
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  return () => {
    if (ws) {
      ws.close();
    }
  };
};

export const sendMessage = (message) => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
  }
};
