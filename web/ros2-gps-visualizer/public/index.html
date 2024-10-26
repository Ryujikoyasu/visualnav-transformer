import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Icon } from 'leaflet';

// マーカーアイコンの設定
const currentPositionIcon = new Icon({
  iconUrl: '/api/placeholder/32/32',
  iconSize: [32, 32],
});

const goalPositionIcon = new Icon({
  iconUrl: '/api/placeholder/32/32',
  iconSize: [32, 32],
});

const MapComponent = () => {
  const [currentPosition, setCurrentPosition] = useState(null);
  const [goalPosition, setGoalPosition] = useState(null);
  const [websocket, setWebsocket] = useState(null);

  // WebSocket接続の確立
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
    
    ws.onopen = () => {
      console.log('Connected to ROS2 bridge');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.current) {
        setCurrentPosition([data.current.latitude, data.current.longitude]);
      }
      if (data.goal) {
        setGoalPosition([data.goal.latitude, data.goal.longitude]);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setWebsocket(ws);

    return () => {
      ws.close();
    };
  }, []);

  // マップの中心を現在位置に自動調整するコンポーネント
  const MapUpdater = () => {
    const map = useMap();
    
    useEffect(() => {
      if (currentPosition) {
        map.setView(currentPosition, map.getZoom());
      }
    }, [currentPosition]);
    
    return null;
  };

  return (
    <div className="h-screen w-full">
      <MapContainer
        center={[35.6812, 139.7671]} // デフォルト位置（東京）
        zoom={15}
        className="h-full w-full"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        
        <MapUpdater />
        
        {currentPosition && (
          <Marker position={currentPosition} icon={currentPositionIcon}>
            <Popup>
              Current Position<br />
              Lat: {currentPosition[0]}<br />
              Lng: {currentPosition[1]}
            </Popup>
          </Marker>
        )}
        
        {goalPosition && (
          <Marker position={goalPosition} icon={goalPositionIcon}>
            <Popup>
              Goal Position<br />
              Lat: {goalPosition[0]}<br />
              Lng: {goalPosition[1]}
            </Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
};

export default MapComponent;