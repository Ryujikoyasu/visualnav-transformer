import React, { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { connectWebSocket } from '../services/websocketService';

// マーカーアイコンの設定
const currentPositionIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const goalPositionIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const MapComponent = () => {
  const [currentPosition, setCurrentPosition] = useState(null);
  const [goalPosition, setGoalPosition] = useState(null);

  const handleWebSocketMessage = useCallback((data) => {
    if (data.current) {
      setCurrentPosition([data.current.latitude, data.current.longitude]);
    }
    if (data.goal) {
      setGoalPosition([data.goal.latitude, data.goal.longitude]);
    }
  }, []);

  useEffect(() => {
    const closeWebSocket = connectWebSocket(handleWebSocketMessage);
    return closeWebSocket;
  }, [handleWebSocketMessage]);

  // マップの中心を現在位置に自動調整するコンポーネント
  const MapUpdater = () => {
    const map = useMap();
    
    useEffect(() => {
      if (currentPosition) {
        map.setView(currentPosition, map.getZoom());
      }
    }, [currentPosition, map]);
    
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
          url="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
          maxZoom={20}
          subdomains={['mt0', 'mt1', 'mt2', 'mt3']}
        />
        
        <MapUpdater />
        
        {currentPosition && (
          <Marker position={currentPosition} icon={currentPositionIcon}>
            <Popup>
              Current Position<br />
              Lat: {currentPosition[0].toFixed(6)}<br />
              Lng: {currentPosition[1].toFixed(6)}
            </Popup>
          </Marker>
        )}
        
        {goalPosition && (
          <Marker position={goalPosition} icon={goalPositionIcon}>
            <Popup>
              Goal Position<br />
              Lat: {goalPosition[0].toFixed(6)}<br />
              Lng: {goalPosition[1].toFixed(6)}
            </Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
};

export default MapComponent;
