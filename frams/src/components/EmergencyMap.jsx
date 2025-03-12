import { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import { Paper, Title, Text, Badge, Group, useMantineColorScheme } from '@mantine/core';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import axios from 'axios';

// Leaflet icon sorunu için çözüm
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

// Özel ikon oluşturma
const createCustomIcon = (color) => {
  return new L.Icon({
    iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-${color}.png`,
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
  });
};

const depotIcon = createCustomIcon('green');
const fireIcon = createCustomIcon('red');

// OSRM ile rota geometrisi alma
const getOsrmRouteGeometry = async (startLat, startLon, endLat, endLon) => {
  try {
    const response = await axios.get(
      `https://router.project-osrm.org/route/v1/driving/${startLon},${startLat};${endLon},${endLat}?overview=full&geometries=geojson`
    );
    
    if (response.data && response.data.routes && response.data.routes.length > 0) {
      const coordinates = response.data.routes[0].geometry.coordinates;
      // OSRM [lon, lat] formatında döndürür, Leaflet [lat, lon] bekler
      return coordinates.map(coord => [coord[1], coord[0]]);
    }
    return [[startLat, startLon], [endLat, endLon]]; // Fallback
  } catch (error) {
    console.error('OSRM rota hatası:', error);
    return [[startLat, startLon], [endLat, endLon]]; // Fallback
  }
};

// Harita sınırlarını güncelleyen bileşen
const MapBoundsUpdater = ({ bounds }) => {
  const map = useMap();
  
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds);
    }
  }, [bounds, map]);
  
  return null;
};

const EmergencyMap = ({ emergencies, routes, depots }) => {
  const { colorScheme } = useMantineColorScheme();
  const isDark = colorScheme === 'dark';
  
  const [center, setCenter] = useState([37.7640, 30.5458]); // Varsayılan merkez
  const [mapBounds, setMapBounds] = useState(null);
  const [routeGeometries, setRouteGeometries] = useState({});
  const mapRef = useRef(null);

  useEffect(() => {
    // Harita sınırlarını hesapla
    if (emergencies.length > 0 || depots.length > 0) {
      const allPoints = [
        ...emergencies.map(e => [parseFloat(e.latitude), parseFloat(e.longitude)]),
        ...depots.map(d => [d.latitude, d.longitude])
      ];
      
      if (allPoints.length > 0) {
        const bounds = L.latLngBounds(allPoints);
        setMapBounds(bounds);
        
        // Merkezi güncelle
        const centerLat = (bounds.getNorth() + bounds.getSouth()) / 2;
        const centerLng = (bounds.getEast() + bounds.getWest()) / 2;
        setCenter([centerLat, centerLng]);
      }
    }
  }, [emergencies, depots]);

  // Rotalar değiştiğinde rota geometrilerini hesapla
  useEffect(() => {
    const fetchRouteGeometries = async () => {
      const geometries = {};
      
      for (const routeId in routes) {
        const route = routes[routeId];
        if (!route.emergencyIds || route.emergencyIds.length === 0) continue;
        
        const depot = depots.find(d => d.id === route.depotId);
        if (!depot) continue;
        
        // Araç tipi kara aracı ise OSRM rotası al, değilse düz çizgi kullan
        if (route.vehicleType === 'ground') {
          let currentLat = depot.latitude;
          let currentLon = depot.longitude;
          const routeSegments = [];
          
          // Her acil durum noktası için rota al
          for (const emergencyId of route.emergencyIds) {
            const emergency = emergencies.find(e => e.id === emergencyId);
            if (!emergency) continue;
            
            const emergencyLat = parseFloat(emergency.latitude);
            const emergencyLon = parseFloat(emergency.longitude);
            
            try {
              // OSRM ile rota geometrisi al
              const segmentGeometry = await getOsrmRouteGeometry(
                currentLat, currentLon, emergencyLat, emergencyLon
              );
              
              routeSegments.push({
                points: segmentGeometry,
                from: { lat: currentLat, lon: currentLon },
                to: { lat: emergencyLat, lon: emergencyLon },
                type: 'segment'
              });
              
              currentLat = emergencyLat;
              currentLon = emergencyLon;
            } catch (error) {
              console.error('Rota segmenti alınamadı:', error);
            }
          }
          
          // Depoya dönüş rotası
          try {
            const returnGeometry = await getOsrmRouteGeometry(
              currentLat, currentLon, depot.latitude, depot.longitude
            );
            
            routeSegments.push({
              points: returnGeometry,
              from: { lat: currentLat, lon: currentLon },
              to: { lat: depot.latitude, lon: depot.longitude },
              type: 'return'
            });
          } catch (error) {
            console.error('Dönüş rotası alınamadı:', error);
          }
          
          geometries[routeId] = {
            segments: routeSegments,
            vehicleType: route.vehicleType,
            vehicleId: route.vehicleId
          };
        } else {
          // Hava aracı için düz çizgiler (kuş uçuşu)
          const routeSegments = [];
          let currentLat = depot.latitude;
          let currentLon = depot.longitude;
          
          // Her acil durum noktasına düz çizgi
          for (const emergencyId of route.emergencyIds) {
            const emergency = emergencies.find(e => e.id === emergencyId);
            if (!emergency) continue;
            
            const emergencyLat = parseFloat(emergency.latitude);
            const emergencyLon = parseFloat(emergency.longitude);
            
            routeSegments.push({
              points: [[currentLat, currentLon], [emergencyLat, emergencyLon]],
              from: { lat: currentLat, lon: currentLon },
              to: { lat: emergencyLat, lon: emergencyLon },
              type: 'segment'
            });
            
            currentLat = emergencyLat;
            currentLon = emergencyLon;
          }
          
          // Depoya dönüş
          routeSegments.push({
            points: [[currentLat, currentLon], [depot.latitude, depot.longitude]],
            from: { lat: currentLat, lon: currentLon },
            to: { lat: depot.latitude, lon: depot.longitude },
            type: 'return'
          });
          
          geometries[routeId] = {
            segments: routeSegments,
            vehicleType: route.vehicleType,
            vehicleId: route.vehicleId
          };
        }
      }
      
      setRouteGeometries(geometries);
    };
    
    if (routes && Object.keys(routes).length > 0 && emergencies.length > 0 && depots.length > 0) {
      fetchRouteGeometries();
    } else {
      setRouteGeometries({});
    }
  }, [routes, emergencies, depots]);

  // Rota renklerini belirle
  const getRouteColor = (vehicleType) => {
    return vehicleType === 'ground' ? '#FF5733' : '#3366FF';
  };

  // Karanlık tema için harita katmanı
  const getTileLayer = () => {
    if (isDark) {
      return 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
    }
    return 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
  };

  return (
    <Paper p="md" withBorder>
      <Title order={3} mb="md">Acil Durum Haritası</Title>
      <MapContainer 
        center={center} 
        zoom={10} 
        style={{ height: '500px', width: '100%' }}
        ref={mapRef}
      >
        <TileLayer
          url={getTileLayer()}
          attribution={isDark 
            ? '&copy; <a href="https://carto.com/attributions">CARTO</a>' 
            : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          }
        />
        
        {mapBounds && <MapBoundsUpdater bounds={mapBounds} />}
        
        {/* Depoları göster */}
        {depots.map((depot) => (
          <Marker 
            key={`depot-${depot.id}`}
            position={[depot.latitude, depot.longitude]}
            icon={depotIcon}
          >
            <Popup>
              <Text weight={700}>{depot.name || `Depo ${depot.id}`}</Text>
              <Text size="sm">Araç Kapasitesi: {depot.vehicleCapacity}</Text>
            </Popup>
          </Marker>
        ))}
        
        {/* Yangın noktalarını göster */}
        {emergencies.map((emergency, index) => (
          <Marker 
            key={`emergency-${emergency.id || index}`}
            position={[parseFloat(emergency.latitude), parseFloat(emergency.longitude)]}
            icon={fireIcon}
          >
            <Popup>
              <Text weight={700}>Yangın {emergency.id || index + 1}</Text>
              <Group spacing="xs">
                <Badge color="red">Yoğunluk: {emergency.fireIntensity}/10</Badge>
                <Badge color="orange">Alan: {emergency.areaSize} ha</Badge>
              </Group>
              <Text size="sm">Arazi Zorluğu: {emergency.terrainDifficulty}/10</Text>
              <Text size="sm">Hava Durumu: {emergency.weatherCondition}</Text>
              <Text size="sm">Yol Durumu: {emergency.roadCondition}</Text>
            </Popup>
          </Marker>
        ))}
        
        {/* Rotaları göster */}
        {Object.entries(routeGeometries).map(([routeId, routeData]) => {
          const { segments, vehicleType, vehicleId } = routeData;
          const color = getRouteColor(vehicleType);
          const isDashed = vehicleType === 'aerial';
          
          return segments.map((segment, segmentIndex) => (
            <Polyline
              key={`route-${routeId}-segment-${segmentIndex}`}
              positions={segment.points}
              color={color}
              weight={3}
              opacity={0.8}
              dashArray={isDashed ? '5, 5' : ''}
            >
              <Popup>
                <Text weight={700}>
                  Araç {vehicleId} ({vehicleType === 'ground' ? 'Kara Aracı' : 'Hava Aracı'})
                </Text>
                <Text size="sm">
                  {segment.type === 'return' ? 'Depoya Dönüş' : 'Yangın Noktasına Gidiş'}
                </Text>
              </Popup>
            </Polyline>
          ));
        })}
      </MapContainer>
    </Paper>
  );
};

export default EmergencyMap; 