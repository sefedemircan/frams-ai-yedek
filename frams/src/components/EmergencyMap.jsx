import { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, Circle, LayersControl } from 'react-leaflet';
import { Paper, Title, Text, Badge, Group, useMantineColorScheme, Button, Switch, Tooltip } from '@mantine/core';
import { IconNavigation, IconNavigationOff, IconRoad } from '@tabler/icons-react';
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
const userIcon = createCustomIcon('blue');

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

// GPS takibi için bileşen
const LocationMarker = ({ followLocation, onAccuracyUpdate }) => {
  const [position, setPosition] = useState(null);
  const [accuracy, setAccuracy] = useState(0);
  const [heading, setHeading] = useState(null);
  const [locationError, setLocationError] = useState(null);
  const [positionHistory, setPositionHistory] = useState([]);
  const [filteredPosition, setFilteredPosition] = useState(null);
  
  const map = useMap();

  // Konum takibi
  useEffect(() => {
    if (!followLocation) return;

    let watchId = null;

    const success = (position) => {
      const lat = position.coords.latitude;
      const lng = position.coords.longitude;
      const acc = position.coords.accuracy;
      const hdg = position.coords.heading;
      
      // Konum geçmişini güncelle (son 5 konumu sakla)
      const newPosition = { lat, lng, acc, timestamp: Date.now() };
      setPositionHistory(prev => {
        const updated = [...prev, newPosition].slice(-5);
        return updated;
      });
      
      setPosition([lat, lng]);
      setAccuracy(acc);
      if (hdg !== null) setHeading(hdg);
      setLocationError(null);

      // Haritayı konum etrafında merkeze al
      if (followLocation && map) {
        map.setView([lat, lng], map.getZoom());
      }

      // Konum doğruluğunu güncellemek için callback
      if (onAccuracyUpdate) {
        onAccuracyUpdate(acc);
      }
    };

    const error = (err) => {
      console.error('Konum alınamadı:', err);
      setLocationError(err.message);
    };

    const options = {
      enableHighAccuracy: true,
      maximumAge: 1000,
      timeout: 15000
    };

    // Konum izlemeyi başlat
    if (navigator.geolocation) {
      watchId = navigator.geolocation.watchPosition(success, error, options);
    } else {
      setLocationError('Tarayıcınız konum hizmetlerini desteklemiyor.');
    }

    // Temizlik fonksiyonu
    return () => {
      if (watchId !== null) {
        navigator.geolocation.clearWatch(watchId);
      }
    };
  }, [followLocation, map, onAccuracyUpdate]);

  // Konum verilerini filtrele ve iyileştir
  useEffect(() => {
    // En az 3 konum kaydı olduğunda filtreleme yap
    if (positionHistory.length >= 3) {
      // Sadece belirli bir doğruluk seviyesinin altındaki konumları kullan
      const filteredPositions = positionHistory.filter(pos => pos.acc < 100);
      
      if (filteredPositions.length >= 2) {
        // En son konumları ağırlıklı olarak hesapla (son konum daha önemli)
        let totalWeight = 0;
        let weightedLat = 0;
        let weightedLng = 0;
        
        filteredPositions.forEach((pos, index) => {
          // Doğruluk değeri daha düşük olan konumlara daha yüksek ağırlık ver
          const accuracyWeight = 1 / (pos.acc || 1);
          // Daha yeni konumlara daha yüksek ağırlık ver
          const recencyWeight = (index + 1) / filteredPositions.length;
          // Toplam ağırlık
          const weight = accuracyWeight * recencyWeight;
          
          weightedLat += pos.lat * weight;
          weightedLng += pos.lng * weight;
          totalWeight += weight;
        });
        
        if (totalWeight > 0) {
          const avgLat = weightedLat / totalWeight;
          const avgLng = weightedLng / totalWeight;
          
          // Ortalama doğruluk değerini hesapla (en iyiden alınır)
          const bestAccuracy = Math.min(...filteredPositions.map(pos => pos.acc));
          
          setFilteredPosition([avgLat, avgLng]);
          
          // Filtrelenmiş konum varsa ve doğruluk yeterince iyiyse, haritayı güncelle
          if (followLocation && map && bestAccuracy < 50) {
            map.setView([avgLat, avgLng], map.getZoom());
          }
        }
      }
    }
  }, [positionHistory, followLocation, map]);

  // Kullanılacak konum bilgisini belirle (ham veya filtrelenmiş)
  const displayPosition = (filteredPosition && positionHistory.length >= 3) ? filteredPosition : position;
  const displayAccuracy = position ? accuracy : 0;

  return displayPosition ? (
    <>
      <Marker position={displayPosition} icon={userIcon}>
        <Popup>
          <Text weight={700}>Konumunuz</Text>
          <Text size="sm">Koordinatlar: {displayPosition[0].toFixed(6)}, {displayPosition[1].toFixed(6)}</Text>
          <Text size="sm">Hassasiyet: ±{Math.round(displayAccuracy)} m</Text>
          {heading !== null && <Text size="sm">Yön: {Math.round(heading)}°</Text>}
          {filteredPosition && positionHistory.length >= 3 && (
            <Text size="sm" color="green">İyileştirilmiş konum aktif</Text>
          )}
        </Popup>
      </Marker>
      <Circle 
        center={displayPosition} 
        radius={displayAccuracy} 
        pathOptions={{ 
          color: displayAccuracy < 20 ? 'green' : displayAccuracy < 50 ? 'orange' : 'red',
          fillColor: displayAccuracy < 20 ? 'green' : displayAccuracy < 50 ? 'orange' : 'red', 
          fillOpacity: 0.1,
          weight: 2
        }}
      />
    </>
  ) : locationError ? (
    <Popup position={map.getCenter()} closeButton={false}>
      <Text color="red">{locationError}</Text>
    </Popup>
  ) : null;
};

const EmergencyMap = ({ emergencies, routes, depots }) => {
  const { colorScheme } = useMantineColorScheme();
  const isDark = colorScheme === 'dark';
  
  const [center, setCenter] = useState([37.7640, 30.5458]); // Varsayılan merkez
  const [mapBounds, setMapBounds] = useState(null);
  const [routeGeometries, setRouteGeometries] = useState({});
  const [followLocation, setFollowLocation] = useState(false);
  const [locationAccuracy, setLocationAccuracy] = useState(null);
  const [showTraffic, setShowTraffic] = useState(false);
  const mapRef = useRef(null);
  const layersControlRef = useRef(null);

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

  // Harita katmanı URL'si
  const getTileLayer = () => {
    if (isDark) {
      return 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
    }
    return 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
  };

  // Trafik katmanı URL'si
  const getTrafficTileLayer = () => {
    // Ücretsiz trafik veri katmanı - OpenTransportMap kullanımı
    if (isDark) {
      // Karanlık tema için alternatif trafik katmanı
      return 'https://tile.thunderforest.com/transport-dark/{z}/{x}/{y}.png?apikey=6170aad10dfd42a38d4d8c709a536f38';
    }
    return 'https://tile.thunderforest.com/transport/{z}/{x}/{y}.png?apikey=6170aad10dfd42a38d4d8c709a536f38';
  };

  // Trafik katmanı açıklama metni
  const getTrafficAttribution = () => {
    return '&copy; <a href="https://www.thunderforest.com/">Thunderforest</a>, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
  };

  // Kullanıcının konumuna git
  const handleGetLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude, accuracy } = position.coords;
          
          // Doğruluk oranına göre zoom seviyesini ayarla
          let zoomLevel = 18; // Varsayılan yüksek zoom seviyesi
          
          if (accuracy > 100) {
            zoomLevel = 14; // Düşük doğruluk için daha geniş görünüm
          } else if (accuracy > 50) {
            zoomLevel = 16; // Orta doğruluk için orta seviye zoom
          }
          
          if (mapRef.current) {
            const map = mapRef.current;
            
            // Önce konuma odaklan
            map.setView([latitude, longitude], zoomLevel);
            
            // Doğruluk durumunu göster
            if (accuracy > 100) {
              alert(`Konum doğruluğu düşük (±${Math.round(accuracy)} metre). Daha iyi sonuçlar için açık alanda olduğunuzdan emin olun.`);
            }
          }
        },
        (error) => {
          console.error('Konum alınamadı:', error);
          let errorMsg = 'Konumunuz alınamadı. ';
          
          switch (error.code) {
            case error.PERMISSION_DENIED:
              errorMsg += 'Konum iznini reddettiniz. Lütfen tarayıcı ayarlarından konum iznini etkinleştirin.';
              break;
            case error.POSITION_UNAVAILABLE:
              errorMsg += 'Konum bilgisi mevcut değil. Lütfen cihazınızın GPS özelliğinin açık olduğundan emin olun.';
              break;
            case error.TIMEOUT:
              errorMsg += 'Konum alınırken zaman aşımı oluştu. Lütfen internet bağlantınızı kontrol edin ve tekrar deneyin.';
              break;
            default:
              errorMsg += 'Bilinmeyen bir hata oluştu.';
          }
          
          alert(errorMsg);
        },
        {
          enableHighAccuracy: true,
          timeout: 15000,
          maximumAge: 0
        }
      );
    } else {
      alert('Tarayıcınız konum hizmetlerini desteklemiyor.');
    }
  };

  // Konum takibini aç/kapa
  const toggleLocationTracking = () => {
    setFollowLocation(!followLocation);
    
    // Konum takibi kapatıldığında doğruluk göstergesini de temizle
    if (followLocation) {
      setLocationAccuracy(null);
    }
  };

  // Konum doğruluğunu güncellemek için callback
  const handleAccuracyUpdate = (accuracy) => {
    setLocationAccuracy(accuracy);
  };

  // Trafik görünümünü aç/kapa
  const toggleTrafficLayer = () => {
    setShowTraffic(!showTraffic);
    
    // Harita referansını kontrol et
    if (mapRef.current) {
      // BaseLayer değişimi için doğrudan DOM manipülasyonu (React-Leaflet sınırlamaları nedeniyle)
      const mapContainer = mapRef.current._container;
      const layerControls = mapContainer.querySelectorAll('.leaflet-control-layers-selector');
      
      if (layerControls && layerControls.length >= 2) {
        // İlk kontrol standart harita, ikinci kontrol trafik haritası
        const targetControl = !showTraffic ? layerControls[1] : layerControls[0];
        if (targetControl && !targetControl.checked) {
          targetControl.click(); // Katmanı değiştir
        }
      }
    }
  };

  // MapContainer yüklendikten sonra referans almak için bileşen
  const MapController = () => {
    const map = useMap();
    
    useEffect(() => {
      if (map && !mapRef.current) {
        mapRef.current = map;
      }
    }, [map]);
    
    return null;
  };

  // Doğruluk metni için stil ve renk
  const getAccuracyStyle = (accuracy) => {
    if (!accuracy) return { color: 'gray', text: 'Konum alınmadı' };
    
    if (accuracy < 20) {
      return { color: 'green', text: `Yüksek doğruluk (±${Math.round(accuracy)}m)` };
    } else if (accuracy < 50) {
      return { color: 'orange', text: `Orta doğruluk (±${Math.round(accuracy)}m)` };
    } else {
      return { color: 'red', text: `Düşük doğruluk (±${Math.round(accuracy)}m)` };
    }
  };
  
  const accuracyStyle = getAccuracyStyle(locationAccuracy);

  return (
    <Paper p="md" withBorder>
      <Group position="apart" mb="md">
        <Group>
          <Title order={3}>Acil Durum Haritası</Title>
          {locationAccuracy !== null && (
            <Badge 
              color={accuracyStyle.color} 
              variant="light"
              size="lg"
            >
              {accuracyStyle.text}
            </Badge>
          )}
        </Group>
        <Group>
          <Tooltip label={showTraffic ? "Standart haritaya geç" : "Trafik haritasına geç"}>
            <Switch 
              checked={showTraffic}
              onChange={toggleTrafficLayer}
              size="md"
              onLabel={<IconRoad size={16} />}
              offLabel={<IconRoad size={16} />}
              color="yellow"
              label="Trafik"
              labelPosition="left"
            />
          </Tooltip>
          <Tooltip label={followLocation ? "Konum takibini kapat" : "Konum takibini aç"}>
            <Switch 
              checked={followLocation}
              onChange={toggleLocationTracking}
              size="md"
              onLabel={<IconNavigation size={16} />}
              offLabel={<IconNavigationOff size={16} />}
            />
          </Tooltip>
          <Button 
            leftSection={<IconNavigation size={16} />} 
            onClick={handleGetLocation}
            size="sm"
            variant="outline"
          >
            Konumuma Git
          </Button>
        </Group>
      </Group>
      
      <MapContainer 
        center={center} 
        zoom={10} 
        style={{ height: '500px', width: '100%' }}
      >
        <MapController />
        
        <LayersControl position="topright" ref={layersControlRef}>
          <LayersControl.BaseLayer checked={!showTraffic} name="Standart Harita">
            <TileLayer
              url={getTileLayer()}
              attribution={isDark 
                ? '&copy; <a href="https://carto.com/attributions">CARTO</a>' 
                : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              }
            />
          </LayersControl.BaseLayer>
          
          <LayersControl.BaseLayer checked={showTraffic} name="Trafik Haritası">
            <TileLayer
              url={getTrafficTileLayer()}
              attribution={getTrafficAttribution()}
            />
          </LayersControl.BaseLayer>
        </LayersControl>
        
        {mapBounds && <MapBoundsUpdater bounds={mapBounds} />}
        
        {/* GPS konumunu göster */}
        <LocationMarker 
          followLocation={followLocation} 
          onAccuracyUpdate={handleAccuracyUpdate}
        />
        
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