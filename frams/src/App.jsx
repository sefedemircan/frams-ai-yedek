import { useState, useEffect } from 'react';
import { Container, Grid, Button, LoadingOverlay, Title, Text, Alert, Group, useMantineColorScheme, ActionIcon, Switch, Tooltip, Paper, Table } from '@mantine/core';
import { IconSun, IconMoon, IconRoad } from '@tabler/icons-react';
import EmergencyForm from './components/EmergencyForm';
import EmergencyMap from './components/EmergencyMap';
import ResultsPanel from './components/ResultsPanel';
import api from './services/api';
import './App.css';

function App() {
  const [emergencies, setEmergencies] = useState([]);
  const [depots, setDepots] = useState([]);
  const [vehicles, setVehicles] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [considerTraffic, setConsiderTraffic] = useState(true);
  const { colorScheme, setColorScheme } = useMantineColorScheme();

  const toggleColorScheme = () => {
    setColorScheme(colorScheme === 'dark' ? 'light' : 'dark');
  };

  const toggleTrafficConsideration = () => {
    setConsiderTraffic(!considerTraffic);
  };

  useEffect(() => {
    // Depoları ve araçları yükle
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        const depotsData = await api.getDepots();
        const vehiclesData = await api.getVehicles();
        
        setDepots(depotsData.depots || []);
        setVehicles(vehiclesData.vehicles || []);
      } catch (err) {
        setError('Veri yüklenirken bir hata oluştu: ' + err.message);
        console.error('Veri yükleme hatası:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, []);

  const handleAddEmergency = (emergency) => {
    // Yeni yangın noktası ekle
    setEmergencies([...emergencies, { ...emergency, id: emergencies.length + 1 }]);
  };

  const handleOptimizeRoutes = async () => {
    if (emergencies.length === 0) {
      setError('Lütfen en az bir yangın noktası ekleyin.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const optimizationResults = await api.optimizeRoutes(emergencies, considerTraffic);
      setResults(optimizationResults);
    } catch (err) {
      setError('Rota optimizasyonu sırasında bir hata oluştu: ' + err.message);
      console.error('Optimizasyon hatası:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setEmergencies([]);
    setResults(null);
    setError(null);
  };

  return (
    <div>
      <div style={{ 
        padding: '1rem', 
        backgroundColor: colorScheme === 'dark' ? '#1A1B1E' : '#f8f9fa', 
        borderBottom: `1px solid ${colorScheme === 'dark' ? '#2C2E33' : '#dee2e6'}`,
        color: colorScheme === 'dark' ? 'white' : 'black'
      }}>
        <Group position="apart">
          <Title order={1}>Frams - Yangın Müdahale Sistemi</Title>
          <Group>
            <Tooltip label={considerTraffic ? 'Trafik verilerini dikkate alıyor' : 'Trafik verileri dikkate alınmıyor'}>
              <Switch
                checked={considerTraffic}
                onChange={toggleTrafficConsideration}
                label="Trafik Verisi Kullan"
                labelPosition="left"
                size="md"
                color="orange"
                onLabel={<IconRoad size={16} />}
                offLabel={<IconRoad size={16} />}
              />
            </Tooltip>
            <ActionIcon 
              variant="outline" 
              color={colorScheme === 'dark' ? 'yellow' : 'blue'} 
              onClick={toggleColorScheme} 
              title={colorScheme === 'dark' ? 'Aydınlık Moda Geç' : 'Karanlık Moda Geç'}
              size="lg"
            >
              {colorScheme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
            </ActionIcon>
          </Group>
        </Group>
      </div>
      
      <Container size="xl" py="xl">
        <LoadingOverlay visible={loading} overlayProps={{ radius: "sm", blur: 2 }}/>
        
        {error && (
          <Alert color="red" title="Hata" mb="md" withCloseButton onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        
        <Grid gutter="md">
          <Grid.Col span={12}>
            <EmergencyMap 
              emergencies={emergencies} 
              routes={results?.routes || {}} 
              depots={depots} 
            />
          </Grid.Col>
          
          <Grid.Col span={4}>
            <EmergencyForm onAddEmergency={handleAddEmergency} />
            
            <Group position="center" mt="md">
              <Button 
                color="green" 
                size="md" 
                onClick={handleOptimizeRoutes}
                disabled={emergencies.length === 0}
              >
                Rotaları Optimize Et
              </Button>
              
              <Button 
                color="red" 
                size="md" 
                onClick={handleReset}
                disabled={emergencies.length === 0}
              >
                Sıfırla
              </Button>
            </Group>
            
            {emergencies.length > 0 && (
              <>
                <Text mt="md" align="center">
                  {emergencies.length} yangın noktası eklendi
                </Text>
                
                <Paper p="md" withBorder mt="md">
                  <Title order={4} mb="sm">Yangın Noktaları</Title>
                  <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
                    <Table>
                      <Table.Thead>
                        <Table.Tr>
                          <Table.Th>No</Table.Th>
                          <Table.Th>Konum</Table.Th>
                          <Table.Th>Yoğunluk</Table.Th>
                          <Table.Th>Alan (ha)</Table.Th>
                          <Table.Th>Arazi Zorluğu</Table.Th>
                          <Table.Th>Hava Durumu</Table.Th>
                          <Table.Th>Yol Durumu</Table.Th>
                          <Table.Th>Su Mesafesi (km)</Table.Th>
                        </Table.Tr>
                      </Table.Thead>
                      <Table.Tbody>
                        {emergencies.map((emergency) => (
                          <Table.Tr 
                            key={emergency.id} 
                            style={{
                              animation: emergency === emergencies[emergencies.length - 1] ? 
                                'highlightRow 2s ease-in-out' : 'none'
                            }}
                          >
                            <Table.Td>{emergency.id}</Table.Td>
                            <Table.Td>{emergency.latitude}, {emergency.longitude}</Table.Td>
                            <Table.Td>{emergency.fireIntensity}</Table.Td>
                            <Table.Td>{emergency.areaSize}</Table.Td>
                            <Table.Td>{emergency.terrainDifficulty}</Table.Td>
                            <Table.Td>
                              {emergency.weatherCondition === 'good' ? 'İyi' : 
                               emergency.weatherCondition === 'moderate' ? 'Orta' : 'Kötü'}
                            </Table.Td>
                            <Table.Td>
                              {emergency.roadCondition === 'good' ? 'İyi' : 
                               emergency.roadCondition === 'moderate' ? 'Orta' : 'Kötü'}
                            </Table.Td>
                            <Table.Td>{emergency.distanceToNearestWater}</Table.Td>
                          </Table.Tr>
                        ))}
                      </Table.Tbody>
                    </Table>
                  </div>
                </Paper>
              </>
            )}
          </Grid.Col>
          
          <Grid.Col span={8}>
            <ResultsPanel results={results} vehicles={vehicles} />
          </Grid.Col>
        </Grid>
      </Container>
    </div>
  );
}

export default App;
