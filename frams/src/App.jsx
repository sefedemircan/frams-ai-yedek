import { useState, useEffect } from 'react';
import { Container, Grid, Button, LoadingOverlay, Title, Text, Alert, Group, AppShell } from '@mantine/core';
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
      
      const optimizationResults = await api.optimizeRoutes(emergencies);
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
      <div style={{ padding: '1rem', backgroundColor: '#f8f9fa', borderBottom: '1px solid #dee2e6' }}>
        <Group position="apart">
          <Title order={1}>Frams - Yangın Müdahale Sistemi</Title>
        </Group>
      </div>
      
      <Container size="xl" py="xl">
        <LoadingOverlay visible={loading} overlayBlur={2} />
        
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
              <Text mt="md" align="center">
                {emergencies.length} yangın noktası eklendi
              </Text>
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
