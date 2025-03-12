import { Paper, Title, Text, Group, Badge, Accordion, List, Divider, useMantineColorScheme } from '@mantine/core';

const ResultsPanel = ({ results, vehicles }) => {
  const { colorScheme } = useMantineColorScheme();
  const isDark = colorScheme === 'dark';

  if (!results || !results.routes || Object.keys(results.routes).length === 0) {
    return (
      <Paper p="md" withBorder>
        <Title order={3} mb="md">Sonuçlar</Title>
        <Text color="dimmed">Henüz sonuç yok. Lütfen önce rota optimizasyonu yapın.</Text>
      </Paper>
    );
  }

  const { routes, stats } = results;

  return (
    <Paper p="md" withBorder>
      <Title order={3} mb="md">Optimizasyon Sonuçları</Title>
      
      <Group mb="md">
        <Badge size="lg" color="blue">Toplam Yangın: {stats.totalEmergencies}</Badge>
        <Badge size="lg" color="green">Atanan: {stats.assignedEmergencies}</Badge>
        <Badge size="lg" color="red">Atanmayan: {stats.unassignedEmergencies}</Badge>
      </Group>
      
      <Divider mb="md" />
      
      <Accordion>
        {Object.values(routes).map((route) => {
          const vehicle = vehicles.find(v => v.id === route.vehicleId);
          
          return (
            <Accordion.Item key={route.vehicleId} value={`vehicle-${route.vehicleId}`}>
              <Accordion.Control>
                <Group>
                  <Text weight={700}>
                    Araç {route.vehicleId} ({route.vehicleType === 'ground' ? 'Kara Aracı' : 'Hava Aracı'})
                  </Text>
                  <Badge color={route.vehicleType === 'ground' ? 'red' : 'blue'}>
                    {route.emergencyIds.length} Yangın
                  </Badge>
                </Group>
              </Accordion.Control>
              <Accordion.Panel>
                {vehicle && (
                  <List size="sm" spacing="xs" mb="md">
                    <List.Item>Kapasite: {vehicle.capacity} L</List.Item>
                    <List.Item>Hız: {vehicle.speed} km/h</List.Item>
                    <List.Item>Menzil: {vehicle.range} km</List.Item>
                  </List>
                )}
                
                <Text weight={600} mb="xs">Atanan Yangınlar:</Text>
                <List>
                  {route.emergencyIds.map(emergencyId => {
                    const emergency = results.emergencies.find(e => e.id === emergencyId);
                    return (
                      <List.Item key={emergencyId}>
                        Yangın {emergencyId}
                        {emergency && (
                          <Text size="xs" color="dimmed">
                            Yoğunluk: {emergency.fireIntensity}/10, 
                            Alan: {emergency.areaSize} ha
                          </Text>
                        )}
                      </List.Item>
                    );
                  })}
                </List>
              </Accordion.Panel>
            </Accordion.Item>
          );
        })}
      </Accordion>
    </Paper>
  );
};

export default ResultsPanel; 