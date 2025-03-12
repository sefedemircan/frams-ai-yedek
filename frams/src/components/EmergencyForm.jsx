import { 
  TextInput, 
  NumberInput, 
  Select, 
  Button, 
  Group, 
  Box, 
  Title, 
  Paper,
  Slider,
  Stack,
  Text
} from '@mantine/core';
import { useForm } from '@mantine/form';

const EmergencyForm = ({ onAddEmergency }) => {
  const form = useForm({
    initialValues: {
      latitude: '',
      longitude: '',
      fireIntensity: 5,
      areaSize: 1,
      terrainDifficulty: 5,
      weatherCondition: 'good',
      roadCondition: 'good',
      distanceToNearestWater: 1,
    },
    validate: {
      latitude: (value) => (value ? null : 'Enlem gerekli'),
      longitude: (value) => (value ? null : 'Boylam gerekli'),
      areaSize: (value) => (value > 0 ? null : 'Alan büyüklüğü pozitif olmalı'),
      distanceToNearestWater: (value) => (value >= 0 ? null : 'Su kaynağı mesafesi negatif olamaz'),
    },
  });

  const handleSubmit = (values) => {
    onAddEmergency(values);
    form.reset();
  };

  return (
    <Paper p="md" withBorder>
      <Title order={3} mb="md">Yangın Noktası Ekle</Title>
      <form onSubmit={form.onSubmit(handleSubmit)}>
        <Stack spacing="md">
          <Group grow>
            <TextInput
              label="Enlem"
              placeholder="37.7640"
              {...form.getInputProps('latitude')}
            />
            <TextInput
              label="Boylam"
              placeholder="30.5458"
              {...form.getInputProps('longitude')}
            />
          </Group>

          <Box>
            <Text size="sm" weight={500} mb={5}>Yangın Yoğunluğu (0-10)</Text>
            <Slider
              min={0}
              max={10}
              step={0.1}
              label={(value) => value.toFixed(1)}
              {...form.getInputProps('fireIntensity')}
            />
          </Box>

          <NumberInput
            label="Alan Büyüklüğü (hektar)"
            placeholder="1.5"
            min={0.1}
            step={0.1}
            {...form.getInputProps('areaSize')}
          />

          <Box>
            <Text size="sm" weight={500} mb={5}>Arazi Zorluğu (0-10)</Text>
            <Slider
              min={0}
              max={10}
              step={0.1}
              label={(value) => value.toFixed(1)}
              {...form.getInputProps('terrainDifficulty')}
            />
          </Box>

          <Select
            label="Hava Durumu"
            data={[
              { value: 'good', label: 'İyi' },
              { value: 'moderate', label: 'Orta' },
              { value: 'bad', label: 'Kötü' },
            ]}
            {...form.getInputProps('weatherCondition')}
          />

          <Select
            label="Yol Durumu"
            data={[
              { value: 'good', label: 'İyi' },
              { value: 'moderate', label: 'Orta' },
              { value: 'bad', label: 'Kötü' },
            ]}
            {...form.getInputProps('roadCondition')}
          />

          <NumberInput
            label="En Yakın Su Kaynağı (km)"
            placeholder="1.0"
            min={0}
            step={0.1}
            {...form.getInputProps('distanceToNearestWater')}
          />

          <Group position="right" mt="md">
            <Button type="submit">Yangın Noktası Ekle</Button>
          </Group>
        </Stack>
      </form>
    </Paper>
  );
};

export default EmergencyForm; 