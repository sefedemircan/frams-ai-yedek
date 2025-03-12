# Frams Backend - Yangın Müdahale Sistemi API

Bu proje, yangın acil durum müdahale sisteminin backend API'sini içerir. Genetik algoritma ve bulanık mantık kullanarak yangın noktalarına en uygun araç atamalarını yapar.

## Özellikler

- Çoklu depo desteği
- Kara ve hava araçları için rota optimizasyonu
- Bulanık mantık ile araç uygunluk değerlendirmesi
- Genetik algoritma ile rota optimizasyonu
- RESTful API

## Kurulum

1. Gerekli Python paketlerini yükleyin:

```bash
pip install -r requirements.txt
```

2. Uygulamayı başlatın:

```bash
python app.py
```

Uygulama varsayılan olarak `http://localhost:5000` adresinde çalışacaktır.

## API Endpoints

### GET /api/depots

Tüm depoları döndürür.

### GET /api/vehicles

Tüm araçları döndürür.

### POST /api/optimize

Yangın noktaları için rota optimizasyonu yapar.

**İstek Gövdesi:**

```json
{
  "emergencies": [
    {
      "latitude": 37.7640,
      "longitude": 30.5458,
      "fireIntensity": 8.5,
      "areaSize": 3.2,
      "terrainDifficulty": 6.7,
      "weatherCondition": "moderate",
      "roadCondition": "good",
      "distanceToNearestWater": 2.5
    },
    ...
  ]
}
```

**Yanıt:**

```json
{
  "routes": {
    "1": {
      "vehicleId": 1,
      "vehicleType": "ground",
      "emergencyIds": [1, 3],
      "depotId": 1
    },
    ...
  },
  "emergencies": [...],
  "stats": {
    "totalEmergencies": 5,
    "assignedEmergencies": 5,
    "unassignedEmergencies": 0
  }
}
```

## Geliştirme

Bu proje, Python 3.8+ ile geliştirilmiştir. Katkıda bulunmak için lütfen bir pull request açın. 