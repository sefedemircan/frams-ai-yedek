# Frams - Yangın Müdahale Sistemi

Bu proje, yangın acil durum müdahale sisteminin frontend uygulamasıdır. React ve Mantine UI kullanılarak geliştirilmiştir.

## Özellikler

- Yangın noktası ekleme ve yönetme
- Harita üzerinde yangın noktalarını ve rotaları görselleştirme
- Rota optimizasyonu sonuçlarını görüntüleme
- Duyarlı ve modern kullanıcı arayüzü

## Kurulum

1. Gerekli paketleri yükleyin:

```bash
npm install
```

2. Geliştirme sunucusunu başlatın:

```bash
npm run dev
```

Uygulama varsayılan olarak `http://localhost:5173` adresinde çalışacaktır.

## Kullanım

1. Yangın noktası eklemek için formu doldurun ve "Yangın Noktası Ekle" düğmesine tıklayın.
2. Birden fazla yangın noktası ekleyebilirsiniz.
3. Tüm yangın noktaları eklendikten sonra "Rotaları Optimize Et" düğmesine tıklayın.
4. Optimizasyon sonuçları harita üzerinde ve sonuç panelinde görüntülenecektir.

## Bağımlılıklar

- React
- Mantine UI
- Leaflet (harita görselleştirme)
- Axios (API istekleri)

## Backend Bağlantısı

Bu uygulama, `http://localhost:5000` adresinde çalışan bir backend API'sine bağlanır. Backend'i çalıştırmak için lütfen `frams-backend` dizinindeki talimatları izleyin.
