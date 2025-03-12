from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from emergency_system import (
    Depot, FireEmergency, EmergencyVehicle, VehicleType, 
    WeatherCondition, RoadCondition, EmergencyGeneticAlgorithm
)

app = Flask(__name__)
CORS(app)

# Depoları tanımla
DEPOTS = {
    1: {"lat": 37.7640, "lon": 30.5458, "name": "İstanbul Deposu"},
    2: {"lat": 37.6255, "lon": 30.5362, "name": "Ankara Deposu"},
    3: {"lat": 38.4237, "lon": 30.5622, "name": "İzmir Deposu"}
}

# Araç konfigürasyonları
VEHICLE_CONFIGS = {
    1: [  # İstanbul deposu
        {"id": 1, "type": VehicleType.GROUND, "capacity": 5000, "speed": 60, "range": 100},
        {"id": 2, "type": VehicleType.AERIAL, "capacity": 3000, "speed": 180, "range": 500}
    ],
    2: [  # Ankara deposu
        {"id": 3, "type": VehicleType.GROUND, "capacity": 4000, "speed": 55, "range": 80},
        {"id": 4, "type": VehicleType.AERIAL, "capacity": 3500, "speed": 170, "range": 450}
    ],
    3: [  # İzmir deposu
        {"id": 5, "type": VehicleType.GROUND, "capacity": 4500, "speed": 58, "range": 90},
        {"id": 6, "type": VehicleType.AERIAL, "capacity": 3200, "speed": 175, "range": 480}
    ]
}

def initialize_vehicles():
    """Araç filosunu oluştur"""
    vehicles = []
    for depot_id, configs in VEHICLE_CONFIGS.items():
        depot = DEPOTS[depot_id]
        for config in configs:
            vehicle = EmergencyVehicle(
                id=config["id"],
                type=config["type"],
                capacity=config["capacity"],
                speed=config["speed"],
                range=config["range"],
                current_lat=depot["lat"],
                current_lon=depot["lon"],
                depot_id=depot_id
            )
            vehicles.append(vehicle)
    return vehicles

def initialize_depots():
    """Depo nesnelerini oluştur"""
    depot_objects = []
    for depot_id, depot_data in DEPOTS.items():
        depot_objects.append(
            Depot(
                id=depot_id,
                latitude=depot_data["lat"],
                longitude=depot_data["lon"],
                name=depot_data["name"],
                vehicle_capacity=len(VEHICLE_CONFIGS[depot_id])
            )
        )
    return depot_objects

@app.route('/api/depots', methods=['GET'])
def get_depots():
    """Tüm depoları döndür"""
    return jsonify({
        "depots": [
            {
                "id": depot_id,
                "latitude": data["lat"],
                "longitude": data["lon"],
                "name": data.get("name", f"Depo {depot_id}"),
                "vehicleCapacity": len(VEHICLE_CONFIGS[depot_id])
            }
            for depot_id, data in DEPOTS.items()
        ]
    })

@app.route('/api/vehicles', methods=['GET'])
def get_vehicles():
    """Tüm araçları döndür"""
    vehicles = initialize_vehicles()
    return jsonify({
        "vehicles": [
            {
                "id": vehicle.id,
                "type": vehicle.type.value,
                "capacity": vehicle.capacity,
                "speed": vehicle.speed,
                "range": vehicle.range,
                "depotId": vehicle.depot_id
            }
            for vehicle in vehicles
        ]
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_routes():
    """Yangın noktaları için rota optimizasyonu yap"""
    try:
        data = request.json
        emergencies_data = data.get('emergencies', [])
        
        # Yangın noktalarını oluştur
        emergencies = []
        for i, emergency_data in enumerate(emergencies_data):
            emergencies.append(
                FireEmergency(
                    id=i + 1,
                    latitude=float(emergency_data['latitude']),
                    longitude=float(emergency_data['longitude']),
                    fire_intensity=float(emergency_data['fireIntensity']),
                    area_size=float(emergency_data['areaSize']),
                    terrain_difficulty=float(emergency_data['terrainDifficulty']),
                    weather_condition=WeatherCondition(emergency_data['weatherCondition']),
                    road_condition=RoadCondition(emergency_data['roadCondition']),
                    distance_to_nearest_water=float(emergency_data['distanceToNearestWater'])
                )
            )
        
        # Depoları ve araçları oluştur
        depots = initialize_depots()
        vehicles = initialize_vehicles()
        
        # Genetik algoritma ile rota optimizasyonu
        ga = EmergencyGeneticAlgorithm(
            depots=depots,
            emergencies=emergencies,
            vehicles=vehicles,
            population_size=50,
            generations=100,
            crossover_rate=0.9,
            mutation_rate=0.35
        )
        
        # Rotaları optimize et
        routes = ga.evolve()
        
        # Sonuçları hazırla
        result = {
            "routes": {},
            "emergencies": [],
            "stats": {
                "totalEmergencies": len(emergencies),
                "assignedEmergencies": 0,
                "unassignedEmergencies": 0
            }
        }
        
        # Rota bilgilerini ekle
        for vehicle_id, emergency_ids in routes.items():
            vehicle = next((v for v in vehicles if v.id == vehicle_id), None)
            if vehicle and emergency_ids:
                result["routes"][vehicle_id] = {
                    "vehicleId": vehicle_id,
                    "vehicleType": vehicle.type.value,
                    "emergencyIds": emergency_ids,
                    "depotId": vehicle.depot_id
                }
        
        # Yangın noktası bilgilerini ekle
        for emergency in emergencies:
            result["emergencies"].append({
                "id": emergency.id,
                "latitude": emergency.latitude,
                "longitude": emergency.longitude,
                "fireIntensity": emergency.fire_intensity,
                "areaSize": emergency.area_size,
                "terrainDifficulty": emergency.terrain_difficulty,
                "weatherCondition": emergency.weather_condition.value,
                "roadCondition": emergency.road_condition.value,
                "distanceToNearestWater": emergency.distance_to_nearest_water
            })
        
        # İstatistikleri hesapla
        assigned_emergency_ids = set()
        for vehicle_id, route_data in result["routes"].items():
            assigned_emergency_ids.update(route_data["emergencyIds"])
        
        result["stats"]["assignedEmergencies"] = len(assigned_emergency_ids)
        result["stats"]["unassignedEmergencies"] = len(emergencies) - len(assigned_emergency_ids)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 