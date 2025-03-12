import math
import random
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
import requests
import folium
from folium import plugins
import webbrowser
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Union

@dataclass
class Chromosome:
    stops: List[int] = field(default_factory=list)
    vehicles: List[int] = field(default_factory=list)
    fitness: float = 0.0

@dataclass
class Depot:
    id: int
    latitude: float
    longitude: float
    name: str
    vehicle_capacity: int  # Maximum number of vehicles that can be stationed

class VehicleType(Enum):
    GROUND = "ground"
    AERIAL = "aerial"

class WeatherCondition(Enum):
    GOOD = "good"
    MODERATE = "moderate"
    BAD = "bad"

class RoadCondition(Enum):
    GOOD = "good"
    MODERATE = "moderate"
    BAD = "bad"

@dataclass
class FireEmergency:
    id: int
    latitude: float
    longitude: float
    fire_intensity: float  # 0-10 scale
    area_size: float      # in hectares
    terrain_difficulty: float  # 0-10 scale
    weather_condition: WeatherCondition
    road_condition: RoadCondition
    distance_to_nearest_water: float  # in kilometers

@dataclass
class EmergencyVehicle:
    def __init__(self, id: int, type: VehicleType, capacity: float,
                 speed: float, range: float, current_lat: float,
                 current_lon: float, depot_id: int):
        self.id = id
        self.type = type
        self.capacity = capacity
        self.speed = speed
        self.range = range
        self.current_lat = current_lat
        self.current_lon = current_lon
        self.depot_id = depot_id
        self.is_available = True

@dataclass
class GAEmergencyRoute:
    vehicle_id: int
    emergency_sequence: List[int]  # Yangın noktalarının sırası
    is_aerial: bool
    fitness: float = 0.0

class EmergencyGeneticAlgorithm:
    def __init__(self,
                 depots: List[Depot],
                 emergencies: List[FireEmergency],
                 vehicles: List[EmergencyVehicle],
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.35):

        self.depots = {depot.id: depot for depot in depots}
        self.emergencies = emergencies
        self.vehicles = vehicles
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fuzzy_system = FuzzyFireResponseSystem()

        self.distance_cache = {}
        self._precompute_distances()

    def _precompute_distances(self):
        """Tüm noktalar arası mesafeleri önceden hesapla"""
        # Tüm depolar ve acil durum noktalarını içeren liste
        all_points = [(d.latitude, d.longitude, f'depot_{d.id}') for d in self.depots.values()]
        all_points.extend([(e.latitude, e.longitude, e.id) for e in self.emergencies])

        # Mesafeleri hesapla ve önbellekte sakla
        for i, (lat1, lon1, id1) in enumerate(all_points):
            for j, (lat2, lon2, id2) in enumerate(all_points[i + 1:], i + 1):
                # Hava araçları için mesafe
                aerial_dist = self._calculate_aerial_distance(lat1, lon1, lat2, lon2)
                self.distance_cache[f'aerial_{id1}_{id2}'] = aerial_dist
                self.distance_cache[f'aerial_{id2}_{id1}'] = aerial_dist

                # Kara araçları için mesafe
                try:
                    ground_dist = get_osrm_distance(lat1, lon1, lat2, lon2)
                    self.distance_cache[f'ground_{id1}_{id2}'] = ground_dist
                    self.distance_cache[f'ground_{id2}_{id1}'] = ground_dist
                except Exception as e:
                    print(f"OSRM error for points {id1}-{id2}: {e}")
                    self.distance_cache[f'ground_{id1}_{id2}'] = aerial_dist * 1.3
                    self.distance_cache[f'ground_{id2}_{id1}'] = aerial_dist * 1.3

    def get_nearest_depot(self, emergency: FireEmergency, vehicle_type: VehicleType) -> int:
        """Acil durum noktasına en yakın ve uygun araç kapasitesi olan depoyu bul"""
        min_distance = float('inf')
        nearest_depot_id = None

        for depot in self.depots.values():
            # İlgili depodaki müsait araç sayısını kontrol et
            available_vehicles = sum(1 for v in self.vehicles
                                     if v.depot_id == depot.id and
                                     v.type == vehicle_type and
                                     v.is_available)

            if available_vehicles > 0:
                distance = self.get_cached_distance(
                    f'depot_{depot.id}',
                    emergency.id,
                    vehicle_type
                )

                if distance < min_distance:
                    min_distance = distance
                    nearest_depot_id = depot.id

        return nearest_depot_id

    def _calculate_aerial_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Haversine formülü ile kuş uçuşu mesafe hesaplama"""
        R = 6371  # Dünya yarıçapı (km)
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def get_cached_distance(self, point1_id: Union[int, str], point2_id: Union[int, str],
                            vehicle_type: VehicleType) -> float:
        """Önbellekteki mesafeyi getir"""
        cache_key = f'{vehicle_type.value}_{point1_id}_{point2_id}'
        return self.distance_cache.get(cache_key, float('inf'))

    def calculate_route_distance(self, route: GAEmergencyRoute) -> float:
        """Önbelleği kullanarak rota mesafesini hesapla - depo bazlı"""
        total_distance = 0
        vehicle = next(v for v in self.vehicles if v.id == route.vehicle_id)
        depot_id = vehicle.depot_id
        current_point = f'depot_{depot_id}'

        for emergency_id in route.emergency_sequence:
            distance = self.get_cached_distance(
                current_point,
                emergency_id,
                VehicleType.AERIAL if route.is_aerial else VehicleType.GROUND
            )
            total_distance += distance
            current_point = emergency_id

        # Kendi deposuna dönüş mesafesi
        total_distance += self.get_cached_distance(
            current_point,
            f'depot_{depot_id}',
            VehicleType.AERIAL if route.is_aerial else VehicleType.GROUND
        )

        return total_distance
    def evaluate_fitness(self, route: GAEmergencyRoute) -> float:
        """Rotanın uygunluğunu değerlendir"""
        vehicle = next(v for v in self.vehicles if v.id == route.vehicle_id)
        total_distance = self.calculate_route_distance(route)

        # Menzil kontrolü
        if total_distance > vehicle.range:
            return 0.0

        # Fitness değeri hesaplama
        base_fitness = 1.0 / (total_distance + 1)

        # Hava durumu ve yol durumu etkisi
        weather_penalty = 1.0
        road_penalty = 1.0

        for emergency_id in route.emergency_sequence:
            emergency = next(e for e in self.emergencies if e.id == emergency_id)

            if route.is_aerial:
                if emergency.weather_condition == WeatherCondition.BAD:
                    weather_penalty *= 0.5
            else:  # Kara aracı
                if emergency.road_condition == RoadCondition.BAD:
                    road_penalty *= 0.5

        return base_fitness * weather_penalty * road_penalty

    def generate_initial_population(self) -> List[GAEmergencyRoute]:
        """Başlangıç popülasyonunu oluştur - en az bir acil durum içerecek şekilde"""
        population = []
        emergency_ids = [e.id for e in self.emergencies]

        for _ in range(self.population_size):
            for vehicle in self.vehicles:
                # Her rota için en az 1 acil durum olacak şekilde
                num_emergencies = random.randint(1, max(1, len(emergency_ids)))
                emergency_sequence = random.sample(emergency_ids, k=num_emergencies)

                route = GAEmergencyRoute(
                    vehicle_id=vehicle.id,
                    emergency_sequence=emergency_sequence,
                    is_aerial=vehicle.type == VehicleType.AERIAL
                )
                route.fitness = self.evaluate_fitness(route)
                population.append(route)

        return population
    def crossover(self, parent1: GAEmergencyRoute, parent2: GAEmergencyRoute) -> GAEmergencyRoute:
        """İki rotayı çaprazla - güvenli crossover noktası seçimi ile"""
        if parent1.vehicle_id != parent2.vehicle_id:
            return parent1

        if random.random() > self.crossover_rate:
            return parent1

        # Eğer rotalardan biri boşsa veya tek elemanlıysa, diğer rotayı döndür
        if len(parent1.emergency_sequence) <= 1 or len(parent2.emergency_sequence) <= 1:
            return parent1 if len(parent1.emergency_sequence) > len(parent2.emergency_sequence) else parent2

        # Güvenli crossover noktası seçimi
        max_point = min(len(parent1.emergency_sequence), len(parent2.emergency_sequence)) - 1
        if max_point < 1:
            return parent1

        crossover_point = random.randint(1, max_point)

        # Yeni sequence oluştur
        child_sequence = parent1.emergency_sequence[:crossover_point]
        remaining_emergencies = [x for x in parent2.emergency_sequence if x not in child_sequence]

        # Kalan acil durumlardan rastgele seç
        if remaining_emergencies:
            num_to_add = random.randint(1, len(remaining_emergencies))
            child_sequence.extend(random.sample(remaining_emergencies, num_to_add))

        return GAEmergencyRoute(
            vehicle_id=parent1.vehicle_id,
            emergency_sequence=child_sequence,
            is_aerial=parent1.is_aerial
        )

    def mutate(self, route: GAEmergencyRoute):
        """Rotayı mutasyona uğrat - güvenli mutasyon kontrolü ile"""
        if random.random() > self.mutation_rate or len(route.emergency_sequence) < 1:
            return

        mutation_type = random.choice(['swap', 'insert', 'reverse'])

        if mutation_type == 'swap' and len(route.emergency_sequence) >= 2:
            idx1, idx2 = random.sample(range(len(route.emergency_sequence)), 2)
            route.emergency_sequence[idx1], route.emergency_sequence[idx2] = \
                route.emergency_sequence[idx2], route.emergency_sequence[idx1]

        elif mutation_type == 'insert' and len(route.emergency_sequence) >= 2:
            from_idx = random.randint(0, len(route.emergency_sequence) - 1)
            to_idx = random.randint(0, len(route.emergency_sequence) - 1)
            value = route.emergency_sequence.pop(from_idx)
            route.emergency_sequence.insert(to_idx, value)

        elif mutation_type == 'reverse' and len(route.emergency_sequence) >= 2:
            if len(route.emergency_sequence) == 2:
                route.emergency_sequence.reverse()
            else:
                start = random.randint(0, len(route.emergency_sequence) - 2)
                end = random.randint(start + 1, len(route.emergency_sequence) - 1)
                route.emergency_sequence[start:end] = reversed(route.emergency_sequence[start:end])

    def optimize_dual_response_routes(self, emergencies: List[FireEmergency]) -> Dict[int, List[FireEmergency]]:
        """Çoklu depo desteği ile rota optimizasyonu"""
        self.final_routes = {vehicle.id: [] for vehicle in self.vehicles}
        unassigned_emergencies = list(emergencies)

        # Yangınları öncelik sırasına göre sırala
        emergency_priorities = []
        for emergency in unassigned_emergencies:
            priority_score = (
                    emergency.fire_intensity * 0.4 +
                    emergency.area_size * 0.3 +
                    emergency.terrain_difficulty * 0.3
            )
            emergency_priorities.append((emergency, priority_score))

        emergency_priorities.sort(key=lambda x: x[1], reverse=True)

        # Her yangın için araç ataması yap
        for emergency, priority_score in emergency_priorities:
            requires_dual_response = (
                    emergency.fire_intensity >= 7.0 or
                    emergency.area_size >= 5.0 or
                    priority_score >= 7.0
            )

            # En yakın depoları bul
            nearest_ground_depot = self.get_nearest_depot(emergency, VehicleType.GROUND)
            nearest_aerial_depot = self.get_nearest_depot(emergency, VehicleType.AERIAL)

            # Müsait araçları en yakın depolardan seç
            available_ground_vehicles = [
                v for v in self.vehicles
                if v.type == VehicleType.GROUND and
                   v.is_available and
                   len(self.final_routes[v.id]) < 3 and
                   v.depot_id == nearest_ground_depot
            ] if nearest_ground_depot else []

            available_aerial_vehicles = [
                v for v in self.vehicles
                if v.type == VehicleType.AERIAL and
                   v.is_available and
                   len(self.final_routes[v.id]) < 3 and
                   v.depot_id == nearest_aerial_depot
            ] if nearest_aerial_depot else []

            if requires_dual_response and emergency.weather_condition != WeatherCondition.BAD:
                # Çift müdahale için araç seçimi
                if available_ground_vehicles and available_aerial_vehicles:
                    # En uygun kara aracını seç
                    ground_vehicle = self._select_best_vehicle(emergency, available_ground_vehicles)
                    if ground_vehicle:
                        self.final_routes[ground_vehicle.id].append(emergency.id)
                        if len(self.final_routes[ground_vehicle.id]) >= 3:
                            ground_vehicle.is_available = False

                    # En uygun hava aracını seç
                    aerial_vehicle = self._select_best_vehicle(emergency, available_aerial_vehicles)
                    if aerial_vehicle:
                        self.final_routes[aerial_vehicle.id].append(emergency.id)
                        if len(self.final_routes[aerial_vehicle.id]) >= 3:
                            aerial_vehicle.is_available = False
            else:
                # Tek araç ataması - en yakın depodan
                all_available = available_ground_vehicles + available_aerial_vehicles
                if all_available:
                    best_vehicle = self._select_best_vehicle(emergency, all_available)
                    if best_vehicle:
                        self.final_routes[best_vehicle.id].append(emergency.id)
                        if len(self.final_routes[best_vehicle.id]) >= 3:
                            best_vehicle.is_available = False

        return self.final_routes
    def _select_best_vehicle(self, emergency: FireEmergency,
                           available_vehicles: List[EmergencyVehicle]) -> EmergencyVehicle:
        """Depo mesafesini de dikkate alarak en uygun aracı seç"""
        vehicle_scores = []

        for vehicle in available_vehicles:
            # Temel uygunluk skorunu hesapla
            base_score = self.fuzzy_system._calculate_vehicle_suitability(emergency, vehicle)

            # Araç deposu ile yangın noktası arası mesafe faktörü
            depot_distance = self.get_cached_distance(
                f'depot_{vehicle.depot_id}',
                emergency.id,
                vehicle.type
            )
            distance_factor = 1.0 / (1.0 + depot_distance * 0.1)

            # Mevcut rota yükü faktörü
            current_route_length = len(self.final_routes.get(vehicle.id, []))
            route_load_factor = 1.0 / (1.0 + current_route_length)

            # Final skor
            final_score = base_score * distance_factor * route_load_factor

            vehicle_scores.append((vehicle, final_score))

        if not vehicle_scores:
            return None

        return max(vehicle_scores, key=lambda x: x[1])[0]

    def _calculate_vehicle_suitability(self, emergency: FireEmergency, vehicle: EmergencyVehicle) -> float:
        """
        Acil durum için araç uygunluğunu hesapla
        """
        base_score = 0.0

        if vehicle.type == VehicleType.GROUND:
            # Kara araçları için uygunluk skorunu hesapla
            road_condition_factor = {
                RoadCondition.GOOD: 1.0,
                RoadCondition.MODERATE: 0.7,
                RoadCondition.BAD: 0.4
            }[emergency.road_condition]

            base_score = (
                    (10 - emergency.terrain_difficulty) * 0.3 +  # Düz arazi avantajı
                    (1.0 / (1.0 + emergency.distance_to_nearest_water)) * 0.3 +  # Su kaynağına yakınlık
                    road_condition_factor * 0.4  # Yol durumu etkisi
            )

        else:  # AERIAL
            # Hava araçları için uygunluk skorunu hesapla
            weather_condition_factor = {
                WeatherCondition.GOOD: 1.0,
                WeatherCondition.MODERATE: 0.7,
                WeatherCondition.BAD: 0.3
            }[emergency.weather_condition]

            base_score = (
                    (emergency.area_size / 10.0) * 0.3 +  # Geniş alan avantajı
                    (emergency.fire_intensity / 10.0) * 0.3 +  # Yüksek yoğunluk avantajı
                    weather_condition_factor * 0.4  # Hava durumu etkisi
            )

        return base_score

    def evolve(self) -> Dict[int, List[int]]:
        """
        Genetik algoritma ile rota optimizasyonu yapar.
        Büyük yangın bölgeleri için çift müdahale (kara+hava) desteği sağlar.
        """
        # Başlangıç popülasyonunu oluştur
        population = self.generate_initial_population()
        best_solution = None
        best_fitness = float('-inf')

        for generation in range(self.generations):
            # Uygunluk değerlerini hesapla
            for route in population:
                route.fitness = self.evaluate_fitness(route)

                # En iyi çözümü güncelle
                if route.fitness > best_fitness:
                    best_fitness = route.fitness
                    best_solution = route

            # Yeni popülasyon oluştur
            new_population = []
            elite_count = max(1, self.population_size // 10)

            # Elit bireyleri yeni popülasyona aktar
            sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
            new_population.extend(sorted_population[:elite_count])

            # Popülasyonun geri kalanını oluştur
            while len(new_population) < self.population_size:
                # Ebeveyn seçimi - turnuva seçimi
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Çaprazlama
                child = self.crossover(parent1, parent2)

                # Mutasyon
                self.mutate(child)

                new_population.append(child)

            population = new_population

        # En iyi çözümü rotalara dönüştür
        final_routes = {vehicle.id: [] for vehicle in self.vehicles}

        # Önce büyük yangınlar için çift müdahale ataması yap
        for emergency in self.emergencies:
            # Büyük yangın kriterleri kontrolü
            requires_dual_response = (
                    emergency.fire_intensity >= 7.0 or  # Yüksek yoğunluk
                    emergency.area_size >= 5.0 or  # Büyük alan
                    emergency.terrain_difficulty >= 7.0  # Zorlu arazi
            )

            if requires_dual_response:
                # Müsait kara ve hava araçlarını bul
                available_ground = [v for v in self.vehicles
                                    if v.type == VehicleType.GROUND and v.is_available]
                available_aerial = [v for v in self.vehicles
                                    if v.type == VehicleType.AERIAL and v.is_available]

                # Hava koşulları uygunsa çift müdahale yap
                if emergency.weather_condition != WeatherCondition.BAD and available_ground and available_aerial:
                    # En uygun kara aracını seç
                    ground_vehicle = self._select_best_vehicle(emergency, available_ground)
                    final_routes[ground_vehicle.id].append(emergency.id)
                    ground_vehicle.is_available = False

                    # En uygun hava aracını seç
                    aerial_vehicle = self._select_best_vehicle(emergency, available_aerial)
                    final_routes[aerial_vehicle.id].append(emergency.id)
                    aerial_vehicle.is_available = False

        # Kalan yangınlar için tek araç ataması yap
        remaining_emergencies = [e for e in self.emergencies
                                 if not any(e.id in route for route in final_routes.values())]

        for emergency in remaining_emergencies:
            # Müsait araçları bul
            available_vehicles = [v for v in self.vehicles if v.is_available]
            if not available_vehicles:
                continue

            # Bulanık mantık ile araç tipi belirle
            required_types = self.fuzzy_system.determine_required_vehicle_types(emergency)
            suitable_vehicles = [v for v in available_vehicles
                                 if v.type in required_types]

            if suitable_vehicles:
                # En uygun aracı seç
                best_vehicle = self._select_best_vehicle(emergency, suitable_vehicles)
                final_routes[best_vehicle.id].append(emergency.id)
                if len(final_routes[best_vehicle.id]) >= 3:  # Maksimum 3 görev kontrolü
                    best_vehicle.is_available = False

        return final_routes

    def _tournament_selection(self, population, tournament_size=3):
        """Turnuva seçimi ile ebeveyn seç"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def _select_best_vehicle(self, emergency: FireEmergency,
                             available_vehicles: List[EmergencyVehicle]) -> EmergencyVehicle:
        """
        Acil durum için en uygun aracı seç
        """
        vehicle_scores = []

        for vehicle in available_vehicles:
            # Temel uygunluk skorunu hesapla
            base_score = self.fuzzy_system._calculate_vehicle_suitability(emergency, vehicle)

            # Rota uzunluğu cezasını hesapla
            current_route_length = 0
            for vid, route in self.final_routes.items() if hasattr(self, 'final_routes') else {}.items():
                if vid == vehicle.id:
                    current_route_length = len(route)
                    break

            # Rota uzunluğu cezası - daha uzun rotalar için daha düşük skor
            route_penalty = 1.0 / (1.0 + current_route_length)

            # Mesafe faktörünü hesapla
            distance = self._calculate_aerial_distance(
                vehicle.current_lat,
                vehicle.current_lon,
                emergency.latitude,
                emergency.longitude
            )
            distance_penalty = 1.0 / (1.0 + distance * 0.1)

            # Kapasite faktörünü hesapla
            capacity_requirement = emergency.area_size * (emergency.fire_intensity / 10)
            capacity_score = min(1.0, vehicle.capacity / (capacity_requirement * 1000))

            # Final skoru hesapla
            final_score = base_score * route_penalty * distance_penalty * capacity_score

            vehicle_scores.append((vehicle, final_score))

        if not vehicle_scores:
            return None

        # En yüksek skorlu aracı döndür
        return max(vehicle_scores, key=lambda x: x[1])[0]

    def optimize_dual_response_routes(self, emergencies: List[FireEmergency]) -> Dict[int, List[FireEmergency]]:
        """
        Büyük yangın bölgeleri için hem kara hem hava aracı ataması yapan optimizasyon fonksiyonu
        """
        self.final_routes = {vehicle.id: [] for vehicle in self.vehicles}
        unassigned_emergencies = list(emergencies)

        # Yangınları öncelik sırasına göre sırala
        emergency_priorities = []
        for emergency in unassigned_emergencies:
            priority_score = (
                    emergency.fire_intensity * 0.4 +
                    emergency.area_size * 0.3 +
                    emergency.terrain_difficulty * 0.3
            )
            emergency_priorities.append((emergency, priority_score))

        emergency_priorities.sort(key=lambda x: x[1], reverse=True)

        # Her yangın için araç ataması yap
        for emergency, priority_score in emergency_priorities:
            requires_dual_response = (
                    emergency.fire_intensity >= 7.0 or
                    emergency.area_size >= 5.0 or
                    priority_score >= 7.0
            )

            available_ground_vehicles = [
                v for v in self.vehicles
                if v.type == VehicleType.GROUND and v.is_available and
                   len(self.final_routes[v.id]) < 3
            ]

            available_aerial_vehicles = [
                v for v in self.vehicles
                if v.type == VehicleType.AERIAL and v.is_available and
                   len(self.final_routes[v.id]) < 3
            ]

            if requires_dual_response and emergency.weather_condition != WeatherCondition.BAD:
                # Çift müdahale için araç seçimi
                if available_ground_vehicles and available_aerial_vehicles:
                    ground_vehicle = self._select_best_vehicle(emergency, available_ground_vehicles)
                    if ground_vehicle:
                        self.final_routes[ground_vehicle.id].append(emergency.id)
                        if len(self.final_routes[ground_vehicle.id]) >= 3:
                            ground_vehicle.is_available = False

                    aerial_vehicle = self._select_best_vehicle(emergency, available_aerial_vehicles)
                    if aerial_vehicle:
                        self.final_routes[aerial_vehicle.id].append(emergency.id)
                        if len(self.final_routes[aerial_vehicle.id]) >= 3:
                            aerial_vehicle.is_available = False
            else:
                # Tek araç ataması
                all_available = available_ground_vehicles + available_aerial_vehicles
                if all_available:
                    best_vehicle = self._select_best_vehicle(emergency, all_available)
                    if best_vehicle:
                        self.final_routes[best_vehicle.id].append(emergency.id)
                        if len(self.final_routes[best_vehicle.id]) >= 3:
                            best_vehicle.is_available = False

        return self.final_routes

class EmergencyInputFrame(tk.Frame):
    def __init__(self, parent, emergency_id):
        super().__init__(parent)
        self.emergency_id = emergency_id

        # Location
        tk.Label(self, text=f"Yangın {emergency_id} Konum (Enlem, Boylam):").grid(row=0, column=0)
        self.lat_entry = tk.Entry(self)
        self.lat_entry.grid(row=0, column=1)
        self.lon_entry = tk.Entry(self)
        self.lon_entry.grid(row=0, column=2)

        # Fire Intensity
        tk.Label(self, text="Yangın Yoğunluğu (0-10):").grid(row=1, column=0)
        self.intensity_scale = ttk.Scale(self, from_=0, to=10, orient='horizontal')
        self.intensity_scale.grid(row=1, column=1, columnspan=2, sticky='ew')

        # Area Size
        tk.Label(self, text="Alan Büyüklüğü (hektar):").grid(row=2, column=0)
        self.area_entry = tk.Entry(self)
        self.area_entry.grid(row=2, column=1)

        # Terrain Difficulty
        tk.Label(self, text="Arazi Zorluğu (0-10):").grid(row=3, column=0)
        self.terrain_scale = ttk.Scale(self, from_=0, to=10, orient='horizontal')
        self.terrain_scale.grid(row=3, column=1, columnspan=2, sticky='ew')

        # Weather Condition
        tk.Label(self, text="Hava Durumu:").grid(row=4, column=0)
        self.weather_var = tk.StringVar(value=WeatherCondition.GOOD.value)
        weather_options = ttk.OptionMenu(
            self, self.weather_var,
            WeatherCondition.GOOD.value,
            *[condition.value for condition in WeatherCondition]
        )
        weather_options.grid(row=4, column=1)

        # Road Condition
        tk.Label(self, text="Yol Durumu:").grid(row=5, column=0)
        self.road_var = tk.StringVar(value=RoadCondition.GOOD.value)
        road_options = ttk.OptionMenu(
            self, self.road_var,
            RoadCondition.GOOD.value,
            *[condition.value for condition in RoadCondition]
        )
        road_options.grid(row=5, column=1)

        # Water Distance
        tk.Label(self, text="En Yakın Su Kaynağı (km):").grid(row=6, column=0)
        self.water_distance_entry = tk.Entry(self)
        self.water_distance_entry.grid(row=6, column=1)

    def get_emergency_data(self) -> FireEmergency:
        try:
            return FireEmergency(
                id=self.emergency_id,
                latitude=float(self.lat_entry.get()),
                longitude=float(self.lon_entry.get()),
                fire_intensity=float(self.intensity_scale.get()),
                area_size=float(self.area_entry.get()),
                terrain_difficulty=float(self.terrain_scale.get()),
                weather_condition=WeatherCondition(self.weather_var.get()),
                road_condition=RoadCondition(self.road_var.get()),
                distance_to_nearest_water=float(self.water_distance_entry.get())
            )
        except ValueError as e:
            raise ValueError(f"Yangın {self.emergency_id} için geçersiz değerler: {str(e)}")

class FuzzyFireResponseSystem:
    def __init__(self):
        # Membership function ranges
        self.fire_intensity_ranges = {
            "low": (0, 0, 3, 4),
            "medium": (3, 4, 6, 7),
            "high": (6, 7, 10, 10)
        }

        self.area_size_ranges = {
            "small": (0, 0, 2, 3),
            "medium": (2, 3, 5, 6),
            "large": (5, 6, 10, 10)
        }

        self.terrain_ranges = {
            "easy": (0, 0, 3, 4),
            "moderate": (3, 4, 6, 7),
            "difficult": (6, 7, 10, 10)
        }

        # Thresholds for requiring both vehicle types
        self.dual_response_thresholds = {
            "fire_intensity": 7.0,  # Yüksek yoğunluklu yangınlar için çift müdahale
            "area_size": 5.0,  # Büyük alanlar için çift müdahale
            "terrain_difficulty": 7.0  # Zorlu arazi için çift müdahale
        }

    def determine_required_vehicle_types(self, emergency: FireEmergency) -> List[VehicleType]:
        """Bulanık mantık kurallarına göre gerekli araç tiplerini belirle"""
        # Üyelik değerlerini hesapla
        fire_high = self._calculate_membership(emergency.fire_intensity, self.fire_intensity_ranges["high"])
        area_large = self._calculate_membership(emergency.area_size, self.area_size_ranges["large"])
        terrain_diff = self._calculate_membership(emergency.terrain_difficulty, self.terrain_ranges["difficult"])

        # Çift müdahale gerektiren durumları kontrol et
        requires_dual_response = False

        # Yangın yoğunluğu kontrolü
        if emergency.fire_intensity >= self.dual_response_thresholds["fire_intensity"] or fire_high > 0.7:
            requires_dual_response = True

        # Alan büyüklüğü kontrolü
        if emergency.area_size >= self.dual_response_thresholds["area_size"] or area_large > 0.7:
            requires_dual_response = True

        # Hava ve yol durumu kontrolü
        weather_suitable = emergency.weather_condition != WeatherCondition.BAD
        road_suitable = emergency.road_condition != RoadCondition.BAD

        if requires_dual_response:
            # Her iki araç tipi de gerekli
            return [VehicleType.GROUND, VehicleType.AERIAL] if weather_suitable else [VehicleType.GROUND]
        else:
            # Tek araç tipi yeterli
            if not weather_suitable or not road_suitable:
                return [VehicleType.GROUND if not weather_suitable else VehicleType.AERIAL]
            else:
                # Standart durumlarda tercih edilen araç tipini belirle
                primary_type = VehicleType.AERIAL if fire_high > 0.5 or area_large > 0.5 else VehicleType.GROUND
                return [primary_type]
    def _calculate_membership(self, x: float, trapezoid: Tuple[float, float, float, float]) -> float:
        """Calculate membership value using trapezoidal membership function"""
        a, b, c, d = trapezoid
        if x <= a or x >= d:
            return 0
        elif b <= x <= c:
            return 1
        elif a < x < b:
            return (x - a) / (b - a)
        else:  # c < x < d
            return (d - x) / (d - c)

    def _calculate_vehicle_suitability(self, emergency: FireEmergency, vehicle: EmergencyVehicle) -> float:
        """Calculate suitability score for a vehicle based on emergency conditions"""
        # Fire intensity membership
        fire_low = self._calculate_membership(emergency.fire_intensity, self.fire_intensity_ranges["low"])
        fire_med = self._calculate_membership(emergency.fire_intensity, self.fire_intensity_ranges["medium"])
        fire_high = self._calculate_membership(emergency.fire_intensity, self.fire_intensity_ranges["high"])

        # Area size membership
        area_small = self._calculate_membership(emergency.area_size, self.area_size_ranges["small"])
        area_med = self._calculate_membership(emergency.area_size, self.area_size_ranges["medium"])
        area_large = self._calculate_membership(emergency.area_size, self.area_size_ranges["large"])

        # Terrain difficulty membership
        terrain_easy = self._calculate_membership(emergency.terrain_difficulty, self.terrain_ranges["easy"])
        terrain_mod = self._calculate_membership(emergency.terrain_difficulty, self.terrain_ranges["moderate"])
        terrain_diff = self._calculate_membership(emergency.terrain_difficulty, self.terrain_ranges["difficult"])

        # Weather and road condition factors
        weather_factor = {
            WeatherCondition.GOOD: 1.0,
            WeatherCondition.MODERATE: 0.7,
            WeatherCondition.BAD: 0.3
        }[emergency.weather_condition]

        road_factor = {
            RoadCondition.GOOD: 1.0,
            RoadCondition.MODERATE: 0.7,
            RoadCondition.BAD: 0.3
        }[emergency.road_condition]

        # Calculate base suitability for each vehicle type
        if vehicle.type == VehicleType.GROUND:
            suitability = (
                              # Ground vehicles are better for smaller fires
                                  (fire_low * 0.9 + fire_med * 0.6 + fire_high * 0.3) +
                                  # Ground vehicles are better for smaller areas
                                  (area_small * 0.9 + area_med * 0.6 + area_large * 0.3) +
                                  # Ground vehicles prefer easier terrain
                                  (terrain_easy * 0.9 + terrain_mod * 0.6 + terrain_diff * 0.2)
                          ) / 3.0 * road_factor

        else:  # AERIAL
            suitability = (
                              # Aerial vehicles are better for larger fires
                                  (fire_low * 0.3 + fire_med * 0.7 + fire_high * 0.9) +
                                  # Aerial vehicles are better for larger areas
                                  (area_small * 0.3 + area_med * 0.7 + area_large * 0.9) +
                                  # Aerial vehicles are less affected by terrain
                                  (terrain_easy * 0.7 + terrain_mod * 0.7 + terrain_diff * 0.7)
                          ) / 3.0 * weather_factor

        # Adjust for water source distance
        water_distance_factor = 1.0 / (1.0 + emergency.distance_to_nearest_water * 0.1)
        suitability *= water_distance_factor

        return suitability

    def select_vehicles_for_emergency(self, emergency: FireEmergency,
                                      available_vehicles: List[EmergencyVehicle]) -> List[EmergencyVehicle]:
        """Acil durum için en uygun araçları seç"""
        required_vehicle_types = self.determine_required_vehicle_types(emergency)
        selected_vehicles = []

        for vehicle_type in required_vehicle_types:
            # İlgili tipteki müsait araçları filtrele
            suitable_vehicles = [
                vehicle for vehicle in available_vehicles
                if vehicle.type == vehicle_type and vehicle.is_available and vehicle not in selected_vehicles
            ]

            if suitable_vehicles:
                # En uygun aracı seç
                best_vehicle = max(
                    suitable_vehicles,
                    key=lambda v: self._calculate_vehicle_suitability(emergency, v)
                )
                selected_vehicles.append(best_vehicle)

        if not selected_vehicles and available_vehicles:
            # Eğer tercih edilen tipte araç yoksa, müsait olan herhangi bir aracı kullan
            best_alternative = max(
                [v for v in available_vehicles if v.is_available],
                key=lambda v: self._calculate_vehicle_suitability(emergency, v)
            )
            selected_vehicles.append(best_alternative)

        return selected_vehicles

class EmergencyRoutingSolverApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.root.title("Acil Durum Araç Rotalama Sistemi")
        self.fuzzy_system = FuzzyFireResponseSystem()

        # Depoları tanımla
        self.depots = {
            1: {"lat": 37.7640, "lon": 30.5458},  # İstanbul deposu
            2: {"lat": 37.6255, "lon": 30.5362},  # Ankara deposu
            3: {"lat": 38.4237, "lon": 30.5622}  # İzmir deposu
        }

        self.vehicles = self._initialize_vehicles()

        # Main container
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)



        # Depot location frame
        self.depot_frame = tk.Frame(self.main_frame)
        self.depot_frame.pack(fill='x', pady=5)

        tk.Label(self.depot_frame, text="Depolar:").pack(side='left')

        # Her depo için konum gösterimi
        for depot_id, location in self.depots.items():
            depot_label = tk.Label(
                self.depot_frame,
                text=f"Depo {depot_id}: ({location['lat']}, {location['lon']})"
            )
            depot_label.pack(side='left', padx=10)


        # Emergency inputs container
        self.emergencies_frame = tk.Frame(self.main_frame)
        self.emergencies_frame.pack(fill='x', pady=5)
        self.emergency_frames = []

        # Buttons
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(fill='x', pady=5)

        tk.Button(self.button_frame, text="Yangın Noktası Ekle",
                  command=self.add_emergency).pack(side='left', padx=5)
        tk.Button(self.button_frame, text="Rotaları Hesapla",
                  command=self.solve_routing).pack(side='left', padx=5)

    def _initialize_vehicles(self) -> List[EmergencyVehicle]:
        """Çoklu depo için araç filosu oluştur"""
        vehicles = []

        # Her depo için araç oluştur
        vehicle_configs = {
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

        # Her depo için araçları oluştur
        for depot_id, configs in vehicle_configs.items():
            depot = self.depots[depot_id]
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
    def create_route_report(self, emergencies: List[FireEmergency], routes: Dict[int, List[int]]):
        report = f"""
        # Acil Durum Müdahale Raporu
        Oluşturma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        ## Yangın Noktaları
        """

        for emergency in emergencies:
            report += f"""
        ### Yangın {emergency.id}
        - Konum: {emergency.latitude}, {emergency.longitude}
        - Yangın Yoğunluğu: {emergency.fire_intensity}/10
        - Alan Büyüklüğü: {emergency.area_size} hektar
        - Arazi Zorluğu: {emergency.terrain_difficulty}/10
        - Hava Durumu: {emergency.weather_condition.value}
        - Yol Durumu: {emergency.road_condition.value}
        """

        report += "\n## Araç Rotaları\n"

        for vehicle in self.vehicles:
            if vehicle.id in routes and routes[vehicle.id]:
                emergency_ids = routes[vehicle.id]
                report += f"""
        ### Araç {vehicle.id} ({vehicle.type.value})
        - Kapasite: {vehicle.capacity}L
        - Hız: {vehicle.speed} km/h
        - Atanan Yangınlar: {', '.join(map(str, emergency_ids))}
        """

        with open('emergency_route_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

    def add_emergency(self):
        emergency_frame = EmergencyInputFrame(
            self.emergencies_frame,
            len(self.emergency_frames) + 1
        )
        emergency_frame.pack(fill='x', pady=5)
        self.emergency_frames.append(emergency_frame)

    def optimize_routes(self, emergencies: List[FireEmergency]) -> Dict[int, List[FireEmergency]]:
        """Çoklu yangın ve çoklu araç desteği ile rota optimizasyonu"""
        routes = {vehicle.id: [] for vehicle in self.vehicles}
        unassigned_emergencies = list(emergencies)
        fuzzy_system = FuzzyFireResponseSystem()

        # Yangınları öncelik sırasına göre sırala
        emergency_priorities = []
        for emergency in unassigned_emergencies:
            priority_score = (
                    emergency.fire_intensity * 0.4 +
                    emergency.area_size * 0.3 +
                    emergency.terrain_difficulty * 0.3
            )
            emergency_priorities.append((emergency, priority_score))

        emergency_priorities.sort(key=lambda x: x[1], reverse=True)

        # Her yangın için araç ataması yap
        for emergency, _ in emergency_priorities:
            available_vehicles = [
                v for v in self.vehicles
                if v.is_available and len(routes[v.id]) < 3  # Bir araca maksimum 3 görev
            ]

            selected_vehicles = fuzzy_system.select_vehicles_for_emergency(
                emergency,
                available_vehicles
            )

            # Seçilen araçları yangına ata
            for vehicle in selected_vehicles:
                routes[vehicle.id].append(emergency)

                # Aracın müsaitlik durumunu güncelle
                if len(routes[vehicle.id]) >= 3:
                    vehicle.is_available = False

        return routes

    def plot_emergency_routes(self, emergencies: List[FireEmergency], routes: Dict[int, List[int]]):
        """Çoklu depo için acil durum rotalarını haritada göster"""
        # Haritanın merkezi için tüm koordinatların ortalamasını al
        all_lats = [e.latitude for e in emergencies]
        all_lons = [e.longitude for e in emergencies]
        all_lats.extend([d['lat'] for d in self.depots.values()])
        all_lons.extend([d['lon'] for d in self.depots.values()])

        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Renk şeması
        colors = {
            VehicleType.GROUND: 'red',
            VehicleType.AERIAL: 'blue'
        }

        # Depo simgeleri için farklı renkler
        depot_colors = ['green', 'darkgreen', 'lightgreen']

        # Tüm depoları işaretle
        for depot_id, location in self.depots.items():
            folium.Marker(
                [location['lat'], location['lon']],
                popup=f'Depo {depot_id}',
                icon=folium.Icon(
                    color=depot_colors[(depot_id - 1) % len(depot_colors)],
                    icon='home'
                )
            ).add_to(m)

        # Her araç için rotaları çiz
        for vehicle in self.vehicles:
            if vehicle.id not in routes or not routes[vehicle.id]:
                continue

            color = colors[vehicle.type]
            depot = self.depots[vehicle.depot_id]
            current_lat, current_lon = depot['lat'], depot['lon']

            for emergency_id in routes[vehicle.id]:
                emergency = next(e for e in emergencies if e.id == emergency_id)

                # Yangın noktasını işaretle
                folium.Marker(
                    [emergency.latitude, emergency.longitude],
                    popup=f'Yangın {emergency.id}\n'
                          f'Araç: {vehicle.id} ({vehicle.type.value})\n'
                          f'Depo: {vehicle.depot_id}',
                    icon=folium.Icon(color='orange', icon='fire')
                ).add_to(m)

                # Araç tipine göre rota çiz
                if vehicle.type == VehicleType.AERIAL:
                    # Hava aracı için düz çizgi
                    folium.PolyLine(
                        [(current_lat, current_lon),
                         (emergency.latitude, emergency.longitude)],
                        color=color,
                        weight=3,
                        opacity=0.8,
                        dash_array='10',
                        popup=f'Hava Aracı {vehicle.id} (Depo {vehicle.depot_id})'
                    ).add_to(m)
                else:
                    # Kara aracı için OSRM rotası
                    try:
                        route_points = get_osrm_route_geometry(
                            current_lat, current_lon,
                            emergency.latitude, emergency.longitude
                        )
                        folium.PolyLine(
                            route_points,
                            color=color,
                            weight=3,
                            opacity=0.8,
                            popup=f'Kara Aracı {vehicle.id} (Depo {vehicle.depot_id})'
                        ).add_to(m)
                    except Exception as e:
                        print(f"OSRM error for vehicle {vehicle.id}: {e}")

                current_lat, current_lon = emergency.latitude, emergency.longitude

            # Depoya dönüş rotası
            if vehicle.type == VehicleType.AERIAL:
                folium.PolyLine(
                    [(current_lat, current_lon), (depot['lat'], depot['lon'])],
                    color=color,
                    weight=3,
                    opacity=0.8,
                    dash_array='10',
                    popup=f'Hava Aracı {vehicle.id} Dönüş (Depo {vehicle.depot_id})'
                ).add_to(m)
            else:
                try:
                    return_points = get_osrm_route_geometry(
                        current_lat, current_lon,
                        depot['lat'], depot['lon']
                    )
                    folium.PolyLine(
                        return_points,
                        color=color,
                        weight=3,
                        opacity=0.8,
                        popup=f'Kara Aracı {vehicle.id} Dönüş (Depo {vehicle.depot_id})'
                    ).add_to(m)
                except Exception as e:
                    print(f"OSRM error for vehicle {vehicle.id} return route: {e}")

        # Harita sınırlarını otomatik ayarla
        m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

        m.save('emergency_routes.html')
        webbrowser.open('emergency_routes.html')
    def get_distance_matrix(self, locations):
        n = len(locations)
        matrix = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = get_osrm_distance(
                        locations[i][0], locations[i][1],
                        locations[j][0], locations[j][1]
                    )
        return matrix

    def solve_routing(self):
        try:
            # Acil durumları topla
            emergencies = []
            for frame in self.emergency_frames:
                emergencies.append(frame.get_emergency_data())

            if not emergencies:
                messagebox.showerror("Hata", "Lütfen en az bir yangın noktası ekleyin")
                return

            # Tüm depoları Depot nesnelerine dönüştür
            depot_objects = []
            for depot_id, depot_location in self.depots.items():
                depot_objects.append(
                    Depot(
                        id=depot_id,
                        latitude=depot_location['lat'],
                        longitude=depot_location['lon'],
                        name=f"Depot {depot_id}",
                        vehicle_capacity=len([v for v in self.vehicles if v.depot_id == depot_id])
                    )
                )

            # Genetik algoritma için parametreleri hazırla
            ga = EmergencyGeneticAlgorithm(
                depots=depot_objects,
                emergencies=emergencies,
                vehicles=self.vehicles,
                population_size=50,
                generations=100,
                crossover_rate=0.9,
                mutation_rate=0.35
            )

            # Tüm depolar için rotaları optimize et
            all_routes = ga.evolve()

            # Sonuçları görselleştir ve raporla
            self.plot_emergency_routes(emergencies, all_routes)
            self.create_route_report(emergencies, all_routes)

            # Başarı mesajı göster
            total_assigned = sum(len(route) for route in all_routes.values())
            total_emergencies = len(emergencies)
            total_unassigned = total_emergencies - total_assigned

            status_message = (
                f"Optimum rotalar hesaplandı.\n"
                f"Toplam {total_emergencies} yangın noktasından:\n"
                f"- {total_assigned} nokta için araç görevlendirildi\n"
                f"- {total_unassigned} nokta atanmadı\n"
                f"Harita ve detaylı rapor tarayıcınızda açılacak."
            )

            messagebox.showinfo("Başarılı", status_message)

        except ValueError as e:
            messagebox.showerror("Hata", f"Geçersiz giriş: {str(e)}")
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {str(e)}")
            raise
    def calculate_route_distance(self, start_loc: Tuple[float, float],
                                 end_loc: Tuple[float, float],
                                 vehicle_type: VehicleType) -> float:
        """Araç tipine göre mesafe hesaplama"""
        if vehicle_type == VehicleType.AERIAL:
            # Hava araçları için kuş uçuşu mesafe (Haversine formülü)
            lat1, lon1 = start_loc
            lat2, lon2 = end_loc

            R = 6371  # Dünya yarıçapı (km)

            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            distance = R * c

            return distance
        else:
            # Kara araçları için OSRM ile karayolu mesafesi
            return get_osrm_distance(
                start_loc[0], start_loc[1],
                end_loc[0], end_loc[1]
            )

    def get_route_points(self, start_loc: Tuple[float, float],
                         end_loc: Tuple[float, float],
                         vehicle_type: VehicleType) -> List[Tuple[float, float]]:
        """Araç tipine göre rota noktalarını hesapla"""
        if vehicle_type == VehicleType.AERIAL:
            # Hava araçları için düz hat
            return [start_loc, end_loc]
        else:
            # Kara araçları için OSRM ile karayolu rotası
            return get_osrm_route_geometry(
                start_loc[0], start_loc[1],
                end_loc[0], end_loc[1]
            )

def get_osrm_route_geometry(lat1, lon1, lat2, lon2, osrm_url="http://router.project-osrm.org/route/v1/driving/"):
    url = f"{osrm_url}{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    try:
        response = requests.get(url)
        data = response.json()
        geometry = data['routes'][0]['geometry']['coordinates']
        return [(lat, lon) for lon, lat in geometry]
    except Exception as e:
        print(f"OSRM request error: {e}")
        return [(lat1, lon1), (lat2, lon2)]  # Fallback to straight line


def get_osrm_distance(lat1, lon1, lat2, lon2, osrm_url="http://router.project-osrm.org/route/v1/driving/"):
    url = f"{osrm_url}{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        response = requests.get(url)
        data = response.json()
        distance = data['routes'][0]['distance'] / 1000
        return distance
    except Exception as e:
        print(f"OSRM request error: {e}")
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111  # Rough approximation in km


if __name__ == "__main__":
    root = tk.Tk()
    app = EmergencyRoutingSolverApp(root)
    root.mainloop()