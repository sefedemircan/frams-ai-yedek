import math
import random
import requests
from dataclasses import dataclass, field
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

class EmergencyGeneticAlgorithm:
    def __init__(self,
                 depots: List[Depot],
                 emergencies: List[FireEmergency],
                 vehicles: List[EmergencyVehicle],
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.35,
                 consider_traffic: bool = True):

        self.depots = {depot.id: depot for depot in depots}
        self.emergencies = emergencies
        self.vehicles = vehicles
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fuzzy_system = FuzzyFireResponseSystem()
        self.consider_traffic = consider_traffic  # Trafik faktörünü dikkate alma

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
                    # Trafiği dikkate alan mesafe hesaplaması
                    ground_dist = get_osrm_distance(lat1, lon1, lat2, lon2, traffic=self.consider_traffic)
                    self.distance_cache[f'ground_{id1}_{id2}'] = ground_dist
                    self.distance_cache[f'ground_{id2}_{id1}'] = ground_dist
                    
                    # Trafik bilgisini içeren rota geometrisini de sakla (harita gösterimi için)
                    route_geometry = get_osrm_route_geometry(lat1, lon1, lat2, lon2, traffic=self.consider_traffic)
                    self.distance_cache[f'geometry_ground_{id1}_{id2}'] = route_geometry
                    self.distance_cache[f'geometry_ground_{id2}_{id1}'] = list(reversed(route_geometry))
                except Exception as e:
                    print(f"OSRM error for points {id1}-{id2}: {e}")
                    self.distance_cache[f'ground_{id1}_{id2}'] = aerial_dist * 1.3
                    self.distance_cache[f'ground_{id2}_{id1}'] = aerial_dist * 1.3
                    # Hata durumunda düz çizgi geometrisi kullan
                    self.distance_cache[f'geometry_ground_{id1}_{id2}'] = [(lat1, lon1), (lat2, lon2)]
                    self.distance_cache[f'geometry_ground_{id2}_{id1}'] = [(lat2, lon2), (lat1, lon1)]

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

    def get_route_geometry(self, point1_id: Union[int, str], point2_id: Union[int, str],
                           vehicle_type: VehicleType) -> List[Tuple[float, float]]:
        """İki nokta arasındaki rotanın geometrisini döndürür"""
        cache_key = f'geometry_{vehicle_type.value}_{point1_id}_{point2_id}'
        
        if vehicle_type == VehicleType.GROUND and cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Eğer yer imi yoksa veya hava aracı ise, düz çizgi döndür
        if vehicle_type == VehicleType.GROUND:
            # Nokta koordinatlarını bul
            point1_coords = None
            point2_coords = None
            
            # Birinci nokta
            if isinstance(point1_id, str) and point1_id.startswith('depot_'):
                depot_id = int(point1_id.split('_')[1])
                depot = self.depots.get(depot_id)
                if depot:
                    point1_coords = (depot.latitude, depot.longitude)
            else:
                emergency = next((e for e in self.emergencies if e.id == point1_id), None)
                if emergency:
                    point1_coords = (emergency.latitude, emergency.longitude)
            
            # İkinci nokta
            if isinstance(point2_id, str) and point2_id.startswith('depot_'):
                depot_id = int(point2_id.split('_')[1])
                depot = self.depots.get(depot_id)
                if depot:
                    point2_coords = (depot.latitude, depot.longitude)
            else:
                emergency = next((e for e in self.emergencies if e.id == point2_id), None)
                if emergency:
                    point2_coords = (emergency.latitude, emergency.longitude)
            
            # Her iki noktanın koordinatları varsa
            if point1_coords and point2_coords:
                try:
                    # Trafik bilgisini içeren rota hesaplama
                    geometry = get_osrm_route_geometry(
                        point1_coords[0], point1_coords[1],
                        point2_coords[0], point2_coords[1],
                        traffic=self.consider_traffic
                    )
                    # Önbelleğe kaydet
                    self.distance_cache[cache_key] = geometry
                    return geometry
                except Exception as e:
                    print(f"Route geometry error: {e}")
            
            # Hata durumunda düz çizgi
            return [point1_coords, point2_coords] if point1_coords and point2_coords else []
        else:
            # Hava aracı için düz çizgi
            # Nokta koordinatlarını bul
            point1_coords = None
            point2_coords = None
            
            # Yukarıdaki gibi koordinatları bul
            # (Kodu kısaltmak için benzer işlem tekrarı)
            
            return [point1_coords, point2_coords] if point1_coords and point2_coords else []

    def evolve(self) -> Dict[int, List[int]]:
        """
        Genetik algoritma ile rota optimizasyonu yapar.
        Büyük yangın bölgeleri için çift müdahale (kara+hava) desteği sağlar.
        Trafik durumunu da dikkate alır.
        """
        # Başlangıç popülasyonunu oluştur
        population = self.generate_initial_population()
        best_solution = None
        best_fitness = float('-inf')
        
        # Optimizasyon bilgisi
        if self.consider_traffic:
            print("Trafik bilgisi dikkate alınarak optimizasyon yapılıyor...")
        else:
            print("Trafik bilgisi dikkate alınmadan optimizasyon yapılıyor...")
        
        # Rota sonuçları için hazırlık
        self.final_routes = {vehicle.id: [] for vehicle in self.vehicles}

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
                    if ground_vehicle:
                        self.final_routes[ground_vehicle.id].append(emergency.id)
                        if len(self.final_routes[ground_vehicle.id]) >= 3:
                            ground_vehicle.is_available = False

                    # En uygun hava aracını seç
                    aerial_vehicle = self._select_best_vehicle(emergency, available_aerial)
                    if aerial_vehicle:
                        self.final_routes[aerial_vehicle.id].append(emergency.id)
                        if len(self.final_routes[aerial_vehicle.id]) >= 3:
                            aerial_vehicle.is_available = False

        # Kalan yangınlar için tek araç ataması yap
        remaining_emergencies = [e for e in self.emergencies
                                 if not any(e.id in route for route in self.final_routes.values())]

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
                if best_vehicle:
                    self.final_routes[best_vehicle.id].append(emergency.id)
                    if len(self.final_routes[best_vehicle.id]) >= 3:  # Maksimum 3 görev kontrolü
                        best_vehicle.is_available = False

        return self.final_routes

def get_osrm_distance(lat1, lon1, lat2, lon2, osrm_url="http://router.project-osrm.org/route/v1/driving/", traffic=False):
    """Verilen iki nokta arasındaki mesafeyi OSRM servisi üzerinden hesaplar.
    
    Args:
        lat1, lon1: Başlangıç noktasının koordinatları
        lat2, lon2: Bitiş noktasının koordinatları
        osrm_url: OSRM API URL'i
        traffic: Trafik durumunu dikkate al (True/False)
    
    Returns:
        float: Mesafe (km)
    """
    # Trafik durumunu belirten parametreler
    params = "?overview=false"
    if traffic:
        # OSRM'in trafik farkındalığı için ekstra parametreleri (etkin ise)
        # Duration tahmini trafik durumunu etkileyecek şekilde hesaplanır
        params += "&annotations=true&steps=true&geometries=polyline&generate_hints=false"
    
    url = f"{osrm_url}{lon1},{lat1};{lon2},{lat2}{params}"
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'routes' not in data or not data['routes']:
            print(f"OSRM response error: {data.get('message', 'Unknown error')}")
            return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111  # Fallback

        # Mesafe (metre) -> km
        distance = data['routes'][0]['distance'] / 1000
        
        # Trafik durumunu dikkate alan ceza faktörü
        if traffic and 'duration' in data['routes'][0]:
            # Süre / mesafe oranı yüksekse trafik var demektir
            # Bu varsayımsal bir formüldür, gerçek veriler üzerine iyileştirilebilir
            duration = data['routes'][0]['duration']  # saniye
            avg_speed = (distance * 1000) / duration if duration > 0 else 50  # m/s
            
            # Ortalama hız düşükse trafik yoğun demektir
            # Trafik yoğunluğuna göre mesafe cezası
            if avg_speed < 5:  # ~18 km/saat (çok yoğun trafik)
                distance *= 1.5
            elif avg_speed < 10:  # ~36 km/saat (yoğun trafik)
                distance *= 1.3
            elif avg_speed < 15:  # ~54 km/saat (orta trafik)
                distance *= 1.1
        
        return distance
    except Exception as e:
        print(f"OSRM request error: {e}")
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111  # Rough approximation in km

def get_osrm_route_geometry(lat1, lon1, lat2, lon2, osrm_url="http://router.project-osrm.org/route/v1/driving/", traffic=False):
    """İki nokta arasındaki rotanın geometrisini OSRM servisi üzerinden alır.
    
    Args:
        lat1, lon1: Başlangıç noktasının koordinatları
        lat2, lon2: Bitiş noktasının koordinatları
        osrm_url: OSRM API URL'i
        traffic: Trafik durumunu dikkate al (True/False)
    
    Returns:
        list: [(lat, lon), ...] şeklinde rota koordinatları
    """
    # Trafik durumunu belirten parametreler
    params = "?overview=full&geometries=geojson"
    if traffic:
        params += "&annotations=true&steps=true&generate_hints=false"
    
    url = f"{osrm_url}{lon1},{lat1};{lon2},{lat2}{params}"
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'routes' not in data or not data['routes']:
            print(f"OSRM geometry response error: {data.get('message', 'Unknown error')}")
            return [(lat1, lon1), (lat2, lon2)]  # Fallback
            
        geometry = data['routes'][0]['geometry']['coordinates']
        
        # Trafik bilgisini ekleyen veri analizi (isteğe bağlı)
        if traffic and 'legs' in data['routes'][0] and data['routes'][0]['legs']:
            leg = data['routes'][0]['legs'][0]
            if 'annotation' in leg:
                # OSRM tarafından sağlanan trafik verileri
                # Örneğin: hız, yoğunluk vb.
                # Bu verileri loglama veya analiz için kullanabiliriz
                print(f"Route traffic information available: {len(leg['annotation'].get('speed', []))} speed points")
        
        return [(lat, lon) for lon, lat in geometry]
    except Exception as e:
        print(f"OSRM route geometry request error: {e}")
        return [(lat1, lon1), (lat2, lon2)]  # Fallback to straight line 