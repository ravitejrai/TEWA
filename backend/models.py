"""
Data models for the Air Defense C2 System.
All positions use (lat, lon) in decimal degrees.
"""
from __future__ import annotations
import uuid
import math
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class ThreatClass(str, Enum):
    DRONE               = "drone"               # slow, low-altitude, cheap
    CRUISE              = "cruise"              # medium speed, low altitude
    FIGHTER             = "fighter"             # fast, maneuverable
    BALLISTIC           = "ballistic"           # very fast, high altitude
    LOITERING_MUNITION  = "loitering_munition" # slow, persistent, hard to detect


class ThreatStatus(str, Enum):
    ACTIVE      = "active"
    ENGAGED     = "engaged"
    NEUTRALIZED = "neutralized"
    ESCAPED     = "escaped"


class EffectorType(str, Enum):
    MISSILE          = "missile"          # long-range, high PK vs fast targets
    ANTI_AIR         = "anti_air"         # medium-range, cost-efficient
    DRONE            = "drone"            # loitering, reusable (patrol radius)
    AIRCRAFT         = "aircraft"         # highest flexibility, refuelable
    KINETIC_MISSILE  = "kinetic_missile"  # PAC-3 MSE – high-altitude intercept, salvo-fire
    DIRECTED_ENERGY  = "directed_energy"  # HELIOS-300 – near-unlimited shots, weather-limited


class AssetStatus(str, Enum):
    READY       = "ready"
    ENGAGED     = "engaged"
    RELOADING   = "reloading"
    DESTROYED   = "destroyed"


# Probability of kill matrix  [effector → threat_class]
# kinetic_missile and directed_energy base_pk values sourced from Gemini effector spec
PK_TABLE: dict[EffectorType, dict[ThreatClass, float]] = {
    EffectorType.MISSILE:         {ThreatClass.DRONE: 0.70, ThreatClass.CRUISE: 0.90, ThreatClass.FIGHTER: 0.85, ThreatClass.BALLISTIC: 0.80, ThreatClass.LOITERING_MUNITION: 0.60},
    EffectorType.ANTI_AIR:        {ThreatClass.DRONE: 0.85, ThreatClass.CRUISE: 0.75, ThreatClass.FIGHTER: 0.55, ThreatClass.BALLISTIC: 0.30, ThreatClass.LOITERING_MUNITION: 0.80},
    EffectorType.DRONE:           {ThreatClass.DRONE: 0.80, ThreatClass.CRUISE: 0.50, ThreatClass.FIGHTER: 0.25, ThreatClass.BALLISTIC: 0.05, ThreatClass.LOITERING_MUNITION: 0.75},
    EffectorType.AIRCRAFT:        {ThreatClass.DRONE: 0.90, ThreatClass.CRUISE: 0.88, ThreatClass.FIGHTER: 0.80, ThreatClass.BALLISTIC: 0.40, ThreatClass.LOITERING_MUNITION: 0.85},
    # Gemini spec: KIN-PAC3-MSE base_pk=0.85, optimal vs ballistic/supersonic_cruise/fighter
    EffectorType.KINETIC_MISSILE: {ThreatClass.DRONE: 0.55, ThreatClass.CRUISE: 0.88, ThreatClass.FIGHTER: 0.85, ThreatClass.BALLISTIC: 0.85, ThreatClass.LOITERING_MUNITION: 0.60},
    # Gemini spec: DEW-HELIOS-300 base_pk=0.99, optimal vs drone_swarm/loitering_munition/subsonic_cruise; weather_degradation=true
    EffectorType.DIRECTED_ENERGY: {ThreatClass.DRONE: 0.97, ThreatClass.CRUISE: 0.90, ThreatClass.FIGHTER: 0.15, ThreatClass.BALLISTIC: 0.05, ThreatClass.LOITERING_MUNITION: 0.99},
}

# Reload time in simulation ticks (each tick = 5 seconds)
RELOAD_TICKS: dict[EffectorType, int] = {
    EffectorType.MISSILE:         8,
    EffectorType.ANTI_AIR:        4,
    EffectorType.DRONE:           0,   # reusable
    EffectorType.AIRCRAFT:        5,   # ~25s sim between sorties
    EffectorType.KINETIC_MISSILE: 6,   # ~30s between salvos
    EffectorType.DIRECTED_ENERGY: 1,   # 5000ms cooldown ≈ 1 tick
}

# Speed in km/h
THREAT_SPEED: dict[ThreatClass, float] = {
    ThreatClass.DRONE:              120,
    ThreatClass.CRUISE:             850,
    ThreatClass.FIGHTER:           1800,
    ThreatClass.BALLISTIC:         5000,
    ThreatClass.LOITERING_MUNITION: 180,   # slow, persistent
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns distance in kilometers between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class Threat:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str = ""
    threat_class: ThreatClass = ThreatClass.DRONE
    lat: float = 0.0
    lon: float = 0.0
    heading: float = 180.0      # degrees true north
    speed_kmh: float = 120.0
    altitude_m: float = 500.0
    priority: int = 5           # 1 (low) – 10 (critical)
    status: ThreatStatus = ThreatStatus.ACTIVE

    # Internal tracking
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    target_lat: float = 59.9    # default target: Stockholm region
    target_lon: float = 18.07
    target_label: str = ""     # human-readable target city name

    # ── Kalman filter sensor-fusion fields (not serialised to JSON) ──
    kalman_filter: Any = field(default=None, repr=False, compare=False)

    # KF output fields (updated each tick, exposed in to_dict)
    track_quality: float = 0.0        # 0.0–1.0 track confidence
    estimated_lat: float = 0.0        # Kalman-filtered position
    estimated_lon: float = 0.0
    estimated_speed_kmh_kf: float = 0.0   # KF-derived speed
    radar_lat: float = 0.0            # raw (noisy) radar return
    radar_lon: float = 0.0
    position_uncertainty_km: float = 5.0  # 1-sigma position uncertainty

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "threat_class": self.threat_class.value,
            "lat": round(self.lat, 5),
            "lon": round(self.lon, 5),
            "heading": round(self.heading, 1),
            "speed_kmh": self.speed_kmh,
            "altitude_m": self.altitude_m,
            "priority": self.priority,
            "status": self.status.value,
            # Kalman filter outputs
            "track_quality": round(self.track_quality, 3),
            "estimated_lat": round(self.estimated_lat, 5) if self.estimated_lat else None,
            "estimated_lon": round(self.estimated_lon, 5) if self.estimated_lon else None,
            "estimated_speed_kmh_kf": round(self.estimated_speed_kmh_kf, 1),
            "radar_lat": round(self.radar_lat, 5) if self.radar_lat else None,
            "radar_lon": round(self.radar_lon, 5) if self.radar_lon else None,
            "position_uncertainty_km": round(self.position_uncertainty_km, 2),
            "target_lat": round(self.target_lat, 5),
            "target_lon": round(self.target_lon, 5),
            "target_label": self.target_label,
        }

    def time_to_target_s(self) -> float:
        dist_km = haversine(self.lat, self.lon, self.target_lat, self.target_lon)
        return (dist_km / max(self.speed_kmh, 1)) * 3600

    def move(self, seconds: float = 5.0):
        """Advance threat position toward target."""
        if self.status not in (ThreatStatus.ACTIVE, ThreatStatus.ENGAGED):
            return
        dist_km = (self.speed_kmh / 3600) * seconds
        # Simple flat-Earth movement approximation
        dlat = math.cos(math.radians(self.heading)) * dist_km / 111.0
        dlon = math.sin(math.radians(self.heading)) * dist_km / (111.0 * math.cos(math.radians(self.lat + 0.001)))
        self.lat += dlat
        self.lon += dlon
        # Recalculate heading toward target
        dy = self.target_lat - self.lat
        dx = (self.target_lon - self.lon) * math.cos(math.radians(self.lat))
        self.heading = math.degrees(math.atan2(dx, dy)) % 360


@dataclass
class Effector:
    """A single effector unit (missile battery, AA gun, drone, interceptor)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str = ""
    effector_type: EffectorType = EffectorType.MISSILE
    lat: float = 0.0
    lon: float = 0.0
    ammo: int = 10               # -1 means unlimited
    max_ammo: int = 10
    range_km: float = 150.0
    status: AssetStatus = AssetStatus.READY
    reload_ticks_remaining: int = 0
    base_id: str = ""
    assigned_threat_id: Optional[str] = None
    # --- Fields from Gemini effector spec ---
    cost_per_shot_usd: float = 0.0          # cost per round (× salvo_size per engagement)
    min_range_km: float = 0.0               # kinematics.min_range_km
    max_altitude_m: float = 100_000.0       # kinematics.max_target_altitude_ft * 0.3048
    salvo_size: int = 1                     # logistics.salvo_policy_default
    cooldown_ticks: int = 0                 # logistics.cooldown_penalty_ms / 5000
    weather_degradation: bool = False       # tewa_constraints.weather_degradation

    def pk(self, threat: Threat) -> float:
        return PK_TABLE[self.effector_type][threat.threat_class]

    def effective_pk(self, threat: Threat) -> float:
        """Combined PK accounting for salvo policy and weather degradation."""
        pk = self.pk(threat)
        if self.salvo_size > 1:
            pk = 1.0 - (1.0 - pk) ** self.salvo_size
        if self.weather_degradation:
            pk *= 0.85   # ~15% degradation in adverse conditions (prototype constant)
        return min(pk, 0.99)

    def in_range(self, threat: Threat) -> bool:
        dist = haversine(self.lat, self.lon, threat.lat, threat.lon)
        altitude_ok = threat.altitude_m <= self.max_altitude_m
        range_ok    = self.min_range_km <= dist <= self.range_km
        return altitude_ok and range_ok

    def can_engage(self) -> bool:
        return (self.status == AssetStatus.READY and
                (self.ammo == -1 or self.ammo >= self.salvo_size))

    def tick_reload(self):
        if self.status == AssetStatus.RELOADING:
            self.reload_ticks_remaining -= 1
            if self.reload_ticks_remaining <= 0:
                self.status = AssetStatus.READY

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "effector_type": self.effector_type.value,
            "lat": round(self.lat, 5),
            "lon": round(self.lon, 5),
            "ammo": self.ammo,
            "max_ammo": self.max_ammo,
            "range_km": self.range_km,
            "status": self.status.value,
            "reload_ticks_remaining": self.reload_ticks_remaining,
            "base_id": self.base_id,
            "assigned_threat_id": self.assigned_threat_id,
            "cost_per_shot_usd": self.cost_per_shot_usd,
            "min_range_km": self.min_range_km,
            "max_altitude_m": self.max_altitude_m,
            "salvo_size": self.salvo_size,
            "weather_degradation": self.weather_degradation,
        }


@dataclass
class Base:
    """A military installation hosting multiple effectors."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str = ""
    lat: float = 0.0
    lon: float = 0.0
    effectors: list[Effector] = field(default_factory=list)
    operational: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "lat": round(self.lat, 5),
            "lon": round(self.lon, 5),
            "operational": self.operational,
            "effector_ids": [e.id for e in self.effectors],
            "ready_count": sum(1 for e in self.effectors if e.can_engage()),
        }


@dataclass
class EngagementOrder:
    """A commit order: effector → threat."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    effector_id: str = ""
    threat_id: str = ""
    effector_type: str = ""
    effector_label: str = ""
    base_id: str = ""
    base_label: str = ""
    threat_label: str = ""
    threat_class: str = ""
    pk: float = 0.0
    distance_km: float = 0.0
    time_to_intercept_s: float = 0.0
    rationale: str = ""
    timestamp: float = 0.0
    outcome: Optional[str] = None   # "hit" | "miss" | "pending"
    cost_usd: float = 0.0           # engagement cost (salvo × cost_per_shot)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "effector_id": self.effector_id,
            "threat_id": self.threat_id,
            "effector_type": self.effector_type,
            "effector_label": self.effector_label,
            "base_id": self.base_id,
            "base_label": self.base_label,
            "threat_label": self.threat_label,
            "threat_class": self.threat_class,
            "pk": round(self.pk, 3),
            "distance_km": round(self.distance_km, 1),
            "time_to_intercept_s": round(self.time_to_intercept_s, 1),
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "outcome": self.outcome,
            "cost_usd": self.cost_usd,
        }
