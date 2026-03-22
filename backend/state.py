"""
Scenario State Manager
=======================
Owns the live simulation state: bases, effectors, threats, orders, log.
Provides thread-safe mutators used by the simulation loop and API handlers.
"""
from __future__ import annotations
import math
import random
import time
import threading
from typing import Optional

from backend.models import (
    Base, Effector, Threat, EngagementOrder,
    ThreatClass, ThreatStatus, EffectorType, AssetStatus,
    THREAT_SPEED, RELOAD_TICKS, haversine
)
from backend.engine.kalman import RadarKalmanFilter


# ---------------------------------------------------------------------------
# Default scenario: Norwegian air-defense exercise
# Swedish Air Force bases (Flygvapnet)
# ---------------------------------------------------------------------------
SCENARIO_BASES = [
    {
        "id": "base-lulea",
        "label": "F 21 Luleå-Kallax Wing",
        "lat": 65.54, "lon": 22.12,   # northernmost fighter wing
        "effectors": [
            {"type": EffectorType.MISSILE,  "label": "GBAD Bty Alpha",   "range_km": 180, "ammo": 8,  "max_ammo": 8,  "cost_per_shot_usd": 500_000},
            {"type": EffectorType.MISSILE,  "label": "GBAD Bty Bravo",   "range_km": 180, "ammo": 8,  "max_ammo": 8,  "cost_per_shot_usd": 500_000},
            {"type": EffectorType.AIRCRAFT, "label": "JAS-39 Sqn 1",     "range_km": 800, "ammo": 8,  "max_ammo": 8,  "cost_per_shot_usd": 120_000},
            {"type": EffectorType.AIRCRAFT, "label": "JAS-39 Sqn 2",     "range_km": 800, "ammo": 8,  "max_ammo": 8,  "cost_per_shot_usd": 120_000},
        ],
    },
    {
        "id": "base-Uppsala",
        "label": "F 16 Uppsala Wing",
        "lat": 59.90, "lon": 17.59,   # central Sweden, near Stockholm
        "effectors": [
            {"type": EffectorType.MISSILE,  "label": "GBAD Bty Charlie", "range_km": 180, "ammo": 8,  "max_ammo": 8,  "cost_per_shot_usd": 500_000},
            {"type": EffectorType.ANTI_AIR, "label": "SHORAD Plt 1",     "range_km": 40,  "ammo": 20, "max_ammo": 20, "cost_per_shot_usd": 30_000},
            {"type": EffectorType.AIRCRAFT, "label": "JAS-39 Sqn 3",     "range_km": 800, "ammo": 8,  "max_ammo": 8,  "cost_per_shot_usd": 120_000},
            {"type": EffectorType.DRONE,    "label": "Heron TP Flt A",   "range_km": 300, "ammo":-1,  "max_ammo":-1,  "cost_per_shot_usd": 5_000},
        ],
    },
    {
        "id": "base-blekinge",
        "label": "F 17 Blekinge Wing",
        "lat": 56.27, "lon": 15.27,   # southern coastal wing (Ronneby)
        "effectors": [
            {"type": EffectorType.MISSILE,  "label": "GBAD Bty Delta",   "range_km": 180, "ammo": 8,  "max_ammo": 8,  "cost_per_shot_usd": 500_000},
            {"type": EffectorType.ANTI_AIR, "label": "SHORAD Plt 2",     "range_km": 40,  "ammo": 20, "max_ammo": 20, "cost_per_shot_usd": 30_000},
            {"type": EffectorType.DRONE,    "label": "Heron TP Flt B",   "range_km": 300, "ammo":-1,  "max_ammo":-1,  "cost_per_shot_usd": 5_000},
            {"type": EffectorType.AIRCRAFT, "label": "JAS-39 Sqn 4",     "range_km": 800, "ammo": 8,  "max_ammo": 8,  "cost_per_shot_usd": 120_000},
        ],
    },
    {
        "id": "base-satenas",
        "label": "F 7 Såtenäs Wing",
        "lat": 58.43, "lon": 12.72,   # western Sweden, near Gothenburg
        "effectors": [
            {"type": EffectorType.MISSILE,         "label": "Patriot Bty 1",    "range_km": 250, "ammo": 6,  "max_ammo": 6,  "cost_per_shot_usd": 3_000_000},
            {"type": EffectorType.ANTI_AIR,        "label": "SHORAD Plt 3",     "range_km": 40,  "ammo": 20, "max_ammo": 20, "cost_per_shot_usd": 30_000},
            # Gemini spec: KIN-PAC3-MSE – salvo_policy=2, reload=3600s, base_pk=0.85
            {"type": EffectorType.KINETIC_MISSILE, "label": "PAC-3 MSE Bty 1",  "range_km": 120, "ammo": 16, "max_ammo": 16,
             "cost_per_shot_usd": 4_000_000, "min_range_km": 5, "max_altitude_m": 30_480, "salvo_size": 2, "weather_degradation": False},
            # Gemini spec: DEW-HELIOS-300 – 300kW laser, $15/shot, weather_degradation=true, max_alt 20000ft
            {"type": EffectorType.DIRECTED_ENERGY, "label": "HELIOS-300 #1",    "range_km": 12,  "ammo": -1,  "max_ammo": -1,
             "cost_per_shot_usd": 15, "min_range_km": 0.1, "max_altitude_m": 6_096, "salvo_size": 1, "cooldown_ticks": 1, "weather_degradation": True},
        ],
    },
]

# Threat spawn origins – surrounding Sweden, within engagement range of bases
SPAWN_ZONE = [
    # --- EAST (Baltic approaches, within JAS-39 + GBAD range) ---
    {"lat": 57.5, "lon": 21.0, "label": "Baltic east"},
    {"lat": 59.0, "lon": 22.0, "label": "Gulf of Finland west"},
    {"lat": 62.0, "lon": 24.0, "label": "Finnish border E"},
    {"lat": 65.0, "lon": 22.5, "label": "Kola approach"},
    {"lat": 68.0, "lon": 21.0, "label": "Barents approach"},
    # --- NORTH (Arctic, over northern Norway) ---
    {"lat": 71.0, "lon": 18.0, "label": "Arctic N"},
    {"lat": 70.0, "lon": 12.0, "label": "Arctic NW"},
    # --- WEST (North Sea, within JAS-39 range) ---
    {"lat": 61.0, "lon":  5.0, "label": "Norwegian Sea W"},
    {"lat": 57.5, "lon":  6.5, "label": "North Sea W"},
    {"lat": 55.5, "lon":  8.5, "label": "North Sea SW"},
    # --- SKAGERRAK / KATTEGAT (within F7 Såtenäs Patriot/PAC-3 MSE range ~195-225km) ---
    # These create a western approach corridor where F7 is the FIRST line of defence
    {"lat": 58.0, "lon":  9.5, "label": "Skagerrak W"},     # 195km from F7
    {"lat": 57.0, "lon": 10.5, "label": "Kattegat N"},      # 205km from F7
    {"lat": 56.5, "lon": 12.0, "label": "Kattegat S"},      # 218km from F7
    # --- SOUTH (Denmark/Baltic, close enough for ground effectors) ---
    {"lat": 54.5, "lon": 11.0, "label": "Southern Baltic"},
    {"lat": 55.0, "lon": 15.0, "label": "Baltic south"},
    {"lat": 54.2, "lon": 18.0, "label": "Gdansk axis"},
]

# Protected targets – major Swedish cities and infrastructure
TARGETS = [
    {"lat": 59.33, "lon": 18.07, "label": "Stockholm"},
    {"lat": 57.71, "lon": 11.97, "label": "Gothenburg"},
    {"lat": 55.61, "lon": 13.00, "label": "Malmö"},
    {"lat": 67.86, "lon": 20.23, "label": "Kiruna"},
]


class ScenarioState:
    """Thread-safe singleton managing all simulation state."""

    def __init__(self):
        self._lock = threading.RLock()
        self.bases: dict[str, Base] = {}
        self.effectors: dict[str, Effector] = {}
        self.threats: dict[str, Threat] = {}
        self.orders: list[EngagementOrder] = []
        self.event_log: list[dict] = []
        self.tick: int = 0
        self.running: bool = False
        self._threat_counter = 0
        self.total_cost_usd: float = 0.0   # cumulative engagement cost
        self._ballistic_marv_countdown: dict[str, int] = {}  # ticks until next MaRV check
        self._marv_cancelled: list[str] = []   # threat IDs whose engagements were voided by MaRV
        self._initialise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialise(self):
        for bdata in SCENARIO_BASES:
            base = Base(id=bdata["id"], label=bdata["label"],
                        lat=bdata["lat"], lon=bdata["lon"])
            for edata in bdata["effectors"]:
                eff = Effector(
                    label=edata["label"],
                    effector_type=edata["type"],
                    lat=bdata["lat"] + random.uniform(-0.05, 0.05),
                    lon=bdata["lon"] + random.uniform(-0.05, 0.05),
                    ammo=edata["ammo"],
                    max_ammo=edata["max_ammo"],
                    range_km=edata["range_km"],
                    status=AssetStatus.READY,
                    base_id=base.id,
                    # Optional fields from Gemini spec (fall back to dataclass defaults)
                    cost_per_shot_usd=edata.get("cost_per_shot_usd", 0.0),
                    min_range_km=edata.get("min_range_km", 0.0),
                    max_altitude_m=edata.get("max_altitude_m", 100_000.0),
                    salvo_size=edata.get("salvo_size", 1),
                    cooldown_ticks=edata.get("cooldown_ticks", 0),
                    weather_degradation=edata.get("weather_degradation", False),
                )
                base.effectors.append(eff)
                self.effectors[eff.id] = eff
            self.bases[base.id] = base

    # ------------------------------------------------------------------
    # Accessors (return copies of current state for broadcasting)
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "tick": self.tick,
                "timestamp": time.time(),
                "threats": [t.to_dict() for t in self.threats.values()],
                "effectors": [e.to_dict() for e in self.effectors.values()],
                "bases": [b.to_dict() for b in self.bases.values()],
                "recent_orders": [o.to_dict() for o in self.orders[-20:]],
                "cost_log": [o.to_dict() for o in self.orders],
                "event_log": self.event_log[-30:],
                "total_cost_usd": self.total_cost_usd,
                "base_readiness": self._compute_base_readiness(),
            }

    def get_threats(self) -> list[Threat]:
        with self._lock:
            return list(self.threats.values())

    def get_effectors(self) -> list[Effector]:
        with self._lock:
            return list(self.effectors.values())

    def _compute_base_readiness(self) -> list[dict]:
        """Returns per-base readiness metrics for the tactical dashboard."""
        result = []
        for base in self.bases.values():
            effs = [e for e in self.effectors.values() if e.base_id == base.id]
            if not effs:
                continue
            total = len(effs)
            ready = sum(1 for e in effs if e.status == AssetStatus.READY)
            finite_effs = [e for e in effs if e.ammo != -1]
            if finite_effs:
                ammo_sum = sum(e.ammo for e in finite_effs)
                max_sum  = sum(e.max_ammo for e in finite_effs)
                ammo_ratio = ammo_sum / max(max_sum, 1)
            else:
                ammo_ratio = 1.0   # unlimited-ammo effectors always "full"

            if ready == 0:
                status = "BLACK"
            elif ammo_ratio < 0.20 or (ready == 1 and total > 2):
                status = "RED"
            elif ammo_ratio < 0.50 or ready < (total + 1) // 2:
                status = "AMBER"
            else:
                status = "GREEN"

            result.append({
                "base_id":    base.id,
                "label":      base.label,
                "total":      total,
                "ready":      ready,
                "ammo_ratio": round(ammo_ratio, 2),
                "status":     status,
            })
        return result

    # ------------------------------------------------------------------
    # Threat management
    # ------------------------------------------------------------------
    def spawn_threat(
        self,
        threat_class: Optional[ThreatClass] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        priority: Optional[int] = None,
        target_idx: Optional[int] = None,
    ) -> Threat:
        with self._lock:
            self._threat_counter += 1
            tc = threat_class or random.choice(list(ThreatClass))
            origin = random.choice(SPAWN_ZONE)
            tlat = lat if lat is not None else origin["lat"] + random.uniform(-1.5, 1.5)
            tlon = lon if lon is not None else origin["lon"] + random.uniform(-2.5, 2.5)
            tgt = TARGETS[target_idx or random.randrange(len(TARGETS))]
            # Ballistic missiles are always treated as top-priority threats
            pri = priority if priority is not None else (10 if tc == ThreatClass.BALLISTIC else random.randint(4, 9))

            threat = Threat(
                label=f"T{self._threat_counter:03d}",
                threat_class=tc,
                lat=tlat,
                lon=tlon,
                speed_kmh=THREAT_SPEED[tc] * random.uniform(0.85, 1.10),
                altitude_m=self._random_altitude(tc),
                priority=pri,
                origin_lat=tlat,
                origin_lon=tlon,
                target_lat=tgt["lat"],
                target_lon=tgt["lon"],
                target_label=tgt["label"],
            )
            # Set initial heading
            dy = threat.target_lat - threat.lat
            dx = (threat.target_lon - threat.lon) * math.cos(math.radians(threat.lat))
            threat.heading = math.degrees(math.atan2(dx, dy)) % 360

            # Warm-start the Kalman filter from the true initial state
            kf = RadarKalmanFilter(dt=5.0)
            kf.initialise(threat.lat, threat.lon, threat.speed_kmh, threat.heading)
            threat.kalman_filter = kf
            threat.estimated_lat = threat.lat
            threat.estimated_lon = threat.lon
            threat.estimated_speed_kmh_kf = threat.speed_kmh
            threat.track_quality = 0.5   # moderate initial confidence

            self.threats[threat.id] = threat
            self._log(f"NEW THREAT {threat.label} [{tc.value}] pri={pri} → {tgt['label']}", "THREAT")
            return threat

    @staticmethod
    def _random_altitude(tc: ThreatClass) -> float:
        return {
            ThreatClass.DRONE:              random.uniform(50,    500),
            ThreatClass.CRUISE:             random.uniform(30,    200),
            ThreatClass.FIGHTER:            random.uniform(3000, 12000),
            ThreatClass.BALLISTIC:          random.uniform(5000, 25000),   # terminal intercept phase
            ThreatClass.LOITERING_MUNITION: random.uniform(50,    300),
        }[tc]

    def remove_threat(self, threat_id: str):
        with self._lock:
            self.threats.pop(threat_id, None)
            self._ballistic_marv_countdown.pop(threat_id, None)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def execute_order(self, order: EngagementOrder):
        """Apply an engagement order: consume ammo, update statuses."""
        with self._lock:
            eff = self.effectors.get(order.effector_id)
            thr = self.threats.get(order.threat_id)
            if not eff or not thr:
                return

            # Stamp cost + base info onto the order before storing it
            engagement_cost = eff.cost_per_shot_usd * eff.salvo_size
            order.cost_usd = engagement_cost
            order.base_id = eff.base_id
            base = self.bases.get(eff.base_id)
            order.base_label = base.label if base else eff.base_id

            # Mark effector as engaged
            eff.status = AssetStatus.ENGAGED
            eff.assigned_threat_id = thr.id
            if eff.ammo != -1:
                eff.ammo = max(0, eff.ammo - eff.salvo_size)   # consume full salvo
            self.total_cost_usd += engagement_cost

            thr.status = ThreatStatus.ENGAGED
            self.orders.append(order)
            self._log(
                f"ENGAGE {eff.label} → {thr.label} | PK={order.pk:.0%} | {order.rationale}",
                "ORDER"
            )

    def resolve_engagement(self, order: EngagementOrder):
        """Resolve outcome after intercept delay."""
        with self._lock:
            eff = self.effectors.get(order.effector_id)
            thr = self.threats.get(order.threat_id)
            hit = random.random() < order.pk

            order.outcome = "hit" if hit else "miss"
            if hit and thr:
                thr.status = ThreatStatus.NEUTRALIZED
                self._log(f"HIT ✓ {eff.label if eff else '?'} → {thr.label} neutralized", "HIT")
            else:
                if thr:
                    thr.status = ThreatStatus.ACTIVE  # back to active for re-engagement
                self._log(f"MISS ✗ {eff.label if eff else '?'} missed {thr.label if thr else '?'} – reassign!", "MISS")

            if eff:
                reload_t = RELOAD_TICKS.get(eff.effector_type, 4)
                if eff.ammo == 0:
                    # Aircraft and drones rearm; missiles/AA batteries are depleted
                    if eff.effector_type in (EffectorType.AIRCRAFT, EffectorType.DRONE):
                        eff.ammo = eff.max_ammo   # rearm
                        eff.status = AssetStatus.RELOADING
                        eff.reload_ticks_remaining = reload_t * 2  # full rearm = 2× sortie time
                    else:
                        eff.status = AssetStatus.DESTROYED  # missile batteries stay depleted
                elif reload_t > 0:
                    eff.status = AssetStatus.RELOADING
                    eff.reload_ticks_remaining = reload_t
                else:
                    eff.status = AssetStatus.READY
                eff.assigned_threat_id = None

    def consume_marv_cancellations(self) -> list[str]:
        """Return threat IDs whose in-flight engagements were voided by MaRV; clears list."""
        with self._lock:
            cancelled = list(self._marv_cancelled)
            self._marv_cancelled.clear()
            return cancelled

    def tick_state(self):
        """Advance all ticking elements by one simulation step (5s)."""
        with self._lock:
            self.tick += 1
            escaped = []
            for thr in list(self.threats.values()):
                # All active/engaged threats keep flying – they don't freeze when targeted
                in_flight = thr.status in (ThreatStatus.ACTIVE, ThreatStatus.ENGAGED)
                if in_flight:
                    thr.move(5.0)
                    dist = haversine(thr.lat, thr.lon, thr.target_lat, thr.target_lon)
                    if dist < 5.0:  # reached target
                        thr.status = ThreatStatus.ESCAPED
                        escaped.append(thr)
                        self._log(f"BREACH! {thr.label} reached target area!", "BREACH")
                        continue

                # ── MaRV (Manoeuvring Re-entry Vehicle) evasion ──────────────
                # Each ballistic has a countdown; when it hits zero there is a
                # chance it retargets, forcing the system to reassign immediately.
                if thr.threat_class == ThreatClass.BALLISTIC and in_flight:
                    cd = self._ballistic_marv_countdown.get(thr.id, random.randint(8, 15))
                    cd -= 1
                    if cd <= 0:
                        if random.random() < 0.60:
                            alt_targets = [
                                t for t in TARGETS
                                if not (abs(t["lat"] - thr.target_lat) < 0.5 and
                                        abs(t["lon"] - thr.target_lon) < 0.5)
                            ]
                            if alt_targets:
                                new_tgt         = random.choice(alt_targets)
                                old_label        = thr.target_label or "?"
                                thr.target_lat   = new_tgt["lat"]
                                thr.target_lon   = new_tgt["lon"]
                                thr.target_label = new_tgt["label"]
                                self._log(
                                    f"⚠ MaRV EVASION: {thr.label} retargeted "
                                    f"{old_label} → {new_tgt['label']} — REASSIGN!",
                                    "THREAT"
                                )
                                if thr.status == ThreatStatus.ENGAGED:
                                    # Void the in-flight engagement: free the assigned effector
                                    for eff in self.effectors.values():
                                        if eff.assigned_threat_id == thr.id:
                                            eff.status = AssetStatus.RELOADING
                                            eff.reload_ticks_remaining = 2
                                            eff.assigned_threat_id = None
                                            break
                                    thr.status = ThreatStatus.ACTIVE
                                    self._marv_cancelled.append(thr.id)
                                cd = random.randint(10, 18)
                        else:
                            cd = random.randint(6, 12)
                    self._ballistic_marv_countdown[thr.id] = cd

            # Tick effector reloads
            for eff in self.effectors.values():
                eff.tick_reload()

    def tick_kalman(self):
        """
        Kalman predict → noisy radar measurement → update for every active threat.
        Called each simulation tick AFTER tick_state() moves the true position.
        """
        with self._lock:
            for thr in self.threats.values():
                if thr.status != ThreatStatus.ACTIVE or thr.kalman_filter is None:
                    continue
                kf: RadarKalmanFilter = thr.kalman_filter

                # ─ Step 1: Predict (physics extrapolation) ─
                kf.predict()

                # ─ Step 2: Simulate noisy radar return ─
                # 500 m std (0.5 km) surveillance radar noise — matches Gemini spec
                noise_lat = random.gauss(0, 0.5 / 111.0)
                noise_lon = random.gauss(0, 0.5 / (111.0 * math.cos(math.radians(thr.lat))))
                thr.radar_lat = thr.lat + noise_lat
                thr.radar_lon = thr.lon + noise_lon

                # ─ Step 3: Update (sensor fusion) ─
                kf.update(thr.radar_lat, thr.radar_lon)

                # ─ Step 4: Write KF outputs back to the threat ─
                est_lat, est_lon = kf.estimated_latlon()
                thr.estimated_lat = est_lat
                thr.estimated_lon = est_lon
                thr.estimated_speed_kmh_kf = kf.estimated_speed_kmh()
                thr.track_quality = kf.track_quality()
                thr.position_uncertainty_km = kf.position_uncertainty_km()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str, category: str = "INFO"):
        entry = {
            "tick": self.tick,
            "time": time.strftime("%H:%M:%S"),
            "category": category,
            "msg": msg,
        }
        self.event_log.append(entry)
        if len(self.event_log) > 500:
            self.event_log = self.event_log[-500:]

    def add_order(self, order: EngagementOrder):
        with self._lock:
            self.orders.append(order)
