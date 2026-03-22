"""
Radar Kalman Filter – Sensor Fusion for Threat Track Estimation
================================================================
Based on the Gemini-proposed RadarKalmanFilter spec.

The state vector is 4-dimensional:  [x, y, vx, vy]
  • x, y   – position in km relative to a fixed reference point
  • vx, vy – velocity in km/s

Working in km-space (not lat/lon degrees) keeps the matrix values
numerically stable and the physics intuitive.

Two-step loop every radar update (5-second tick):
  1. predict()  – extrapolate forward using equations of motion (F matrix)
  2. update()   – correct with a noisy radar measurement (Kalman gain K)

After each cycle the filter exposes:
  • estimated position in lat/lon
  • estimated speed (km/h)
  • 1-sigma position uncertainty (km)
  • track_quality [0.0–1.0]
"""
from __future__ import annotations
import math
import numpy as np

# ---------------------------------------------------------------------------
# Flat-Earth reference point (centre of Norway scenario)
# ---------------------------------------------------------------------------
REF_LAT = 65.0   # degrees N
REF_LON = 15.0   # degrees E

_KM_PER_DEG_LAT = 111.0
_KM_PER_DEG_LON = 111.0 * math.cos(math.radians(REF_LAT))   # ~46.8 km/°


def _latlon_to_xy(lat: float, lon: float) -> tuple[float, float]:
    """Convert (lat, lon) degrees to (x, y) km offset from reference point."""
    x = (lon - REF_LON) * _KM_PER_DEG_LON
    y = (lat - REF_LAT) * _KM_PER_DEG_LAT
    return x, y


def _xy_to_latlon(x: float, y: float) -> tuple[float, float]:
    """Convert (x, y) km offset back to (lat, lon) degrees."""
    lat = REF_LAT + y / _KM_PER_DEG_LAT
    lon = REF_LON + x / _KM_PER_DEG_LON
    return lat, lon


class RadarKalmanFilter:
    """
    Constant-velocity Kalman filter for radar track estimation.

    Parameters
    ----------
    dt : float
        Time step in seconds (default 5.0 = one simulation tick).
    radar_noise_km : float
        Standard deviation of radar position measurement in km.
        Gemini spec used 50m for a precision radar; we use 0.5 km (500 m)
        which is realistic for a long-range surveillance radar.
    process_noise : float
        Q scaling factor – how much we expect the target to manoeuvre.
        Higher values allow tracking evasive targets but make tracks jumpier.
    """

    def __init__(self, dt: float = 5.0, radar_noise_km: float = 0.5, process_noise: float = 0.001):
        self.dt = dt

        # ── 1. State vector [x, y, vx, vy]  (km, km, km/s, km/s)
        self.x = np.zeros((4, 1))

        # ── 2. State Transition Matrix (F) – constant-velocity physics
        #       new_x = old_x + vx*dt  (same pattern for y)
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        # ── 3. Measurement Matrix (H) – radar only sees position, not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # ── 4. Estimate Covariance (P) – start with high uncertainty
        self.P = np.eye(4) * 100.0

        # ── 5. Measurement Noise (R) – radar position noise
        #       Gemini spec: "50 metres of radar noise variance"
        #       We use 0.5 km radar noise std → variance = 0.25 km²
        self.R = np.eye(2) * (radar_noise_km ** 2)

        # ── 6. Process Noise (Q) – target manoeuvre uncertainty
        #       High Q = tracks evasive hypersonics / low-observable targets
        self.Q = np.eye(4) * process_noise

        self.initialized: bool = False
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def initialise(self, lat: float, lon: float, speed_kmh: float, heading_deg: float):
        """Warm-start the filter from a known initial threat state."""
        x, y = _latlon_to_xy(lat, lon)
        speed_km_s = speed_kmh / 3600.0
        heading_rad = math.radians(heading_deg)
        vx = math.sin(heading_rad) * speed_km_s
        vy = math.cos(heading_rad) * speed_km_s

        self.x = np.array([[x], [y], [vx], [vy]])
        self.P = np.eye(4) * 1.0   # moderate initial confidence after warm-start
        self.initialized = True

    # ------------------------------------------------------------------
    # Predict step: extrapolate state forward using physics
    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        """Math:  x = F·x   |   P = F·P·Fᵀ + Q"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    # ------------------------------------------------------------------
    # Update step: correct prediction using a radar measurement
    # ------------------------------------------------------------------
    def update(self, lat_meas: float, lon_meas: float) -> np.ndarray:
        """
        Fuse a noisy radar measurement into the track estimate.

        Math:
          y = z - H·x                        (Innovation)
          S = H·P·Hᵀ + R                     (Innovation covariance)
          K = P·Hᵀ·S⁻¹                       (Kalman gain)
          x = x + K·y                         (State update)
          P = (I - K·H)·P                     (Covariance update)
        """
        z = np.array(_latlon_to_xy(lat_meas, lon_meas)).reshape(2, 1)

        y = z - self.H @ self.x                         # Innovation
        S = self.H @ self.P @ self.H.T + self.R         # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain

        self.x = self.x + K @ y                         # State update
        self.P = (np.eye(4) - K @ self.H) @ self.P      # Covariance update

        self._update_count += 1
        return self.x.flatten()

    # ------------------------------------------------------------------
    # Derived outputs
    # ------------------------------------------------------------------
    def estimated_latlon(self) -> tuple[float, float]:
        return _xy_to_latlon(float(self.x[0, 0]), float(self.x[1, 0]))

    def estimated_speed_kmh(self) -> float:
        vx, vy = float(self.x[2, 0]), float(self.x[3, 0])
        return math.sqrt(vx ** 2 + vy ** 2) * 3600.0

    def estimated_heading_deg(self) -> float:
        vx, vy = float(self.x[2, 0]), float(self.x[3, 0])
        return math.degrees(math.atan2(vx, vy)) % 360

    def position_uncertainty_km(self) -> float:
        """1-sigma position uncertainty (combined x+y) in km."""
        return math.sqrt(max(0.0, float(self.P[0, 0]) + float(self.P[1, 1])))

    def track_quality(self) -> float:
        """
        0.0 (no track / highly uncertain) to 1.0 (tight, confident track).
        Derived from position covariance: quality saturates at <0.1 km uncertainty.
        """
        unc = self.position_uncertainty_km()
        return max(0.0, min(1.0, 1.0 - unc / 5.0))
