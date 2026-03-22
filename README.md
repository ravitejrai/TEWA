# TEWA — Threat Evaluation & Weapon Assignment C2 System

Intelligent decision support system for air defense — built for the Swedish Air Force scenario.

## Features

- **Real-time tactical map** (Leaflet.js, dark CartoDB) with live threat tracks and effector ranges
- **Kalman filter** track quality estimation per threat
- **Hungarian algorithm** optimal weapon-to-threat assignment
- **Multi-class threats**: ballistic, cruise, drone, fighter, loitering munition — each with distinct speeds, altitudes and PK profiles
- **16 effectors** across 4 Swedish bases (F21 Luleå, F17 Kallinge, F7 Såtenäs, F10 Ängelholm)
- **Shoot-Look-Shoot (SLS) doctrine**: ground SAMs engage in-flight ballistics as a layer-2 kill-chain
- **MaRV retargeting**: ballistic missiles randomly change target mid-flight, cancelling in-flight engagements
- **Double-tap doctrine**: P9-10 threats with PK < 80% auto-queue a second effector
- **Strategic reserve policy**: withholds high-value assets (JAS-39, PAC-3 MSE) from low-priority work
- **Cost tracking**: per-engagement and cumulative USD cost log with breakdown modal
- **Base readiness panel**: GREEN/AMBER/RED/BLACK status per base with ammo bars
- **Threat trajectory lines**: dashed polylines from current KF position to target city

## Architecture

```
air-defense-c2/
├── main.py                   # FastAPI app entry point
├── backend/
│   ├── api.py                # REST endpoints + WebSocket
│   ├── models.py             # Threat, Effector, EngagementOrder dataclasses
│   ├── state.py              # ScenarioState — all sim state + spawn logic
│   ├── simulation.py         # Background tick loop (3s real / 5s sim per tick)
│   └── engine/
│       ├── kalman.py         # 4D flat-Earth Kalman filter
│       └── optimizer.py      # Composite scorer + Hungarian assignment + SLS
└── frontend/
    └── index.html            # Single-page tactical map UI
```

## Quick Start

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn numpy scipy

# Run
uvicorn main:app --host 0.0.0.0 --port 8000

# Open in browser
open http://localhost:8000
```

## Swedish Bases & Effectors

| Base | Effectors |
|---|---|
| F21 Luleå Wing | JAS-39 Sqn 1 & 2, Patriot Bty 2, GBAD Bty Alpha |
| F17 Kallinge Wing | JAS-39 Sqn 3 & 4, Patriot Bty 3, GBAD Bty Bravo |
| F7 Såtenäs Wing | Heron TP Flt A & B, Patriot Bty 1, PAC-3 MSE Bty 1, SHORAD Bty 1, HELIOS-300 |
| F10 Ängelholm Wing | JAS-39 Sqn 5 & 6, Patriot Bty 4, GBAD Bty Delta |

## Spawn Zones (16 total)

Threats spawn from 16 zones covering all approach corridors: Baltic Sea, Gulf of Finland, Gulf of Bothnia, North Sea, Skagerrak, Kattegat, and eastern land approaches.

## Threat Classes

| Class | Speed | Notes |
|---|---|---|
| Ballistic | ~5000 km/h | P10, terminal phase 5-25km alt, MaRV evasion |
| Cruise | ~850 km/h | Smart altitude variation, weather-sensitive |
| Fighter | ~1800 km/h | P7-9, high maneuverability penalty |
| Drone | ~120 km/h | Low PK for most effectors |
| Loitering munition | ~180 km/h | Small RCS, SHORAD/EW optimised |
